"""
src/rag/retriever.py
---------------------
Hybrid retriever: FAISS (vector) + BM25 (keyword) → RRF merge → parent lookup.

The key insight from fin-rag-lab §4.5:
  - RETRIEVE using child chunks (precise semantic match)
  - RETURN parent chunks to the LLM (full context for generation)
"""

from __future__ import annotations
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from rank_bm25 import BM25Okapi

from src.rag.ingest import DocumentChunk, rrf_merge, INDEX_DIR



@dataclass
class RetrievalResult:
    """What the generator receives: parent chunk text + citation metadata."""
    text: str            # parent chunk text (full context)
    ticker: str
    doc_type: str        # "10K" or "earnings"
    page: int
    source_pdf: str
    relevance_score: float
    child_chunk_id: str  # the child that triggered retrieval (for tracing)


class FinancialRetriever:
    """
    Hybrid retriever for financial documents.
    
    Implements the two-stage pattern from fin-rag-lab:
    1. Recall-focused: hybrid BM25 + vector retrieval (wide net)
    2. Precision via RRF merge (re-rank without a separate reranker)
    3. Parent lookup: return full parent context, not just the matched child
    """

    def __init__(self, ticker: str, index_dir: Path = INDEX_DIR):
        self.ticker = ticker
        self._load_index(index_dir / ticker)

    def _load_index(self, ticker_dir: Path) -> None:
        import src.rag.ingest
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.faiss_store = FAISS.load_local(
            str(ticker_dir / "faiss"),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        import json
        with open(ticker_dir / "metadata.json", "r") as f:
            meta = json.load(f)

        from src.rag.ingest import DocumentChunk
        self.parent_map = {
           k: DocumentChunk(
                chunk_id=v["chunk_id"],
                parent_id=v["parent_id"],
                text=v["text"],
                metadata=v["metadata"],
            ) for k, v in meta["parent_map"].items()
        }
        self.child_chunks = [
            DocumentChunk(
                chunk_id=c["chunk_id"],
                parent_id=c["parent_id"],
                text=c["text"],
                metadata=c["metadata"],
            ) for c in meta["child_chunks"]
        ]
        self.bm25_corpus = meta["bm25_corpus"]
        self.bm25 = BM25Okapi(self.bm25_corpus)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """
        Hybrid retrieval: vector + BM25 → RRF → parent lookup.
        Returns top_k parent chunks for the LLM context.
        """
        # 1. Vector retrieval (top 20 for recall)
        vector_docs = self.faiss_store.similarity_search_with_score(query, k=20)

        # 2. BM25 retrieval (top 20)
        tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokens)
        top_bm25_idx = np.argsort(bm25_scores)[::-1][:20]
        bm25_docs = [
            (self._idx_to_doc(i), float(bm25_scores[i]))
            for i in top_bm25_idx
        ]

        # 3. RRF merge
        merged = rrf_merge(vector_docs, bm25_docs, top_n=top_k * 2)

        # 4. Parent lookup: deduplicate by parent_id, return parent text
        seen_parents: set[str] = set()
        results: list[RetrievalResult] = []

        for doc, score in merged:
            child_id = doc.metadata.get("chunk_id", "")
            parent_id = doc.metadata.get("parent_id", child_id)

            if parent_id in seen_parents:
                continue
            seen_parents.add(parent_id)

            parent = self.parent_map.get(parent_id)
            if parent is None:
                continue

            results.append(RetrievalResult(
                text=parent.text,
                ticker=self.ticker,
                doc_type=parent.metadata.get("doc_type", "unknown"),
                page=parent.metadata.get("page", 0),
                source_pdf=parent.metadata.get("source_pdf", ""),
                relevance_score=score,
                child_chunk_id=child_id,
            ))

            if len(results) >= top_k:
                break

        return results

    def _idx_to_doc(self, idx: int):
        """Convert BM25 index position to a LangChain Document-like object."""
        from langchain_core.documents import Document
        chunk = self.child_chunks[idx]
        return Document(
            page_content=chunk.text,
            metadata={"chunk_id": chunk.chunk_id, "parent_id": chunk.parent_id,
                      **chunk.metadata},
        )


class MultiTickerRetriever:
    """
    Retrieves across multiple tickers simultaneously.
    Used by the Analyst agent to gather context for a given stock.
    """

    def __init__(self, tickers: list[str]):
        self.retrievers = {}
        for ticker in tickers:
            try:
                self.retrievers[ticker] = FinancialRetriever(ticker)
            except Exception as e:
                print(f"Warning: could not load index for {ticker}: {e}")

    def retrieve_for_ticker(
        self, ticker: str, query: str, top_k: int = 5
    ) -> list[RetrievalResult]:
        """Retrieve docs for a specific ticker."""
        if ticker not in self.retrievers:
            return []
        return self.retrievers[ticker].retrieve(query, top_k)

    def retrieve_cross_ticker(
        self, query: str, top_k_per_ticker: int = 3
    ) -> dict[str, list[RetrievalResult]]:
        """Retrieve across all tickers (for portfolio-level queries)."""
        return {
            ticker: r.retrieve(query, top_k_per_ticker)
            for ticker, r in self.retrievers.items()
        }
