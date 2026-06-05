"""
src/rag/ingest.py
-----------------
RAG ingestion pipeline for financial PDFs (10-K, earnings transcripts).
Pattern: fin-rag-lab notebooks 01-03 (PyMuPDF parse → parent-child chunk → hybrid index)

Usage:
    python -m src.rag.ingest --tickers NVDA MSFT AAPL
    python -m src.rag.ingest --pdf_dir data/filings/
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import numpy as np
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from rank_bm25 import BM25Okapi
from tqdm import tqdm

load_dotenv()

# ── Constants (mirroring fin-rag-lab parent-child defaults) ──────────────────
PARENT_CHUNK_SIZE = 800
PARENT_CHUNK_OVERLAP = 150
CHILD_CHUNK_SIZE = 200
CHILD_CHUNK_OVERLAP = 30
INDEX_DIR = Path("data/index")
FILINGS_DIR = Path("data/filings")


# ── Data structures ──────────────────────────────────────────────────────────
@dataclass
class DocumentChunk:
    """One chunk with full lineage for citation."""
    chunk_id: str
    parent_id: str
    text: str
    metadata: dict = field(default_factory=dict)
    # metadata keys: ticker, doc_type (10K/earnings), fiscal_year, page, source_pdf


@dataclass
class ParentChildIndex:
    """
    Parent-child index (fin-rag-lab §4.5 pattern).
    
    Insight: small child chunks embed cleanly for retrieval,
    large parent chunks give the LLM enough context to answer.
    """
    child_chunks: list[DocumentChunk]       # indexed in FAISS + BM25
    parent_map: dict[str, DocumentChunk]    # parent_id → parent chunk
    faiss_store: Optional[object] = None
    bm25_index: Optional[BM25Okapi] = None
    bm25_corpus: list[list[str]] = field(default_factory=list)


# ── Step 1: PDF Parsing (PyMuPDF — fin-rag-lab nb01 pattern) ─────────────────
def parse_pdf(pdf_path: Path, ticker: str, doc_type: str = "10K") -> list[dict]:
    """
    Extract structured text blocks from a financial PDF.
    Returns list of {page, text, bbox} dicts.
    
    Financial PDFs have heavy table structure — we extract text
    page-by-page and preserve page numbers for citation.
    """
    doc = fitz.open(str(pdf_path))
    pages = []
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        # Drop boilerplate: headers/footers (short lines at top/bottom)
        lines = [l.strip() for l in text.split("\n") if len(l.strip()) > 20]
        clean_text = "\n".join(lines)
        if len(clean_text) > 100:  # skip near-empty pages
            pages.append({
                "page": page_num + 1,
                "text": clean_text,
                "ticker": ticker,
                "doc_type": doc_type,
                "source_pdf": pdf_path.name,
            })
    doc.close()
    print(f"  Parsed {len(pages)} pages from {pdf_path.name}")
    return pages

def parse_html(html_path: Path, ticker: str, doc_type: str = "10K") -> list[dict]:
    """
    Extract text from SEC HTML filings.
    Strips all HTML tags, preserves text content.
    """
    from bs4 import BeautifulSoup
    
    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    
    # Remove script/style tags
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    
    text = soup.get_text(separator="\n")
    # Clean up excessive whitespace
    lines = [l.strip() for l in text.split("\n") if len(l.strip()) > 30]
    clean_text = "\n".join(lines)
    
    # Split into page-sized chunks (HTML has no pages, simulate ~3000 chars per page)
    page_size = 3000
    pages = []
    for i in range(0, len(clean_text), page_size):
        chunk = clean_text[i:i+page_size]
        pages.append({
            "page": i // page_size + 1,
            "text": chunk,
            "ticker": ticker,
            "doc_type": doc_type,
            "source_pdf": html_path.name,
        })
    
    print(f"  Parsed {len(pages)} pages from {html_path.name}")
    return pages


# ── Step 2: Parent-Child Chunking ─────────────────────────────────────────────
def build_parent_child_chunks(
    pages: list[dict],
    parent_size: int = PARENT_CHUNK_SIZE,
    child_size: int = CHILD_CHUNK_SIZE,
) -> tuple[list[DocumentChunk], dict[str, DocumentChunk]]:
    """
    Two-pass chunking (fin-rag-lab §4.5):
      Pass 1: split into parent chunks (800 tokens) → good generation context
      Pass 2: split each parent into child chunks (200 tokens) → good retrieval
    
    Returns (child_chunks, parent_map).
    """
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_size,
        chunk_overlap=PARENT_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_size,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )

    parent_map: dict[str, DocumentChunk] = {}
    child_chunks: list[DocumentChunk] = []

    for page in pages:
        full_text = page["text"]
        parent_texts = parent_splitter.split_text(full_text)

        for p_idx, p_text in enumerate(parent_texts):
            # Stable content-addressed ID (fin-rag-lab pattern)
            p_id = hashlib.md5(p_text.encode()).hexdigest()[:12]
            parent = DocumentChunk(
                chunk_id=p_id,
                parent_id=p_id,
                text=p_text,
                metadata={
                    **{k: page[k] for k in ["ticker", "doc_type", "source_pdf"]},
                    "page": page["page"],
                    "chunk_type": "parent",
                },
            )
            parent_map[p_id] = parent

            # Children of this parent
            child_texts = child_splitter.split_text(p_text)
            for c_idx, c_text in enumerate(child_texts):
                c_id = f"{p_id}_c{c_idx}"
                child = DocumentChunk(
                    chunk_id=c_id,
                    parent_id=p_id,
                    text=c_text,
                    metadata={**parent.metadata, "chunk_type": "child"},
                )
                child_chunks.append(child)

    print(f"  Built {len(parent_map)} parents, {len(child_chunks)} children")
    return child_chunks, parent_map


# ── Step 3: Hybrid Index (FAISS + BM25 — fin-rag-lab nb03 pattern) ────────────
def build_hybrid_index(child_chunks: list[DocumentChunk]) -> tuple:
    """
    Index child chunks in both vector and keyword stores.
    At query time, results from both are merged with RRF.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Vector index
    texts = [c.text for c in child_chunks]
    metadatas = [{"chunk_id": c.chunk_id, "parent_id": c.parent_id, **c.metadata}
                 for c in child_chunks]

    print("  Building FAISS index...")
    faiss_store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)

    # BM25 index (tokenise by whitespace — simple but effective for financial text)
    print("  Building BM25 index...")
    tokenised = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenised)

    return faiss_store, bm25, tokenised


# ── Reciprocal Rank Fusion (fin-rag-lab src/retrievers/rrf.py pattern) ────────
def rrf_merge(
    vector_results: list,    # (doc, score) from FAISS
    bm25_results: list,      # (doc, score) from BM25
    k: int = 60,             # RRF constant — 60 is the canonical default
    top_n: int = 8,
) -> list:
    """
    Reciprocal Rank Fusion: merge two ranked lists without score normalisation.
    
    Score = Σ 1/(k + rank_i)  for each result list i
    
    Why RRF over weighted-score combination:
    - No need to normalise incompatible score scales (cosine vs BM25)
    - More robust to outlier scores
    - Canonical default for production (fin-rag-lab, Cohere rerank docs)
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, object] = {}

    for rank, (doc, _) in enumerate(vector_results):
        cid = doc.metadata.get("chunk_id", str(rank))
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
        doc_map[cid] = doc

    for rank, (doc, _) in enumerate(bm25_results):
        cid = doc.metadata.get("chunk_id", str(rank))
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
        doc_map[cid] = doc

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [(doc_map[cid], scores[cid]) for cid in sorted_ids[:top_n] if cid in doc_map]


# ── Main Ingestion Entrypoint ─────────────────────────────────────────────────
def ingest_ticker(ticker: str, pdf_dir: Path = FILINGS_DIR) -> ParentChildIndex:
    """Full pipeline: PDF → parse → chunk → index."""
    pdfs = list(pdf_dir.glob(f"{ticker}*.html")) + list(pdf_dir.glob(f"{ticker}*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found for {ticker} in {pdf_dir}")

    all_children: list[DocumentChunk] = []
    all_parents: dict[str, DocumentChunk] = {}

    for pdf_path in tqdm(pdfs, desc=f"Parsing {ticker}"):
        doc_type = "earnings" if "earnings" in pdf_path.name.lower() else "10K"
        if pdf_path.suffix == ".html":
            pages = parse_html(pdf_path, ticker, doc_type)
        else:
            pages = parse_pdf(pdf_path, ticker, doc_type)
        children, parents = build_parent_child_chunks(pages)
        all_children.extend(children)
        all_parents.update(parents)

    faiss_store, bm25, corpus = build_hybrid_index(all_children)

    idx = ParentChildIndex(
        child_chunks=all_children,
        parent_map=all_parents,
        faiss_store=faiss_store,
        bm25_index=bm25,
        bm25_corpus=corpus,
    )

    # Persist
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss_store.save_local(str(INDEX_DIR / ticker / "faiss"))
    import json
    metadata = {
        "parent_map": {
            k: {
                "chunk_id": v.chunk_id,
                "parent_id": v.parent_id,
                "text": v.text,
                "metadata": v.metadata,
            } for k, v in all_parents.items()
        },
        "child_chunks": [
            {
                "chunk_id": c.chunk_id,
                "parent_id": c.parent_id,
                "text": c.text,
                "metadata": c.metadata,
            } for c in all_children
        ],
        "bm25_corpus": corpus,
    }
    with open(INDEX_DIR / ticker / "metadata.json", "w") as f:
        json.dump(metadata, f)

    print(f"✓ Indexed {ticker}: {len(all_parents)} parents, {len(all_children)} children")
    return idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=["NVDA", "MSFT", "AAPL"])
    parser.add_argument("--pdf_dir", default="data/filings")
    args = parser.parse_args()
    for ticker in args.tickers:
        ingest_ticker(ticker, Path(args.pdf_dir))
