"""
src/rag/reranker.py
--------------------
LLM-as-reranker (S5 §5.3 pattern).

Adds a precision-focused reranking step after RRF recall:
  FAISS + BM25 → RRF (recall) → LLM rerank (precision) → top_k

Uses gpt-4o-mini with JSON mode for deterministic, structured scoring.
Cap at 10 candidates per call to control latency and cost.
"""

from __future__ import annotations
import json
from openai import AsyncOpenAI

client = AsyncOpenAI()

RERANK_PROMPT = """\
You are a financial document relevance scorer.

Query: "{query}"

Score each document for relevance to the query (0-10):
- 9-10: Directly answers the query with specific numbers or facts
- 6-8:  Highly relevant, partially answers
- 3-5:  Tangentially related (same company but wrong topic)
- 0-2:  Not relevant at all

Documents:
{documents}

Return ONLY JSON: {{"scores": [{{"doc_id": "...", "score": N, "reason": "..."}}]}}
"""


async def llm_rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    model: str = "gpt-4o-mini",
) -> list[dict]:
    """
    Rerank candidates using LLM scoring (S5 §5.3).

    Args:
        query: the user's retrieval query
        candidates: list of dicts with at least 'chunk_id' and 'text'
        top_k: how many to return after reranking
        model: LLM to use for scoring

    Returns:
        top_k candidates sorted by LLM relevance score (descending)
    """
    if not candidates:
        return []

    # Cap at 10 candidates to control latency (S5: "cap at 20; LLM context isn't free")
    # We use 10 since financial docs are longer
    capped = candidates[:10]

    # Build document string for the prompt
    docs_str = "\n\n".join(
        f"[doc_id={c['chunk_id']}]\n{c['text'][:400]}"
        for c in capped
    )

    resp = await client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[{
            "role": "user",
            "content": RERANK_PROMPT.format(query=query, documents=docs_str),
        }],
        temperature=0,
    )

    try:
        parsed = json.loads(resp.choices[0].message.content)
        scores = {s["doc_id"]: s["score"] for s in parsed["scores"]}
    except (json.JSONDecodeError, KeyError):
        # If parsing fails, return candidates in original RRF order
        return capped[:top_k]

    # Sort by LLM score descending, fall back to 0 for unscored docs
    ranked = sorted(capped, key=lambda c: scores.get(c["chunk_id"], 0), reverse=True)
    return ranked[:top_k]
