"""
eval/generate_agent_outputs.py
-------------------------------
Bridge between the agent and the LLM-judge.

Problem this solves:
  - judge_eval.py reads `agent_outputs.csv` (real agent analyses WITH citations)
  - but nothing was producing that file — the judge was falling back to the
    RAGAS CSV, whose answers use a conservative "INSUFFICIENT_CONTEXT" prompt
    with no citations. That is why citation_quality κ collapsed to 0.

This script runs the agent's analyst logic over the golden set and writes
`eval/results/agent_outputs.csv` with the SAME forced-citation prompt the
portfolio_graph analyst now uses, so the judge evaluates the agent's true output.

Reuses:
  - golden set loading (from ragas_eval.py)
  - MultiTickerRetriever (from src.rag.retriever)
  - the page-aware context formatting + citation rules (from portfolio_graph analyst_node)

Run:
    python eval/generate_agent_outputs.py
    python eval/judge_eval.py        # now reads the file this produced
"""

from __future__ import annotations
import asyncio
import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from src.rag.retriever import MultiTickerRetriever

load_dotenv()

# Same model + settings as the agent (portfolio_graph.py)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

GOLDEN_PATH = Path("data/golden_set/golden.jsonl")
RESULTS_DIR = Path("eval/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Citation-forcing analyst prompt (mirrors portfolio_graph analyst_node) ─────
ANALYST_PROMPT = """You are a financial analyst. Answer the question using ONLY
the context passages provided below.

## Context Passages (each is tagged with its exact source page)
{context}

## Question
{question}

## Task
Write a concise, fact-grounded analysis answering the question.

CRITICAL CITATION RULES:
- The context blocks above are tagged with headers like "(source: 10-K p.45)".
- EVERY specific number MUST be immediately followed by its exact citation,
  copied verbatim from the tag of the passage it came from.
- Format MUST be: number + "(source: 10-K p.XX)"
  CORRECT:  "revenue was $60.9B (source: 10-K p.45)"
  WRONG:    "revenue was $60.9B per the filing"
  WRONG:    "revenue was $60.9B (10-K)"   <- missing page number
- If a number is not present in the context above, do NOT make it up.
- If the context does not contain the answer, say "INSUFFICIENT_CONTEXT".

Answer:"""


# ── Load golden set (same pattern as ragas_eval.build_ragas_dataset) ───────────
def load_golden_cases(max_cases: int = 25) -> list[dict]:
    with open(GOLDEN_PATH) as f:
        golden_cases = [json.loads(l) for l in f if l.strip()]
    # RAG eval cases only (exclude backtest), same filter as ragas_eval
    rag_cases = [c for c in golden_cases if c.get("category") != "backtest"]
    return rag_cases[:max_cases]


# ── Generate one agent analysis with citations ─────────────────────────────────
async def generate_one(
    case: dict,
    retriever: MultiTickerRetriever,
) -> dict:
    ticker = case["ticker"]
    question = case["question"]
    ground_truth = case.get("ground_truth", "")

    # 1. Retrieve context (same enhanced query style as ragas_eval)
    enhanced_query = f"{ticker} financial data: {question}"
    results = await retriever.aretrieve_for_ticker(ticker, enhanced_query, top_k=4)

    # 2. Build page-tagged context blocks (mirrors portfolio_graph analyst_node)
    context_blocks = []
    raw_contexts = []
    for r in results:
        cite_tag = f"(source: {r.doc_type} p.{r.page})"
        context_blocks.append(f"=== CITATION TAG TO USE: {cite_tag} ===\n{r.text}")
        raw_contexts.append(r.text)

    if not context_blocks:
        context_blocks = [f"No indexed documents for {ticker}"]
        raw_contexts = [f"No indexed documents for {ticker}"]

    context_str = "\n\n---\n\n".join(context_blocks)

    # 3. Generate citation-grounded analysis
    prompt = ANALYST_PROMPT.format(context=context_str, question=question)
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    answer = response.content.strip()

    print(f"  {case['id']} — {question[:55]}...")

    return {
        "case_id": case["id"],
        "question": question,
        "answer": answer,            # judge reads this
        "contexts": raw_contexts,    # judge reads this for context_summary
        "ground_truth": ground_truth,
        "category": case.get("category", "unknown"),
    }


# ── Main ───────────────────────────────────────────────────────────────────────
async def main(max_cases: int = 25):
    cases = load_golden_cases(max_cases)
    print(f"\nGenerating agent outputs (with citations) for {len(cases)} cases...")

    tickers = list(set(c["ticker"] for c in cases))
    retriever = MultiTickerRetriever(tickers)

    rows = []
    for case in cases:
        row = await generate_one(case, retriever)
        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "agent_outputs.csv"
    df.to_csv(out_path, index=False)

    # Quick sanity check: how many answers actually contain a page citation?
    has_page_cite = df["answer"].str.contains(r"p\.\d+", regex=True).sum()
    print(f"\nSaved {len(df)} agent outputs to {out_path}")
    print(f"Answers containing a page-level citation: {has_page_cite}/{len(df)}")
    print("\nExample answer (case 1):")
    print(df.iloc[0]["answer"][:400])


if __name__ == "__main__":
    asyncio.run(main())
