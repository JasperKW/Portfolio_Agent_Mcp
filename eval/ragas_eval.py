"""
eval/ragas_eval.py
-------------------
Layer 1: RAG Quality Evaluation using RAGAS.

Measures whether the agent's financial claims are grounded in retrieved docs.
Mirrors fin-rag-lab notebook 05 pattern.

Metrics:
  - faithfulness:        Are claims supported by context? (hallucination detector)
  - answer_relevancy:    Is the answer on-topic?
  - context_precision:   Did we retrieve the RIGHT chunks?
  - context_recall:      Did we retrieve ALL needed chunks?

Run:
    python eval/ragas_eval.py --experiment baseline
    python eval/ragas_eval.py --experiment improved  (parent-child enabled)
"""

from __future__ import annotations
import asyncio
import json
import os
from pathlib import Path

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from src.rag.retriever import MultiTickerRetriever
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

load_dotenv()

GOLDEN_PATH = Path("data/golden_set/golden.jsonl")
RESULTS_DIR = Path("eval/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Per-claim hallucination detector (fin-rag-lab nb05 pattern) ───────────────
CLAIM_CHECK_PROMPT = """You are a fact-checker for financial documents.

Given a CLAIM and a set of SOURCE PASSAGES, determine if the claim is supported.

CLAIM: {claim}

SOURCE PASSAGES:
{context}

Answer with ONLY a JSON object:
{{"supported": true/false, "reason": "one sentence", "source_quote": "exact quote from passages or null"}}
"""

async def check_claim(claim: str, context_chunks: list[str]) -> dict:
    """
    Per-claim hallucination check (fin-rag-lab §05 pattern).
    Each factual statement in the answer is checked against retrieved context.
    """
    from langchain_core.messages import HumanMessage
    context_str = "\n\n---\n\n".join(context_chunks[:3])
    prompt = CLAIM_CHECK_PROMPT.format(claim=claim, context=context_str)
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    
    try:
        raw = response.content.strip().strip("```json").strip("```").strip()
        return json.loads(raw)
    except Exception:
        return {"supported": False, "reason": "parse error", "source_quote": None}


async def extract_claims(answer: str) -> list[str]:
    """Split an answer into individual verifiable claims."""
    from langchain_core.messages import HumanMessage
    prompt = f"""Extract all factual claims from this financial analysis.
Each claim should be a single verifiable statement.

Text: {answer}

Respond ONLY with a JSON array of strings:
["claim 1", "claim 2", ...]
"""
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    try:
        raw = response.content.strip().strip("```json").strip("```").strip()
        return json.loads(raw)
    except Exception:
        return [answer]  # treat whole answer as one claim if extraction fails


# ── Build RAGAS dataset from golden set ───────────────────────────────────────
async def build_ragas_dataset(
    experiment_name: str = "baseline",
    max_cases: int = 25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run agent on golden set, collect (question, answer, contexts, ground_truth).
    Returns two DataFrames: ragas_dataset, per_claim_results.
    """
    with open(GOLDEN_PATH) as f:
        golden_cases = [json.loads(l) for l in f if l.strip()]

    # Filter to RAG eval cases only (not backtest)
    rag_cases = [c for c in golden_cases if c.get("category") != "backtest"][:max_cases]

    # Get all unique tickers
    tickers = list(set(c["ticker"] for c in rag_cases))
    retriever = MultiTickerRetriever(tickers)

    ragas_rows = []
    claim_rows = []

    for case in rag_cases:
        ticker = case["ticker"]
        question = case["question"]
        ground_truth = case["ground_truth"]

        print(f"  Evaluating: {case['id']} — {question[:60]}...")

        # 1. Retrieve context
        enhanced_query = f"{ticker} financial data: {question}"
        results = retriever.retrieve_for_ticker(ticker, enhanced_query, top_k=4)
        contexts = [r.text for r in results]

        if not contexts:
            contexts = [f"No indexed documents for {ticker}"]

        # 2. Generate answer using retrieved context
        context_str = "\n\n---\n\n".join(contexts)
        from langchain_core.messages import HumanMessage
        gen_prompt = f"""Answer the question using ONLY the provided context.
If the context does not contain the answer, say "INSUFFICIENT_CONTEXT".

Context:
{context_str}

Question: {question}

Answer:"""
        response = await llm.ainvoke([HumanMessage(content=gen_prompt)])
        answer = response.content.strip()

        ragas_rows.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth,
            "case_id": case["id"],
            "category": case.get("category", "unknown"),
        })

        # 3. Per-claim hallucination check
        if ground_truth != "INSUFFICIENT_CONTEXT" and len(answer) > 50:
            claims = await extract_claims(answer)
            for claim in claims[:4]:  # limit to 4 claims per question
                check = await check_claim(claim, contexts)
                claim_rows.append({
                    "case_id": case["id"],
                    "ticker": ticker,
                    "claim": claim,
                    "supported": check["supported"],
                    "reason": check["reason"],
                    "source_quote": check.get("source_quote"),
                })

    ragas_df = pd.DataFrame(ragas_rows)
    claim_df = pd.DataFrame(claim_rows)

    return ragas_df, claim_df


# ── Run RAGAS metrics ──────────────────────────────────────────────────────────
def run_ragas_metrics(ragas_df: pd.DataFrame) -> dict:
    """Run the four RAGAS metrics on the eval dataset."""
    dataset = Dataset.from_dict({
        "question": ragas_df["question"].tolist(),
        "answer": ragas_df["answer"].tolist(),
        "contexts": ragas_df["contexts"].tolist(),
        "ground_truth": ragas_df["ground_truth"].tolist(),
    })

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    return result


# ── Experiment runner ─────────────────────────────────────────────────────────
async def run_experiment(experiment_name: str = "baseline") -> None:
    """
    Experiment design (S18 §4 requirement):
    
    Control (A):   pure price signals → B-L → weights (no RAG)
    Experiment B:  + RAG fundamentals
    Experiment C:  + RAG + news sentiment
    
    This function runs the RAG eval component for a given experiment.
    """
    print(f"\n{'='*60}")
    print(f"Running RAGAS eval: {experiment_name}")
    print("="*60)

    ragas_df, claim_df = await build_ragas_dataset(experiment_name)

    # RAGAS metrics
    metrics = run_ragas_metrics(ragas_df)
    print(f"\nRAGAS Results ({experiment_name}):")
    print(f"  Faithfulness:      {metrics['faithfulness']:.3f}")
    print(f"  Answer Relevancy:  {metrics['answer_relevancy']:.3f}")
    print(f"  Context Precision: {metrics['context_precision']:.3f}")
    print(f"  Context Recall:    {metrics['context_recall']:.3f}")

    # Per-claim hallucination summary
    if not claim_df.empty:
        support_rate = claim_df["supported"].mean()
        print(f"\nPer-claim hallucination check:")
        print(f"  Claims checked:  {len(claim_df)}")
        print(f"  Support rate:    {support_rate:.1%}")
        print(f"  Hallucinated:    {(~claim_df['supported']).sum()} claims")

    # Save results
    out = {
        "experiment": experiment_name,
        "ragas_metrics": {
            "faithfulness": float(metrics["faithfulness"]),
            "answer_relevancy": float(metrics["answer_relevancy"]),
            "context_precision": float(metrics["context_precision"]),
            "context_recall": float(metrics["context_recall"]),
        },
        "per_claim_support_rate": float(claim_df["supported"].mean()) if not claim_df.empty else None,
        "n_cases": len(ragas_df),
        "n_claims_checked": len(claim_df),
    }

    with open(RESULTS_DIR / f"ragas_{experiment_name}.json", "w") as f:
        json.dump(out, f, indent=2)

    ragas_df.to_csv(RESULTS_DIR / f"ragas_{experiment_name}_rows.csv", index=False)
    if not claim_df.empty:
        claim_df.to_csv(RESULTS_DIR / f"claims_{experiment_name}.csv", index=False)

    print(f"\nResults saved to eval/results/ragas_{experiment_name}.json")
    return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="baseline",
                        choices=["baseline", "improved"])
    args = parser.parse_args()
    asyncio.run(run_experiment(args.experiment))
