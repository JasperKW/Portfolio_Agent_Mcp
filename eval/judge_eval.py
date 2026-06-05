"""
eval/judge_eval.py
-------------------
Layer 3: LLM-as-Judge for reasoning quality + Cohen's kappa calibration.

Pattern from S12:
  - Judge scores on 1-5 scale for: data_grounding, logic_coherence, citation_quality
  - Cross-family judge: GPT judges reasoning (avoids self-preference bias)
  - Cohen's kappa against second judge (or human proxy) — gate: κ ≥ 0.6

Run:
    python eval/judge_eval.py
"""

from __future__ import annotations
import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from sklearn.metrics import cohen_kappa_score

load_dotenv()

RESULTS_DIR = Path("eval/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Cross-family judges (S12: GPT judges Claude output to avoid self-preference)
judge_primary   = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
judge_secondary = ChatGroq(model="qwen/qwen3-32b", temperature=0)


# ── Judge rubric ──────────────────────────────────────────────────────────────
JUDGE_PROMPT = """You are an expert financial analyst evaluating an AI-generated investment analysis.

## Investment Analysis to Evaluate
{analysis}

## Source Documents Available to the Agent
{context_summary}

Score the analysis on THREE dimensions (1-5 scale each):

1. **data_grounding** (1-5): Are claims backed by specific numbers?
   1 = no numbers mentioned at all
   2 = vague references ("revenue grew")
   3 = approximate numbers ("revenue was about $200B")
   4 = specific numbers ("revenue was $215.9B")
   5 = specific numbers with source ("revenue was $215.9B per 10-K p.66")

2. **logic_coherence** (1-5): Is the reasoning from data to conclusion consistent?
   1 = conclusion contradicts the data
   2 = conclusion loosely related to data
   3 = logical but missing steps
   4 = clear cause-effect chain
   5 = tight causal chain with explicit reasoning at every step

3. **citation_quality** (1-5): Are source references specific and verifiable?
   1 = no citations at all
   2 = mentions "the filing" without specifics
   3 = mentions document type ("10-K" or "earnings")
   4 = mentions document + section ("10-K Risk Factors")
   5 = mentions document + page number ("10-K p.34")

Respond ONLY with a JSON object:
{{
  "data_grounding": <1-5>,
  "logic_coherence": <1-5>,
  "citation_quality": <1-5>,
  "overall": <1-5>,
  "key_issue": "one sentence on the biggest weakness"
}}
"""


@dataclass
class JudgeScore:
    case_id: str
    judge: str
    data_grounding: int
    logic_coherence: int
    citation_quality: int
    overall: int
    key_issue: str


async def score_case(
    case_id: str,
    analysis: str,
    context_summary: str,
    judge,
    judge_name: str,
) -> JudgeScore:
    """Run judge on a single case."""
    prompt = JUDGE_PROMPT.format(
        analysis=analysis[:2000],  # truncate to avoid token overflow
        context_summary=context_summary[:500],
    )
    response = await judge.ainvoke([HumanMessage(content=prompt)])

    try:
        raw = response.content.strip()
    
    # 去掉Qwen3/DeepSeek reasoning model的<think>...</think>部分
        if "<think>" in raw:
            raw = raw.split("</think>")[-1].strip()
    
    # 去掉markdown代码块
        raw = raw.strip("```json").strip("```").strip()
    
        parsed = json.loads(raw)
        return JudgeScore(
            case_id=case_id,
            judge=judge_name,
            data_grounding=int(parsed.get("data_grounding", 3)),
            logic_coherence=int(parsed.get("logic_coherence", 3)),
            citation_quality=int(parsed.get("citation_quality", 3)),
            overall=int(parsed.get("overall", 3)),
            key_issue=parsed.get("key_issue", ""),
        )
    except Exception as e:
        print(f"  Judge parse error for {case_id}: {e}")
        return JudgeScore(case_id, judge_name, 3, 3, 3, 3, "parse error")


# ── Load agent outputs from previous eval run ─────────────────────────────────
def load_agent_outputs(results_dir: Path = RESULTS_DIR) -> list[dict]:
    """
    Load agent-generated analyses.
    Priority: agent_outputs.csv (real agent memos with citations)
              → RAGAS CSV (fallback)
              → synthetic (last resort)
    """
    agent_csv = results_dir / "agent_outputs.csv"
    if agent_csv.exists():
        df = pd.read_csv(agent_csv)
        return df.to_dict("records")

    ragas_csv = results_dir / "ragas_improved_rows.csv"
    if not ragas_csv.exists():
        ragas_csv = results_dir / "ragas_baseline_rows.csv"

    if not ragas_csv.exists():
        print("Warning: No RAGAS results found. Using synthetic test cases.")
        return _synthetic_test_cases()

    df = pd.read_csv(ragas_csv)
    return df.to_dict("records")


def _synthetic_test_cases() -> list[dict]:
    """Synthetic cases for testing the judge pipeline."""
    return [
        {
            "case_id": "syn_001",
            "question": "What drove NVDA revenue growth?",
            "answer": "NVDA data center revenue grew 142% per the FY2025 10-K (p.34), driven by H100 GPU demand from cloud hyperscalers. Gross margin expanded to 73.5% due to favorable product mix per earnings transcript.",
            "contexts": ["NVDA 10-K data center revenue grew 142% year-over-year..."],
            "ground_truth": "data center growth drove revenue",
        },
        {
            "case_id": "syn_002",
            "question": "What is MSFT's Azure growth rate?",
            "answer": "Azure grew approximately 31% year-over-year according to Microsoft's latest earnings. AI services contributed about 7 percentage points to that growth.",
            "contexts": ["Microsoft Azure and other cloud services revenue grew 31%..."],
            "ground_truth": "Azure grew ~31%",
        },
        {
            "case_id": "syn_003",
            "question": "What are AAPL's key growth drivers?",
            "answer": "Apple's growth is driven by services revenue which reached $100 billion annually, iPhone upgrades, and emerging markets expansion. The company's installed base of 2 billion devices supports recurring revenue.",
            "contexts": ["Apple services revenue..."],
            "ground_truth": "services and iPhone",
        },
    ] * 10  # repeat to get 30 cases


# ── Compute Cohen's kappa ─────────────────────────────────────────────────────
def compute_kappa(
    scores_primary: list[int],
    scores_secondary: list[int],
    metric_name: str,
) -> float:
    """
    Cohen's kappa between primary and secondary judge.
    Gate: κ ≥ 0.6 (S12 §5 requirement).
    
    Quadratic weights: partial credit for near-misses (1 vs 2 is less bad than 1 vs 5)
    """
    kappa = cohen_kappa_score(scores_primary, scores_secondary, weights="quadratic")
    print(f"  κ ({metric_name}): {kappa:.3f} {'✓' if kappa >= 0.6 else '✗ BELOW GATE'}")
    return kappa


# ── Main judge evaluation ─────────────────────────────────────────────────────
async def run_judge_eval(n_cases: int = 30) -> dict:
    """
    Score n_cases with both judges, compute kappa.
    
    S12 requirement: κ ≥ 0.6 on at least one dimension.
    """
    cases = load_agent_outputs()[:n_cases]
    print(f"\nRunning judge eval on {len(cases)} cases...")

    # Score with both judges (concurrently for speed)
    primary_tasks = [
        score_case(
            case.get("case_id", f"case_{i}"),
            case.get("answer", ""),
            str(case.get("contexts", [])[:1]),
            judge_primary,
            "llama-3.3-70b",
        )
        for i, case in enumerate(cases)
    ]
    secondary_tasks = [
        score_case(
            case.get("case_id", f"case_{i}"),
            case.get("answer", ""),
            str(case.get("contexts", [])[:1]),
            judge_secondary,
            "qwen3-32b",
        )
        for i, case in enumerate(cases)
    ]

    print("  Scoring with primary judge (llama-3.3-70b via Groq)...")
    primary_scores = []
    for i, task in enumerate(primary_tasks):
        score = await task
        primary_scores.append(score)
        print(f"    [{i+1}/{len(primary_tasks)}] done")
        await asyncio.sleep(4)   

    print("  Scoring with secondary judge (qwen3-32b via Groq)...")
    secondary_scores = []
    for i, task in enumerate(secondary_tasks):
        score = await task
        secondary_scores.append(score)
        print(f"    [{i+1}/{len(secondary_tasks)}] done")
        await asyncio.sleep(4)

    # Build DataFrames
    primary_df = pd.DataFrame([vars(s) for s in primary_scores])
    secondary_df = pd.DataFrame([vars(s) for s in secondary_scores])

    # Cohen's kappa per dimension
    print("\nCohen's kappa (primary vs secondary judge):")
    kappas = {}
    for dim in ["data_grounding", "logic_coherence", "citation_quality", "overall"]:
        k = compute_kappa(
            primary_df[dim].tolist(),
            secondary_df[dim].tolist(),
            dim,
        )
        kappas[dim] = round(k, 3)

    # Summary stats
    print(f"\nPrimary Judge Mean Scores:")
    for dim in ["data_grounding", "logic_coherence", "citation_quality", "overall"]:
        print(f"  {dim}: {primary_df[dim].mean():.2f}/5")

    # Gate check
    max_kappa = max(kappas.values())
    gate_passed = max_kappa >= 0.6
    print(f"\nMax kappa: {max_kappa:.3f} — Gate (κ≥0.6): {'PASSED ✓' if gate_passed else 'FAILED ✗'}")

    # Save
    results = {
        "n_cases": len(cases),
        "kappas": kappas,
        "gate_passed": gate_passed,
        "max_kappa": max_kappa,
        "primary_mean_scores": {
            dim: round(float(primary_df[dim].mean()), 3)
            for dim in ["data_grounding", "logic_coherence", "citation_quality", "overall"]
        },
    }

    with open(RESULTS_DIR / "judge_results.json", "w") as f:
        json.dump(results, f, indent=2)

    primary_df.to_csv(RESULTS_DIR / "judge_primary.csv", index=False)
    secondary_df.to_csv(RESULTS_DIR / "judge_secondary.csv", index=False)

    print(f"\nResults saved to eval/results/judge_results.json")
    return results


if __name__ == "__main__":
    asyncio.run(run_judge_eval(n_cases=30))
