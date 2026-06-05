"""
eval/run_all_evals.py
----------------------
Master eval runner: runs all three layers and produces the final results table.

Usage:
    python eval/run_all_evals.py
    python eval/run_all_evals.py --skip_ragas  (if RAG index not ready)
"""

from __future__ import annotations
import argparse
import asyncio
import json
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path("eval/results")


async def main(skip_ragas: bool = False, skip_backtest: bool = False):
    results = {}
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    print("\n" + "="*60)
    print("Portfolio Agent — Full Evaluation Harness")
    print(f"Timestamp: {timestamp}")
    print("="*60)

    # ── Layer 1: RAGAS ──────────────────────────────────────────────────────
    if not skip_ragas:
        print("\n[1/3] Running RAGAS evaluation (baseline vs improved)...")
        from eval.ragas_eval import run_experiment
        baseline = await run_experiment("baseline")
        improved = await run_experiment("improved")
        results["ragas"] = {
            "baseline": baseline["ragas_metrics"],
            "improved": improved["ragas_metrics"],
            "delta": {
                k: round(improved["ragas_metrics"][k] - baseline["ragas_metrics"][k], 3)
                for k in baseline["ragas_metrics"]
            }
        }
        print("\nRAGAS Delta (improved - baseline):")
        for k, v in results["ragas"]["delta"].items():
            arrow = "↑" if v > 0 else ("↓" if v < 0 else "→")
            print(f"  {k}: {arrow} {v:+.3f}")
    else:
        print("\n[1/3] Skipping RAGAS (--skip_ragas)")

    # ── Layer 2: Backtest ──────────────────────────────────────────────────
    if not skip_backtest:
        print("\n[2/3] Running backtest evaluation (A vs B vs C)...")
        from eval.backtest_eval import run_backtest_experiments
        bt_df = await run_backtest_experiments()
        summary = bt_df.groupby("experiment").agg({
            "portfolio_return_90d": "mean",
            "excess_vs_spy": "mean",
            "sharpe_ratio": "mean",
            "max_drawdown": "mean",
            "view_direction_accuracy": "mean",
        }).round(4)
        results["backtest"] = summary.to_dict()
    else:
        print("\n[2/3] Skipping backtest (--skip_backtest)")

    # ── Layer 3: LLM Judge ────────────────────────────────────────────────
    print("\n[3/3] Running LLM judge evaluation + Cohen's kappa...")
    from eval.judge_eval import run_judge_eval
    judge_results = await run_judge_eval(n_cases=30)
    results["judge"] = judge_results

    # ── Final Summary Table ────────────────────────────────────────────────
    print("\n" + "="*60)
    print("FINAL EVALUATION SUMMARY")
    print("="*60)

    if "ragas" in results:
        print("\n📊 RAG Quality (RAGAS):")
        print(f"  {'Metric':<25} {'Baseline':>10} {'Improved':>10} {'Δ':>8}")
        print(f"  {'-'*53}")
        for k in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            b = results["ragas"]["baseline"].get(k, 0)
            i = results["ragas"]["improved"].get(k, 0)
            d = results["ragas"]["delta"].get(k, 0)
            arrow = "↑" if d > 0 else ("↓" if d < 0 else "→")
            print(f"  {k:<25} {b:>10.3f} {i:>10.3f} {arrow}{d:>+7.3f}")

    if "backtest" in results:
        print("\n📈 Portfolio Performance (90-day backtest):")
        print(f"  {'Experiment':<20} {'Return':>8} {'vs SPY':>8} {'Sharpe':>8} {'MaxDD':>8}")
        print(f"  {'-'*52}")
        for exp in ["A_equal_weight", "B_price_only", "C_rag_agent"]:
            bt = results["backtest"]
            ret = bt.get("portfolio_return_90d", {}).get(exp, 0)
            exc = bt.get("excess_vs_spy", {}).get(exp, 0)
            shr = bt.get("sharpe_ratio", {}).get(exp, 0)
            mdd = bt.get("max_drawdown", {}).get(exp, 0)
            print(f"  {exp:<20} {ret:>8.2%} {exc:>+8.2%} {shr:>8.2f} {mdd:>8.2%}")

    print("\n🎯 LLM Judge Quality:")
    jm = results["judge"]["primary_mean_scores"]
    print(f"  Data grounding:    {jm.get('data_grounding', 0):.2f}/5")
    print(f"  Logic coherence:   {jm.get('logic_coherence', 0):.2f}/5")
    print(f"  Citation quality:  {jm.get('citation_quality', 0):.2f}/5")
    print(f"  Overall:           {jm.get('overall', 0):.2f}/5")
    print(f"\n  Cohen's kappa: {results['judge']['max_kappa']:.3f} "
          f"({'PASSED ✓' if results['judge']['gate_passed'] else 'FAILED ✗'})")

    # Save master results
    with open(RESULTS_DIR / "all_results.json", "w") as f:
        json.dump({"timestamp": timestamp, **results}, f, indent=2, default=str)
    print(f"\nAll results saved to eval/results/all_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_ragas", action="store_true")
    parser.add_argument("--skip_backtest", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args.skip_ragas, args.skip_backtest))
