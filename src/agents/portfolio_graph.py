"""
src/agents/portfolio_graph.py
------------------------------
LangGraph multi-agent portfolio research system.

Architecture (S9 Pattern B — Supervisor → Specialists + parallel fan-out):

  Planner  →  [Analyst_NVDA, Analyst_MSFT, ...]  →  Synthesizer
               (parallel via Send API)

State machine:
  START → planner → [analyst × N via Send] → synthesizer → END

Why LangGraph (not plain asyncio):
  - State is shared via TypedDict + reducers (no data loss on fan-out)
  - Send API handles dynamic N without compile-time edge count
  - Checkpointer gives free resume-on-crash
  - get_state_history() for audit / time-travel debugging
"""

from __future__ import annotations
import asyncio
import json
import os
from typing import Annotated, Any, Optional, Sequence, TypedDict
import operator

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langgraph.checkpoint.memory import MemorySaver

from src.rag.retriever import MultiTickerRetriever
from src.tools.financial_tools import ToolRegistry, ToolResult
from src.mcp.client import discover_mcp_tools

# MCP server URL — reads from env so it's easy to swap in prod
MCP_SERVER_URL = os.getenv("FINANCIAL_MCP_URL", "http://localhost:8001/mcp")

# Module-level registry, populated at startup via _get_registry()
_registry: ToolRegistry | None = None


async def _get_registry() -> ToolRegistry:
    """
    Build tool registry on first call.

    Strategy: try MCP server first (preferred — dynamic discovery, decoupled).
    Fall back to local Python tools if server is unreachable.
    This lets the agent work in both dev (no server running) and prod.
    """
    global _registry
    if _registry is not None:
        return _registry

    _registry = ToolRegistry()

    try:
        # Dynamic MCP discovery: connect → tools/list → register adapters
        # The agent doesn't hardcode tool names — it discovers them at runtime
        adapters = await discover_mcp_tools(MCP_SERVER_URL, "financial")
        for adapter in adapters:
            _registry.register(adapter)
        print(f"[Registry] Using MCP tools from {MCP_SERVER_URL}")

    except Exception as e:
        # Fallback: register local Python tools directly (dev mode)
        print(f"[Registry] MCP server unreachable ({e}), falling back to local tools")
        from src.tools.financial_tools import PriceTool, TechnicalTool, BlackLittermanTool
        _registry.register(PriceTool())
        _registry.register(TechnicalTool())
        _registry.register(BlackLittermanTool())

    return _registry

load_dotenv()

MODEL = "gpt-4o-mini"
llm = ChatOpenAI(model=MODEL, temperature=0)


# ── State Schema (S9 §2.2) ────────────────────────────────────────────────────
class AnalystOutput(TypedDict):
    ticker: str
    fundamental_summary: str      # RAG-derived
    technical_signal: str         # tool-derived
    view: dict                    # {"ticker", "expected_alpha", "confidence"}
    citations: list[str]          # source PDFs + pages


class PortfolioState(TypedDict):
    # Input
    user_query: str
    tickers: list[str]            # overwrite — set by planner

    # Accumulator fields (operator.add reducer = list-extend, safe for fan-out)
    analyst_outputs: Annotated[list[AnalystOutput], operator.add]
    cost_usd: Annotated[float, operator.add]

    # Single-writer fields (overwrite — only synthesizer writes)
    portfolio_weights: Optional[dict[str, float]]
    final_report: Optional[str]
    optimization_stats: Optional[dict]


# ── Node 1: Planner ───────────────────────────────────────────────────────────
async def planner_node(state: PortfolioState) -> dict:
    """
    Decompose user query into a list of tickers to analyse.
    Bounded autonomy pattern: ONE LLM call with structured output.
    """
    prompt = f"""You are a portfolio research planner.
    
User request: {state["user_query"]}

Select 3-6 US large-cap stocks most relevant to this request.
Respond ONLY with a JSON object:
{{"tickers": ["NVDA", "MSFT", "AAPL"], "rationale": "one sentence why"}}

Rules:
- Only include stocks with liquid options (S&P 500 constituents preferred)
- If the user specifies tickers, use them directly
- For "tech defensive", prefer software/cloud over semis
"""
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    
    try:
        raw = response.content.strip()
        # Strip markdown fences if present
        if "```" in raw:
            raw = raw.split("```")[1].strip("json").strip()
        parsed = json.loads(raw)
        tickers = parsed.get("tickers", ["NVDA", "MSFT", "AAPL", "GOOGL", "AMZN"])
    except Exception:
        tickers = ["NVDA", "MSFT", "AAPL", "GOOGL", "AMZN"]

    cost = _estimate_cost(response)
    print(f"[Planner] Selected tickers: {tickers}")
    return {"tickers": tickers, "cost_usd": cost}


# ── Fan-out Router ────────────────────────────────────────────────────────────
def route_to_analysts(state: PortfolioState) -> list[Send]:
    """
    LangGraph Send API: dynamically spawn one Analyst node per ticker.
    The graph doesn't know N at compile time — Send handles this at runtime.
    
    Each Send passes a sub-state to the 'analyst' node.
    Results accumulate in analyst_outputs via the operator.add reducer.
    """
    return [
        Send("analyst", {
            "ticker": ticker,
            "user_query": state["user_query"],
        })
        for ticker in state["tickers"]
    ]


# ── Node 2: Analyst (runs N times in parallel) ────────────────────────────────
class AnalystSubState(TypedDict):
    """Minimal sub-state passed to each Analyst via Send."""
    ticker: str
    user_query: str


async def analyst_node(sub_state: AnalystSubState) -> dict:
    """
    Per-ticker research agent.
    
    Steps:
    1. RAG: retrieve financial doc context for this ticker
    2. Tool: get price data + technical signals
    3. LLM: synthesise a structured View
    
    Returns partial PortfolioState update (analyst_outputs list).
    """
    ticker = sub_state["ticker"]
    query = sub_state["user_query"]
    print(f"  [Analyst:{ticker}] Starting research...")

    # ── 1. RAG retrieval ──────────────────────────────────────────────────────
    retriever = MultiTickerRetriever([ticker])
    rag_results = retriever.retrieve_for_ticker(
        ticker,
        query=f"{ticker} revenue growth margins outlook risk factors",
        top_k=4,
    )

    # Format context + citations
    context_blocks = []
    citations = []
    for r in rag_results:
        context_blocks.append(f"[{r.doc_type}, p.{r.page}]\n{r.text}")
        citations.append(f"{r.source_pdf} p.{r.page}")
    
    rag_context = "\n\n---\n\n".join(context_blocks) if context_blocks else \
        f"No indexed documents found for {ticker}. Using market data only."

    # ── 2. Tool calls (price + technicals) ───────────────────────────────────
    # MCP-aware registry: tries namespaced MCP tools first, falls back to local
    registry = await _get_registry()
    all_tool_names = [t["function"]["name"] for t in registry.get_openai_tools()]

    # Namespaced MCP names (financial__*) vs local fallback names
    price_tool = "financial__get_price_history" if "financial__get_price_history" in all_tool_names else "get_price_data"
    tech_tool = "financial__get_technical_signals" if "financial__get_technical_signals" in all_tool_names else "get_technical_signals"

    price_result: ToolResult = await registry.execute(price_tool, {"ticker": ticker})
    tech_result: ToolResult = await registry.execute(tech_tool, {"ticker": ticker})

    price_str = price_result.to_llm_str()
    tech_str = tech_result.to_llm_str()
    print(f"  [Analyst:{ticker}] Price: {price_str[:300]}")
    print(f"  [Analyst:{ticker}] Tech: {tech_str[:300]}")

    # ── 3. LLM view generation ────────────────────────────────────────────────
    # Instruction: cite docs, don't fabricate numbers (refusal-on-empty from fin-rag-lab)
    analyst_prompt = f"""You are a financial analyst. Analyse {ticker} and generate a structured investment view.

## Financial Document Context (from 10-K / earnings filings)
{rag_context}

## Price Data
{price_str}

## Technical Signals  
{tech_str}

## Task
Based ONLY on the above information:
1. Write a 3-sentence fundamental_summary (cite specific numbers from the documents)
2. Summarise the technical_signal in 1 sentence
3. Generate a structured view

CRITICAL RULES:
- If a number is not in the documents or tools above, do NOT make it up
- If context is insufficient, say "insufficient data" for that field
- Cite your sources: "per 10-K p.X" or "per price data"

Respond ONLY with JSON:
{{
  "fundamental_summary": "...",
  "technical_signal": "...",
  "view": {{
    "ticker": "{ticker}",
    "expected_alpha": <float, e.g. 0.04 for +4% expected excess return>,
    "confidence": <float 0.0-1.0>
  }},
  "reasoning": "2-3 sentences explaining the view"
}}
"""

    response = await llm.ainvoke([HumanMessage(content=analyst_prompt)])

    try:
        raw = response.content.strip().strip("```json").strip("```").strip()
        parsed = json.loads(raw)
    except Exception:
        # Graceful degradation — neutral view if parsing fails
        parsed = {
            "fundamental_summary": f"Parse error for {ticker}",
            "technical_signal": tech_str[:200],
            "view": {"ticker": ticker, "expected_alpha": 0.0, "confidence": 0.3},
            "reasoning": "Could not parse analyst output",
        }

    cost = _estimate_cost(response)
    analyst_out: AnalystOutput = {
        "ticker": ticker,
        "fundamental_summary": parsed.get("fundamental_summary", ""),
        "technical_signal": parsed.get("technical_signal", ""),
        "view": parsed.get("view", {"ticker": ticker, "expected_alpha": 0.0, "confidence": 0.3}),
        "citations": citations,
    }

    print(f"  [Analyst:{ticker}] Done. View: alpha={analyst_out['view'].get('expected_alpha', 0):.2%}, "
          f"conf={analyst_out['view'].get('confidence', 0):.0%}")

    # operator.add reducer extends the list safely across parallel branches
    return {"analyst_outputs": [analyst_out], "cost_usd": cost}


# ── Node 3: Synthesizer ───────────────────────────────────────────────────────
async def synthesizer_node(state: PortfolioState) -> dict:
    """
    Collect all analyst views, run Black-Litterman, produce final report.
    
    Two steps:
    1. Tool call: BlackLittermanTool with all views → optimal weights
    2. LLM: generate human-readable portfolio report with citations
    """
    tickers = state["tickers"]
    outputs = state["analyst_outputs"]

    # ── 1. Black-Litterman optimization ──────────────────────────────────────
    views = [o["view"] for o in outputs if o["view"].get("expected_alpha") is not None]
    
    registry = await _get_registry()
    all_tool_names = [t["function"]["name"] for t in registry.get_openai_tools()]
    bl_tool = "financial__optimize_portfolio" if "financial__optimize_portfolio" in all_tool_names else "optimize_portfolio"

    bl_result: ToolResult = await registry.execute(
        bl_tool, {"tickers": tickers, "views": views}
    )

    if not bl_result.success:
        # Fallback to equal weight
        weights = {t: round(1 / len(tickers), 4) for t in tickers}
        opt_stats = {"error": bl_result.error, "fallback": "equal_weight"}
        print(f"[Synthesizer] B-L failed, using equal weight: {bl_result.error}")
    else:
        weights = bl_result.output["weights"]
        opt_stats = {
            "expected_return": bl_result.output["expected_annual_return"],
            "volatility": bl_result.output["annual_volatility"],
            "sharpe_ratio": bl_result.output["sharpe_ratio"],
            "method": bl_result.output["optimization"],
        }

    # ── 2. Generate portfolio report ──────────────────────────────────────────
    analyst_summaries = "\n\n".join([
        f"**{o['ticker']}** (weight: {weights.get(o['ticker'], 0):.1%})\n"
        f"Fundamental: {o['fundamental_summary']}\n"
        f"Technical: {o['technical_signal']}\n"
        f"View: alpha={o['view'].get('expected_alpha', 0):.2%}, "
        f"confidence={o['view'].get('confidence', 0):.0%}\n"
        f"Source docs: {'; '.join(o['citations']) or 'market data only'}"
        for o in outputs
    ])

    synthesis_prompt = f"""You are a portfolio manager writing a final investment memo.

## User Request
{state["user_query"]}

## Analyst Research (with sources)
{analyst_summaries}

## Optimized Portfolio (Black-Litterman)
Weights: {json.dumps(weights, indent=2)}
Stats: {json.dumps(opt_stats, indent=2)}

Write a concise investment memo (400-500 words) covering:
1. Portfolio rationale (why these weights)
2. Key risks
3. What to monitor (1-2 catalysts per position)

## CITATION RULES — MANDATORY
- Every specific number MUST be followed by its source in parentheses.
- Format: "Revenue was $60.9B (10-K p.45)" or "RSI at 68 (price data)"
- If the source doc name is available, use it. If only market data, write "(market data)".
- If you cannot find a source for a number, write "insufficient data" instead of guessing.
- Do NOT write any sentence with a specific number that lacks a citation.

Example of correct citation style:
"NVDA reported data center revenue of $47.5B (10-K FY2025), up 142% YoY (10-K FY2025 p.32).
 The RSI stands at 71.2 (price data), suggesting near-term overbought conditions."
"""

    response = await llm.ainvoke([HumanMessage(content=synthesis_prompt)])
    cost = _estimate_cost(response)

    print(f"[Synthesizer] Done. Weights: {weights}")

    return {
        "portfolio_weights": weights,
        "final_report": response.content,
        "optimization_stats": opt_stats,
        "cost_usd": cost,
    }


# ── Graph Assembly ─────────────────────────────────────────────────────────────
def build_portfolio_graph(use_checkpointer: bool = True):
    """
    Compile the LangGraph state machine.
    
    Graph topology:
      START → planner → <Send fan-out> → analyst (×N) → synthesizer → END
    
    The conditional_edges from planner → route_to_analysts → analyst
    implements dynamic fan-out using the Send API.
    """
    graph = StateGraph(PortfolioState)

    graph.add_node("planner", planner_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.add_edge(START, "planner")
    # Dynamic fan-out: route_to_analysts returns list[Send] at runtime
    graph.add_conditional_edges("planner", route_to_analysts, ["analyst"])
    graph.add_edge("analyst", "synthesizer")
    graph.add_edge("synthesizer", END)

    checkpointer = MemorySaver() if use_checkpointer else None
    return graph.compile(checkpointer=checkpointer)


# ── Main entrypoint ────────────────────────────────────────────────────────────
async def run_portfolio_agent(
    query: str,
    thread_id: str = "default",
) -> dict:
    """Run the full portfolio research pipeline."""
    app = build_portfolio_graph()
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: PortfolioState = {
        "user_query": query,
        "tickers": [],
        "analyst_outputs": [],
        "cost_usd": 0.0,
        "portfolio_weights": None,
        "final_report": None,
        "optimization_stats": None,
    }

    result = await app.ainvoke(initial_state, config=config)
    return result


# ── Helpers ────────────────────────────────────────────────────────────────────
def _estimate_cost(response) -> float:
    """Rough token cost estimate (gpt-4o-mini pricing)."""
    if hasattr(response, "usage_metadata"):
        tokens = response.usage_metadata.get("total_tokens", 0)
        return tokens * 0.00000015  # ~$0.15 per 1M tokens
    return 0.0


if __name__ == "__main__":
    result = asyncio.run(
        run_portfolio_agent(
            "用 10 万美元构建一个防御性科技投资组合。"
            "重点关注具有强劲现金流和人工智能敞口的公司"
        )
    )
    print("\n" + "="*60)
    print("PORTFOLIO WEIGHTS:")
    for ticker, weight in result["portfolio_weights"].items():
        print(f"  {ticker}: {weight:.1%}")
    print("\nOPTIMIZATION STATS:")
    print(json.dumps(result["optimization_stats"], indent=2))
    print("\nFINAL REPORT:")
    print(result["final_report"])
    print(f"\nTotal cost: ${result['cost_usd']:.4f}")
