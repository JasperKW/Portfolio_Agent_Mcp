# Portfolio Research Agent

> AI-Powered Portfolio Construction via RAG + MCP Tools + Multi-Agent Orchestration

A multi-agent system that ingests SEC financial filings (10-K), pulls real-time price and technical signals via MCP, and uses Black-Litterman optimization to output portfolio weights — with a three-layer eval harness.

---

## Architecture

```
User Input: "Build a defensive tech portfolio with $100k, focus on AI exposure"
                           ↓
                    ┌──────────────┐
                    │   Planner    │  LLM selects 3-5 tickers
                    └──────┬───────┘
                           ↓  LangGraph Send API (parallel fan-out)
              ┌────────────┼────────────┐
              ↓            ↓            ↓
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ Analyst  │ │ Analyst  │ │ Analyst  │
        │  NVDA    │ │  MSFT    │ │  AAPL    │
        │          │ │          │ │          │
        │ RAG ──→  │ │ RAG ──→  │ │ RAG ──→  │  Retrieve 10-K context
        │ MCP ──→  │ │ MCP ──→  │ │ MCP ──→  │  Price + technical signals
        │ LLM ──→  │ │ LLM ──→  │ │ LLM ──→  │  Generate structured View
        │ View     │ │ View     │ │ View     │
        └────┬─────┘ └────┬─────┘ └────┬─────┘
             └─────────────┼─────────────┘
                           ↓  operator.add reducer
                    ┌──────────────┐
                    │ Synthesizer  │  B-L optimization → weights
                    │              │  LLM → investment memo (forced citations)
                    └──────────────┘
                           ↓
                    Portfolio Weights + Report
```

---

## S18 Requirements Compliance

### 2.1 The Agent — 4 of 6 Components Used

| Component | Where | What it does |
|-----------|-------|-------------|
| **RAG** | `src/rag/` | Parent-child chunking, hybrid BM25+FAISS retrieval, RRF merge |
| **MCP** | `src/mcp/` | FastMCP server exposes financial tools over Streamable HTTP |
| **Tools** | `src/tools/` | yfinance price data, technical indicators, Black-Litterman optimizer |
| **Multi-agent** | `src/agents/` | LangGraph: Planner → parallel Analysts → Synthesizer |

### 2.2 Eval — All 4 Builds Complete

| Build | Output | Status |
|-------|--------|--------|
| 1. Golden set | `data/golden_set/golden.jsonl` | ✅ 25 cases with `input / expected_facts / forbidden_facts / tags` |
| 2. Unit tests | `tests/unit/test_agent_components.py` | ✅ **15 passed**, 0 failed — tool routing, error paths, RRF, B-L math, LangGraph reducer |
| 3. Semantic metric | RAGAS faithfulness + context recall | ✅ Faithfulness = **1.000** (Round 0), 0.796 (Round 1) |
| 4. LLM-judge | `eval/judge_eval.py` with Cohen's κ | ✅ Max κ = **0.774**, all 4 dimensions ≥ 0.6 ✓ |

---

## Module 1: RAG Pipeline

### Design: Parent-Child Chunking (fin-rag-lab §4.5 pattern)

The core insight: small chunks embed well for retrieval, but large chunks give the LLM enough context to answer correctly. So we split twice:

```
10-K HTML → Parse (BeautifulSoup, ~100 pages)
         → Parent chunks (800 tokens) — for LLM generation context
         → Child chunks (200 tokens) — for precise retrieval matching
```

At query time: retrieve using child embeddings, but return the parent text to the LLM.

### Hybrid Retrieval: FAISS + BM25 → RRF

Vector search alone misses exact terms like "H100" or "$215.9B". BM25 keyword search alone misses semantic paraphrases. We use both and merge with Reciprocal Rank Fusion:

```
Score(doc) = 1/(60 + rank_vector) + 1/(60 + rank_bm25)
```

RRF avoids score normalization — cosine similarity and BM25 scores are on incompatible scales.

### Index Statistics

| Ticker | Parents | Children | Source |
|--------|---------|----------|--------|
| NVDA | 599 | 2,563 | FY2026 10-K (Jan 2026) |
| MSFT | 534 | 2,353 | FY2025 10-K (Jun 2025) |
| AAPL | 347 | 1,500 | FY2024 10-K (Sep 2024) |
| AMZN | 500 | 2,127 | FY2025 10-K (Dec 2025) |
| GOOGL | 593 | 2,594 | FY2025 10-K (Dec 2025) |

---

## Module 2: Financial Tools (via MCP)

### MCP Server Architecture

Tools are exposed as a standalone MCP server (`financial-data-server`) over Streamable HTTP, not hardcoded into the agent process:

```
┌──────────────┐    JSON-RPC/HTTP     ┌─────────────────────┐
│ Analyst Agent│ ──────────────────→  │ financial-data-server│
│ (MCP Client) │ ←──────────────────  │  (MCP Server)        │
│              │   localhost:8001     │                      │
│ Discovers    │                      │ get_price_history()  │
│ tools at     │                      │ get_technical_signals│
│ runtime      │                      │ optimize_portfolio() │
└──────────────┘                      └─────────────────────┘
```

Why MCP instead of direct function calls:
- **Dynamic discovery**: Agent calls `tools/list` at startup, doesn't hardcode tool names
- **Decoupled deployment**: Server can be updated without changing agent code
- **Graceful fallback**: If MCP server is unreachable, agent falls back to local Python tools automatically

### Three Tools

**get_price_history** — Fetches historical OHLCV from yfinance, computes annualized return, volatility, Sharpe ratio, and correlation with SPY. Used to build the covariance matrix Σ for Black-Litterman.

**get_technical_signals** — Computes RSI(14), MA20/MA50 crossover, and 3-month momentum. Outputs a composite signal (bullish/neutral/bearish) via 4-indicator vote.

**optimize_portfolio** — Black-Litterman optimizer. Takes LLM-generated Views and converts them to optimal portfolio weights:

```
LLM View: {"ticker": "NVDA", "expected_alpha": 0.04, "confidence": 0.75}
    ↓
P matrix (pick matrix): which asset each view covers
Q vector (expected returns): from expected_alpha
Ω matrix (uncertainty): derived from confidence — high confidence → low Ω → stronger effect
    ↓
B-L posterior: μ_BL = [(τΣ)^{-1} + P'Ω^{-1}P]^{-1} [(τΣ)^{-1}π + P'Ω^{-1}Q]
    ↓
Mean-variance optimization → weights w*
```

---

## Module 3: Multi-Agent Orchestration

### LangGraph State Machine (S9 Pattern B — Supervisor + Parallel Fan-out)

```
START → planner → [analyst × N via Send] → synthesizer → END
```

Key design decisions:

**State reducers**: `analyst_outputs: Annotated[list, operator.add]` — the `operator.add` reducer ensures parallel Analyst nodes extend (not overwrite) the shared list. This is the most common LangGraph bug (S9 §2.3).

**Dynamic fan-out via Send API**: The number of Analyst nodes is determined at runtime by the Planner, not at compile time. `route_to_analysts()` returns `list[Send]`, one per ticker.

**Graceful degradation**: If RAG has no index for a ticker, the Analyst uses market data only and notes "data insufficient" in its View, setting confidence=0.3 so B-L gives it minimal weight.

### Sample Output

```
PORTFOLIO WEIGHTS:
  AAPL: 40.0%
  MSFT: 32.6%
  NVDA: 27.4%

OPTIMIZATION STATS:
  Expected Annual Return: 4.97%
  Annual Volatility:      18.21%
  Sharpe Ratio:           0.273
  Method: Black-Litterman + Max Sharpe

Total cost: $0.0007
```

---

## Eval Harness

### Layer 1: RAGAS — RAG Quality (`eval/ragas_eval.py`)

Tests whether agent's financial claims are grounded in retrieved documents.

| Metric | Baseline (Round 0) | Improved (Round 1) | Change |
|--------|--------------------|--------------------|--------|
| Faithfulness | **1.000** | 0.796 | ↓ -0.204 |
| Answer Relevancy | 0.248 | **0.316** | ↑ +0.068 |
| Context Precision | 0.289 | 0.267 | ↓ -0.022 |
| Context Recall | 0.333 | 0.333 | → |
| Per-claim hallucinations | **0/18** | 13/52 | ↑ worse |

**Experiment design**: Round 0 uses a conservative prompt ("answer using ONLY the provided context"). Round 1 uses a detailed analyst prompt encouraging comprehensive answers.

**Key finding**: More detailed prompts improve Answer Relevancy (+29%) but decrease Faithfulness (-20%), introducing 13 unsupported claims. For financial applications, Faithfulness takes priority — one fabricated number can lead to incorrect investment decisions. **Decision: retain Round 0.**

### Layer 2: Backtest — Signal Quality (`eval/backtest_eval.py`)

Tests whether agent's investment views translate to portfolio alpha using historical returns.

| Experiment | 90-day Return | Excess vs SPY | Sharpe | Max Drawdown |
|-----------|---------------|---------------|--------|-------------|
| A: Equal weight (control) | 15.66% | +9.29% | 3.33 | -3.65% |
| B: Price signals only | 15.58% | +9.21% | 3.33 | -3.50% |
| **C: RAG Agent** | **18.15%** | **+11.78%** | 3.24 | -3.59% |

**Key findings**:
- RAG Agent (C) outperforms equal weight (A) by +2.5pp in 90-day return — fundamental analysis adds real alpha
- Pure price signals (B) provide no improvement over equal weight, suggesting momentum was not informative in this period
- C's slightly lower Sharpe reflects higher concentration on conviction names — a classic active management characteristic

### Layer 3: LLM-Judge — Reasoning Quality (`eval/judge_eval.py`)

Two independent LLM judges (gpt-4o-mini as primary, gpt-4o as secondary) score each analysis on a 1–5 scale with anchored rubrics (BARS), then Cohen's kappa measures inter-judge agreement across 25 cases.

**Final Results (Round 3 — forced citation format):**

| Dimension | κ (primary vs secondary) | Gate (≥0.6) | Primary mean score |
|-----------|--------------------------|-------------|-------------------|
| data_grounding | 0.606 | ✓ | 1.52/5 |
| logic_coherence | 0.774 | ✓ | 2.04/5 |
| citation_quality | 0.725 | ✓ | 1.12/5 |
| overall | 0.625 | ✓ | 1.60/5 |
| **Max κ** | **0.774** | **PASSED ✓** | |

---

## Experiment Log (S18 §2.3)

> **S18 §2.3 pattern followed**: Baseline → Optimize → N rounds of iteration.
> Each experiment starts with a naive baseline, makes one explicit change, and records whether the metric moved up, down, or not at all.
> A negative result (change made things worse) is treated as a valid conclusion, not hidden.

### Layer 1 (RAGAS) Experiment

**Baseline**: conservative prompt → **Optimize Round 1**: detailed analyst prompt → **Decision**: revert (negative result)

| Round | Change | Metric delta | Conclusion |
|-------|--------|-------------|------------|
| 0 — Baseline | Conservative prompt: "answer ONLY from context" | Faithfulness = 1.000, Relevancy = 0.248 | Zero hallucinations but low relevancy |
| 1 — Detailed prompt | Full analyst prompt encouraging comprehensive answers | Faithfulness = 0.796 (↓), Relevancy = 0.316 (↑) | Relevancy improved but 13 hallucinations introduced — unacceptable for finance |
| **Decision** | Revert to Round 0 | Faithfulness > Relevancy for financial use case | "A negative result is still a result" (S18) |

### Layer 3 (LLM-Judge) Experiment

**Baseline**: vague judge prompt → **Optimize Round 2**: BARS anchors → **Optimize Round 3**: fix agent output

| Round | Change | Metric delta | Conclusion |
|-------|--------|-------------|------------|
| 1 — Baseline judge | Vague rubric: only described scores 1 and 5 | Max κ = 0.619; citation_quality κ = 0.151 — 3/4 dims failed | Under-specified rubric; judges guessed differently on middle scores |
| 2 — BARS prompt | Explicit anchor for every score level (e.g. score 3 = "revenue was about $200B") | Max κ = 0.688; data_grounding fixed ✓, citation_quality still 0.457 ✗ | Judge agreement improved but citation_quality still failing — root cause is agent output, not judge |
| 3 — Forced citation | Added `## CITATION RULES` block to synthesizer prompt with mandatory format `number (source: 10-K p.XX)` | Max κ = 0.774; **all 4 dims ≥ 0.6 ✓** | Fixing agent output upstream solved what prompt engineering downstream could not |

**Cross-layer finding**: Round 2 was a negative result that revealed the real problem. The judge prompt improvement was necessary but not sufficient — the agent itself needed to produce citable output before judges could consistently evaluate it.

---

## Failure Analysis (S18 §2.4)

**Case 1 — Citation quality κ passes but mean score stays low (1.12/5)**

Both judges consistently agree the agent's citations are weak — they see "10-K" as a source but rarely a page number. The fix is to pass page-level metadata from RAG chunks through the analyst output, so the synthesizer has specific page numbers to cite. Not yet implemented.

**Case 2 — Planner selects tickers without indexed filings (e.g. ORCL, ADBE)**

Early runs had the Planner freely select tickers with no 10-K data. RAG gracefully returns empty results, but the Analyst then fabricates fundamentals. Partial fix: analyst prompt now states "insufficient data" when RAG returns nothing, setting confidence=0.3 so B-L minimizes the position.

**Case 3 — B-L optimizer falls back to equal weight**

When yfinance returns incomplete covariance data (ticker with <252 days of history), PyPortfolioOpt raises an exception. Synthesizer catches this and falls back to equal weights — safe but loses the B-L signal entirely for that run.

---

## Trade-off Statement

Forcing citation format (Round 3) improved `citation_quality` κ from 0.457 → 0.725, but mean scores dropped across all dimensions (e.g. `logic_coherence` mean: 3.87 → 2.04). The stricter rubric revealed that agent output quality was lower than the vague rubric suggested — **this is a more honest measurement, not a regression**. Quality / measurement accuracy were traded against apparent score inflation.

---

## Quick Start

```bash
# 1. Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add OPENAI_API_KEY

# 2. Index documents
export PYTHONPATH=$(pwd)
python -m src.rag.ingest --tickers NVDA MSFT AAPL AMZN GOOGL

# 3. Start MCP Server (separate terminal)
python -m src.mcp.financial_server

# 4. Run Agent
python -m src.agents.portfolio_graph

# 5. Run unit tests
.venv/bin/pytest tests/unit/test_agent_components.py -v

# 6. Run Eval (separate venv for ragas compatibility)
python3 -m venv .venv_eval && source .venv_eval/bin/activate
pip install ragas==0.1.21 langchain==0.2.17 langchain-community==0.2.17 langchain-openai==0.1.25
python eval/ragas_eval.py --experiment baseline
source .venv/bin/activate
python eval/backtest_eval.py
python eval/judge_eval.py
```

---

## Directory Structure

```
portfolio_agent_MCP/
├── src/
│   ├── rag/
│   │   ├── ingest.py           # PDF/HTML parse → parent-child chunk → FAISS+BM25 index
│   │   └── retriever.py        # Hybrid retrieval with RRF fusion
│   ├── mcp/
│   │   ├── financial_server.py # FastMCP server (3 tools, Streamable HTTP)
│   │   └── client.py           # MCPToolAdapter + dynamic tool discovery
│   ├── tools/
│   │   └── financial_tools.py  # PriceTool, TechnicalTool, BlackLittermanTool, ToolRegistry
│   └── agents/
│       └── portfolio_graph.py  # LangGraph: Planner → Analyst×N → Synthesizer
├── eval/
│   ├── ragas_eval.py           # Layer 1: RAGAS faithfulness + per-claim check
│   ├── backtest_eval.py        # Layer 2: Historical signal accuracy (A/B/C)
│   ├── judge_eval.py           # Layer 3: LLM judge + Cohen's kappa
│   └── run_all_evals.py        # Master eval runner
├── data/
│   ├── filings/                # SEC 10-K HTML files
│   ├── golden_set/
│   │   └── golden.jsonl        # 25 eval cases
│   └── index/                  # FAISS + BM25 indices per ticker
├── tests/unit/
│   ├── conftest.py
│   └── test_agent_components.py  # 15 pytest tests, no LLM calls
├── .env                        # API keys (gitignored)
├── requirements.txt
└── README.md
```

---

## Engineering Decisions Log

| Problem | Decision | Rationale |
|---------|----------|-----------|
| PDF extraction quality | Switched to HTML from SEC EDGAR | PyMuPDF couldn't extract text from SEC's interactive PDFs |
| Pickle deserialization errors | Switched to JSON for metadata storage | Pickle depends on exact class paths across environments; JSON is portable |
| ragas + langgraph version conflict | Separate `.venv_eval` environment | ragas requires langchain 0.2.x; langgraph requires 0.3.x+. Cleanest isolation |
| Planner selects stocks without indexes | Allow free selection, RAG degrades gracefully | Agent flexibility > forcing predefined tickers; confidence penalizes missing data |
| Round 1 prompt increased hallucinations | Reverted to Round 0 conservative prompt | Faithfulness > Relevancy for financial applications |
| citation_quality κ failing after BARS | Fixed synthesizer prompt, not judge prompt | Root cause was agent output lacking citations, not judge ambiguity |

---

## Known Limitations

1. **RAG Context Precision (0.289)**: SEC filings contain heavy table/legal boilerplate that reduces retrieval precision. A reranker (Cohere, ColBERT) would improve this.
2. **Eval environment split**: ragas requires langchain 0.2.x while langgraph requires 0.3.x+. Two virtual environments are required.
3. **Backtest lookahead risk**: LLM training data may include historical financial results, making "predictions" partially memory-based rather than true inference.
4. **Citation depth**: Synthesizer cites document type ("10-K") but rarely page numbers. Page-level metadata passthrough from RAG chunks is the next improvement.
