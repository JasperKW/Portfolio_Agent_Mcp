"""
src/mcp/financial_server.py
-----------------------------
MCP Server: exposes financial data tools over Streamable HTTP.

Pattern: FastMCP @mcp.tool() decorator (S7 §5 — "no JSON-Schema-by-hand").
FastMCP introspects type hints + docstring → generates inputSchema automatically.

Transport choice: Streamable HTTP (not stdio) because:
  - Multiple Analyst agents connect concurrently (multi-tenant)
  - Can sit behind a load balancer
  - Sessions survive transient disconnects via Mcp-Session-Id header
  (S7 §4.2: "Best fit: enterprise SaaS, multi-user systems")

Start the server:
    python -m src.mcp.financial_server          # default port 8001
    MCP_PORT=8002 python -m src.mcp.financial_server

Then verify with MCP Inspector:
    npx @modelcontextprotocol/inspector http://localhost:8001/mcp
"""

from __future__ import annotations
import os
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from fastmcp import FastMCP
from pypfopt import BlackLittermanModel, EfficientFrontier, risk_models
from pypfopt.black_litterman import market_implied_prior_returns

load_dotenv()

# FastMCP server instance — name shows up in MCP Inspector + client logs
mcp = FastMCP("financial-data-server")


# ── Tool 1: Price History ─────────────────────────────────────────────────────
@mcp.tool()
def get_price_history(ticker: str, period_days: int = 252) -> dict:
    """
    Fetch historical price data and return return statistics.

    Args:
        ticker: Stock ticker symbol, e.g. NVDA, MSFT, AAPL
        period_days: Lookback window in calendar days (default 252 ≈ 1 trading year)

    Returns dict with: current_price, annualised_return, annualised_volatility,
    sharpe_ratio, correlation_with_spy, daily_returns_last5
    """
    end = datetime.today()
    start = end - timedelta(days=period_days + 30)

    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        return {"error": f"No price data found for {ticker}"}

    prices = df["Close"].squeeze().dropna()
    daily_ret = prices.pct_change().dropna()

    spy = yf.download(
        "SPY", start=start, end=end, progress=False, auto_adjust=True
    )["Close"].squeeze().pct_change().dropna()

    common = daily_ret.index.intersection(spy.index)
    corr_spy = float(daily_ret[common].corr(spy[common]))

    return {
        "ticker": ticker,
        "current_price": round(float(prices.iloc[-1]), 2),
        "annualised_return": round(float(daily_ret.mean() * 252), 4),
        "annualised_volatility": round(float(daily_ret.std() * np.sqrt(252)), 4),
        "sharpe_ratio": round(
            float(daily_ret.mean() / daily_ret.std() * np.sqrt(252)), 2
        ),
        "correlation_with_spy": round(corr_spy, 3),
        "period_days": len(daily_ret),
        "daily_returns_last5": daily_ret.tail(5).round(4).tolist(),
    }


# ── Tool 2: Technical Signals ────────────────────────────────────────────────
@mcp.tool()
def get_technical_signals(ticker: str) -> dict:
    """
    Compute technical indicators: RSI(14), MA crossover, 3-month momentum.

    Args:
        ticker: Stock ticker symbol

    Returns composite signal (bullish/neutral/bearish) plus raw indicator values.
    """
    df = yf.download(ticker, period="6mo", progress=False, auto_adjust=True)
    if df.empty:
        return {"error": f"No data for {ticker}"}

    close = df["Close"].squeeze()

    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = float((100 - 100 / (1 + gain / loss)).iloc[-1])

    # MA crossover
    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma50 = float(close.rolling(50).mean().iloc[-1])

    # 3-month momentum
    mom_3m = float(close.iloc[-1] / close.iloc[-63] - 1) if len(close) >= 63 else 0.0

    bullish_count = sum([rsi < 70, rsi > 40, ma20 > ma50, mom_3m > 0])
    signal = "bullish" if bullish_count >= 3 else ("bearish" if bullish_count <= 1 else "neutral")

    return {
        "ticker": ticker,
        "rsi_14": round(rsi, 1),
        "ma20": round(ma20, 2),
        "ma50": round(ma50, 2),
        "ma_crossover": "bullish" if ma20 > ma50 else "bearish",
        "momentum_3m": round(mom_3m, 4),
        "composite_signal": signal,
        "signal_confidence": f"{bullish_count}/4 indicators bullish",
    }


# ── Tool 3: Black-Litterman Optimizer ────────────────────────────────────────
@mcp.tool()
def optimize_portfolio(
    tickers: list[str],
    views: list[dict],
    risk_aversion: float = 2.5,
) -> dict:
    """
    Run Black-Litterman optimization given analyst views.

    Args:
        tickers: List of ticker symbols in the portfolio, e.g. ["NVDA", "MSFT"]
        views: List of analyst views. Each view: {
            "ticker": "NVDA",
            "expected_alpha": 0.05,   # expected excess return (5% = 0.05)
            "confidence": 0.7          # confidence 0.0-1.0
        }
        risk_aversion: Risk aversion parameter delta (default 2.5)

    Returns optimal weights, expected return, volatility, Sharpe ratio.
    """
    prices = yf.download(
        tickers, period="1y", progress=False, auto_adjust=True
    )["Close"].dropna()

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

    # Market caps for B-L prior
    caps = {}
    for t in tickers:
        try:
            caps[t] = float(yf.Ticker(t).fast_info.market_cap or 1e11)
        except Exception:
            caps[t] = 1e11

    market_prices = yf.download(
        "SPY", period="1y", progress=False, auto_adjust=True
    )["Close"].squeeze().dropna()

    pi = market_implied_prior_returns(
        market_caps=caps, risk_aversion=risk_aversion, cov_matrix=S
    )

    # Build P, Q, Ω from views
    P, Q, Omega = _build_view_matrices(views, tickers, S)

    bl = BlackLittermanModel(S, pi=pi, P=P, Q=Q, omega=Omega)
    ef = EfficientFrontier(bl.bl_returns(), bl.bl_cov())
    ef.add_constraint(lambda w: w >= 0.02)
    ef.add_constraint(lambda w: w <= 0.40)
    ef.max_sharpe()
    weights = ef.clean_weights()
    perf = ef.portfolio_performance(verbose=False)

    return {
        "weights": {k: round(v, 4) for k, v in weights.items() if v > 0.001},
        "expected_annual_return": round(perf[0], 4),
        "annual_volatility": round(perf[1], 4),
        "sharpe_ratio": round(perf[2], 3),
        "views_applied": len(views),
        "optimization": "Black-Litterman + Max Sharpe",
    }


# ── Helper ────────────────────────────────────────────────────────────────────
def _build_view_matrices(views, tickers, S):
    n_v, n_a = len(views), len(tickers)
    idx = {t: i for i, t in enumerate(tickers)}
    P = np.zeros((n_v, n_a))
    Q = np.zeros(n_v)
    omega = np.zeros(n_v)
    for i, v in enumerate(views):
        t = v.get("ticker", "")
        if t not in idx:
            continue
        P[i, idx[t]] = 1.0
        Q[i] = float(v.get("expected_alpha", 0.0))
        conf = max(0.05, min(0.95, float(v.get("confidence", 0.5))))
        omega[i] = 0.05 * float(P[i] @ S.values @ P[i].T) * (1 / conf - 1)
    return P, Q, np.diag(omega)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("MCP_PORT", "8001"))
    print(f"Starting financial-data-server on http://localhost:{port}/mcp")
    print("Tools: get_price_history, get_technical_signals, optimize_portfolio")
    # transport="streamable-http" is the 2026 standard (replaces deprecated SSE)
    mcp.run(transport="streamable-http", host="0.0.0.0", port=port)
