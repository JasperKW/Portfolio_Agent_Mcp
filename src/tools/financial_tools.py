"""
src/tools/financial_tools.py
------------------------------
Financial Tools for the Portfolio Agent.

Three tool categories (S8 BaseTool pattern):
  1. PriceTool       — yfinance historical OHLCV + returns
  2. TechnicalTool   — RSI, Moving Averages, momentum signals
  3. OptimizerTool   — Black-Litterman portfolio construction

All tools follow the S8 ToolRegistry pattern: never raise, always return
a ToolResult so the LLM can self-correct on error.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from pypfopt import BlackLittermanModel, EfficientFrontier, risk_models, expected_returns
from pypfopt.black_litterman import market_implied_prior_returns


# ── S8 BaseTool pattern ───────────────────────────────────────────────────────
@dataclass
class ToolResult:
    tool_name: str
    input_args: dict
    success: bool
    output: Any = None
    error: Optional[str] = None

    def to_llm_str(self) -> str:
        """Serialise for the LLM observation turn."""
        if not self.success:
            return f"[Tool Error] {self.tool_name}: {self.error}"
        return json.dumps(self.output, indent=2, default=str)


class BaseTool:
    name: str = ""
    description: str = ""

    def to_openai_function(self) -> dict:
        raise NotImplementedError

    async def execute(self, **kwargs) -> ToolResult:
        raise NotImplementedError


# ── Tool 1: Price Data ────────────────────────────────────────────────────────
class PriceTool(BaseTool):
    """
    Fetch historical OHLCV data and compute return statistics.
    Used by Analyst agent to build the Σ (covariance matrix) for B-L.
    """
    name = "get_price_data"
    description = (
        "Fetch historical stock prices and return statistics for a ticker. "
        "Returns: daily returns, annualised return, volatility, correlation with SPY."
    )

    def to_openai_function(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string", "description": "Stock ticker, e.g. NVDA"},
                        "period_days": {"type": "integer", "description": "Lookback window in days (default 252)"},
                    },
                    "required": ["ticker"],
                    "additionalProperties": False,
                },
            },
        }

    async def execute(self, ticker: str, period_days: int = 252) -> ToolResult:
        try:
            end = datetime.today()
            start = end - timedelta(days=period_days + 30)  # buffer for weekends
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if df.empty:
                return ToolResult(self.name, {"ticker": ticker}, False,
                                  error=f"No data for {ticker}")

            prices = df["Close"].dropna()
            daily_ret = prices.pct_change().dropna()

            # SPY correlation
            spy = yf.download("SPY", start=start, end=end, progress=False,
                               auto_adjust=True)["Close"].pct_change().dropna()
            common = daily_ret.index.intersection(spy.index)
            corr_spy = float(daily_ret[common].corr(spy[common]))

            output = {
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
            return ToolResult(self.name, {"ticker": ticker, "period_days": period_days},
                              True, output=output)

        except Exception as e:
            return ToolResult(self.name, {"ticker": ticker}, False, error=str(e))


# ── Tool 2: Technical Indicators ─────────────────────────────────────────────
class TechnicalTool(BaseTool):
    """
    Compute technical signals: RSI, Moving Average crossover, momentum.
    Used to generate short-term signals that complement the fundamental Views.
    """
    name = "get_technical_signals"
    description = (
        "Compute technical indicators for a stock: RSI(14), MA20/MA50 crossover, "
        "3-month momentum. Returns a signal: 'bullish', 'bearish', or 'neutral'."
    )

    def to_openai_function(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string"},
                    },
                    "required": ["ticker"],
                    "additionalProperties": False,
                },
            },
        }

    async def execute(self, ticker: str) -> ToolResult:
        try:
            df = yf.download(ticker, period="6mo", progress=False, auto_adjust=True)
            if df.empty:
                return ToolResult(self.name, {"ticker": ticker}, False,
                                  error=f"No data for {ticker}")

            close = df["Close"].squeeze()

            # RSI(14)
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / loss
            rsi = float((100 - 100 / (1 + rs)).iloc[-1])

            # MA crossover
            ma20 = float(close.rolling(20).mean().iloc[-1])
            ma50 = float(close.rolling(50).mean().iloc[-1])
            ma_signal = "bullish" if ma20 > ma50 else "bearish"

            # 3-month momentum (63 trading days)
            mom_3m = float((close.iloc[-1] / close.iloc[-63] - 1) if len(close) >= 63 else 0)

            # Composite signal
            bullish_count = sum([
                rsi < 70,       # not overbought
                rsi > 40,       # not oversold (double-check)
                ma_signal == "bullish",
                mom_3m > 0,
            ])
            signal = "bullish" if bullish_count >= 3 else ("bearish" if bullish_count <= 1 else "neutral")

            output = {
                "ticker": ticker,
                "rsi_14": round(rsi, 1),
                "ma20": round(ma20, 2),
                "ma50": round(ma50, 2),
                "ma_crossover": ma_signal,
                "momentum_3m": round(mom_3m, 4),
                "composite_signal": signal,
                "signal_confidence": f"{bullish_count}/4 indicators bullish",
            }
            return ToolResult(self.name, {"ticker": ticker}, True, output=output)

        except Exception as e:
            return ToolResult(self.name, {"ticker": ticker}, False, error=str(e))


# ── Tool 3: Black-Litterman Optimizer ────────────────────────────────────────
class BlackLittermanTool(BaseTool):
    """
    Black-Litterman portfolio optimizer.
    
    Takes structured LLM Views and outputs optimal portfolio weights.
    
    B-L formula:
        μ_BL = [(τΣ)^{-1} + P'Ω^{-1}P]^{-1} [(τΣ)^{-1}π + P'Ω^{-1}Q]
    
    Where:
        π  = market-implied equilibrium returns (from CAPM)
        P  = pick matrix (which assets each view covers)
        Q  = view returns (LLM-generated)
        Ω  = view uncertainty (inverse of LLM confidence)
        τ  = scaling factor (~0.05)
    """
    name = "optimize_portfolio"
    description = (
        "Run Black-Litterman optimization given analyst views. "
        "Views format: [{'ticker': 'NVDA', 'expected_alpha': 0.05, 'confidence': 0.7}]. "
        "Returns: optimal weights, expected return, volatility, Sharpe ratio."
    )

    def to_openai_function(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "strict": False,   # views is a complex nested object
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tickers": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of tickers in the portfolio",
                        },
                        "views": {
                            "type": "array",
                            "description": "List of analyst views from RAG analysis",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "ticker": {"type": "string"},
                                    "expected_alpha": {"type": "number",
                                        "description": "Expected excess return, e.g. 0.05 = 5%"},
                                    "confidence": {"type": "number",
                                        "description": "Confidence 0-1; higher = lower Ω uncertainty"},
                                },
                            },
                        },
                        "risk_aversion": {
                            "type": "number",
                            "description": "Risk aversion parameter δ (default 2.5)",
                        },
                    },
                    "required": ["tickers", "views"],
                    "additionalProperties": False,
                },
            },
        }

    async def execute(
        self,
        tickers: list[str],
        views: list[dict],
        risk_aversion: float = 2.5,
    ) -> ToolResult:
        try:
            # 1. Fetch price history
            prices = yf.download(
                tickers, period="1y", progress=False, auto_adjust=True
            )["Close"].dropna()

            if prices.shape[1] != len(tickers):
                available = list(prices.columns)
                return ToolResult(self.name, {"tickers": tickers}, False,
                                  error=f"Only got data for {available}")

            # 2. Build Σ (covariance) and π (market-implied returns)
            S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
            market_prices = yf.download("SPY", period="1y", progress=False,
                                         auto_adjust=True)["Close"].dropna()
            pi = market_implied_prior_returns(
                market_caps=_get_market_caps(tickers),
                risk_aversion=risk_aversion,
                cov_matrix=S,
            )

            # 3. Build P and Q matrices from LLM views
            P, Q, Omega = _build_view_matrices(views, tickers, S)

            # 4. Black-Litterman posterior
            bl = BlackLittermanModel(S, pi=pi, P=P, Q=Q, omega=Omega)
            bl_returns = bl.bl_returns()
            bl_cov = bl.bl_cov()

            # 5. Mean-variance optimization on B-L posterior
            ef = EfficientFrontier(bl_returns, bl_cov)
            ef.add_constraint(lambda w: w >= 0.02)   # min 2% per position
            ef.add_constraint(lambda w: w <= 0.40)   # max 40% per position
            weights = ef.max_sharpe()
            cleaned = ef.clean_weights()

            perf = ef.portfolio_performance(verbose=False)

            output = {
                "weights": {k: round(v, 4) for k, v in cleaned.items() if v > 0.001},
                "expected_annual_return": round(perf[0], 4),
                "annual_volatility": round(perf[1], 4),
                "sharpe_ratio": round(perf[2], 3),
                "views_applied": len(views),
                "optimization": "Black-Litterman + Max Sharpe",
            }
            return ToolResult(self.name, {"tickers": tickers}, True, output=output)

        except Exception as e:
            return ToolResult(self.name, {"tickers": tickers}, False, error=str(e))


# ── Helpers ────────────────────────────────────────────────────────────────────
def _get_market_caps(tickers: list[str]) -> dict[str, float]:
    """Fetch market caps from yfinance for the B-L prior."""
    caps = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).fast_info
            caps[t] = float(info.market_cap or 1e11)
        except Exception:
            caps[t] = 1e11  # fallback: $100B
    return caps


def _build_view_matrices(
    views: list[dict],
    tickers: list[str],
    S: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert LLM views → Black-Litterman P, Q, Ω matrices.
    
    Each view: {"ticker": "NVDA", "expected_alpha": 0.05, "confidence": 0.7}
    
    P: (n_views × n_assets) pick matrix  — 1 for the asset in each view
    Q: (n_views,) expected excess returns
    Ω: (n_views × n_views) diagonal view uncertainty
         Ω_ii = (1/confidence_i - 1) * P_i Σ P_i'   (He-Litterman formula)
    """
    n_views = len(views)
    n_assets = len(tickers)
    ticker_idx = {t: i for i, t in enumerate(tickers)}

    P = np.zeros((n_views, n_assets))
    Q = np.zeros(n_views)
    omega_diag = np.zeros(n_views)

    for i, v in enumerate(views):
        t = v["ticker"]
        if t not in ticker_idx:
            continue
        P[i, ticker_idx[t]] = 1.0
        Q[i] = float(v.get("expected_alpha", 0.0))

        # Ω_ii ∝ 1/confidence — high confidence → small uncertainty
        conf = float(v.get("confidence", 0.5))
        conf = max(0.05, min(0.95, conf))   # clamp to (0.05, 0.95)
        tau = 0.05
        variance = tau * float(P[i] @ S.values @ P[i].T)
        omega_diag[i] = variance * (1 / conf - 1)

    return P, Q, np.diag(omega_diag)


# ── Tool Registry (S8 pattern) ───────────────────────────────────────────────
class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get_openai_tools(self) -> list[dict]:
        return [t.to_openai_function() for t in self._tools.values()]

    async def execute(self, name: str, arguments: dict) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(name, arguments, False,
                              error=f"Unknown tool: {name}. Available: {list(self._tools)}")
        return await tool.execute(**arguments)


# Default registry with all financial tools
tool_registry = ToolRegistry()
tool_registry.register(PriceTool())
tool_registry.register(TechnicalTool())
tool_registry.register(BlackLittermanTool())
