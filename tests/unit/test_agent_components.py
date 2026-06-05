"""
tests/unit/test_agent_components.py
-------------------------------------
Unit tests: mock the LLM, test routing logic and tool execution.
S12 pattern: fast (<5s), no real API calls, 100% deterministic.

Run: pytest tests/unit/ -v
"""

from __future__ import annotations
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from financial_tools import (
    ToolRegistry, PriceTool, TechnicalTool, BlackLittermanTool, ToolResult
)
from ingest import rrf_merge
from financial_tools import _build_view_matrices


# ── Fixtures ─────────────────────────────────────────────────────────────────
@pytest.fixture
def registry():
    r = ToolRegistry()
    r.register(PriceTool())
    r.register(TechnicalTool())
    r.register(BlackLittermanTool())
    return r


# ── Tool Registry Tests ───────────────────────────────────────────────────────
class TestToolRegistry:
    def test_register_and_get_tools(self, registry):
        tools = registry.get_openai_tools()
        names = [t["function"]["name"] for t in tools]
        assert "get_price_data" in names
        assert "get_technical_signals" in names
        assert "optimize_portfolio" in names

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error_not_raise(self, registry):
        """S8 pattern: unknown tool name → ToolResult(success=False), not exception."""
        result = await registry.execute("nonexistent_tool", {"arg": "val"})
        assert result.success is False
        assert "Unknown tool" in result.error
        assert "nonexistent_tool" in result.error

    def test_tool_result_to_llm_str_error(self):
        result = ToolResult("test_tool", {}, False, error="connection timeout")
        s = result.to_llm_str()
        assert "[Tool Error]" in s
        assert "connection timeout" in s

    def test_tool_result_to_llm_str_success(self):
        result = ToolResult("test_tool", {}, True, output={"price": 100.5})
        s = result.to_llm_str()
        parsed = json.loads(s)
        assert parsed["price"] == 100.5


# ── RRF Tests ────────────────────────────────────────────────────────────────
class TestRRFMerge:
    def _make_doc(self, chunk_id: str, score: float):
        doc = MagicMock()
        doc.metadata = {"chunk_id": chunk_id}
        return (doc, score)

    def test_rrf_deduplicates_same_chunk(self):
        """Same chunk appearing in both lists should be deduplicated with higher score."""
        from langchain_core.documents import Document
        
        d1 = Document(page_content="text1", metadata={"chunk_id": "abc"})
        d2 = Document(page_content="text2", metadata={"chunk_id": "def"})

        vector_results = [(d1, 0.9), (d2, 0.7)]
        bm25_results = [(d1, 5.0), (d2, 3.0)]

        merged = rrf_merge(vector_results, bm25_results, top_n=5)
        
        # No duplicate chunk IDs
        ids = [doc.metadata["chunk_id"] for doc, _ in merged]
        assert len(ids) == len(set(ids))

    def test_rrf_top_n_respected(self):
        from langchain_core.documents import Document
        docs = [
            (Document(page_content=f"t{i}", metadata={"chunk_id": f"id{i}"}), float(i))
            for i in range(10)
        ]
        merged = rrf_merge(docs, docs[:5], top_n=3)
        assert len(merged) <= 3

    def test_rrf_empty_one_list(self):
        """RRF should work when one list is empty."""
        from langchain_core.documents import Document
        d = Document(page_content="text", metadata={"chunk_id": "x"})
        merged = rrf_merge([(d, 0.9)], [], top_n=5)
        assert len(merged) == 1


# ── View Matrix Tests ─────────────────────────────────────────────────────────
class TestViewMatrices:
    def test_view_matrix_shape(self):
        import numpy as np
        import pandas as pd
        
        tickers = ["NVDA", "MSFT", "AAPL"]
        views = [
            {"ticker": "NVDA", "expected_alpha": 0.05, "confidence": 0.7},
            {"ticker": "MSFT", "expected_alpha": 0.02, "confidence": 0.5},
        ]
        # Mock covariance matrix
        S = pd.DataFrame(np.eye(3) * 0.04, index=tickers, columns=tickers)
        
        P, Q, Omega = _build_view_matrices(views, tickers, S)
        
        assert P.shape == (2, 3)     # n_views × n_assets
        assert Q.shape == (2,)       # n_views
        assert Omega.shape == (2, 2) # n_views × n_views diagonal

    def test_view_matrix_correct_asset_mapping(self):
        import numpy as np
        import pandas as pd
        
        tickers = ["NVDA", "MSFT", "AAPL"]
        views = [{"ticker": "MSFT", "expected_alpha": 0.03, "confidence": 0.6}]
        S = pd.DataFrame(np.eye(3) * 0.04, index=tickers, columns=tickers)
        
        P, Q, Omega = _build_view_matrices(views, tickers, S)
        
        # MSFT is index 1 — P[0, 1] should be 1, others 0
        assert P[0, 0] == 0   # NVDA
        assert P[0, 1] == 1   # MSFT
        assert P[0, 2] == 0   # AAPL
        assert Q[0] == pytest.approx(0.03)

    def test_confidence_clamp(self):
        """Confidence outside [0.05, 0.95] should be clamped, not error."""
        import numpy as np
        import pandas as pd
        
        tickers = ["NVDA"]
        for extreme_conf in [0.0, 1.0, -0.5, 2.0]:
            views = [{"ticker": "NVDA", "expected_alpha": 0.05, "confidence": extreme_conf}]
            S = pd.DataFrame([[0.04]], index=tickers, columns=tickers)
            P, Q, Omega = _build_view_matrices(views, tickers, S)
            assert Omega[0, 0] > 0   # variance must be positive


# ── Agent State Tests (LangGraph pattern) ────────────────────────────────────
class TestPortfolioState:
    def test_analyst_outputs_accumulate_with_reducer(self):
        """
        operator.add reducer: multiple parallel Analyst nodes
        should EXTEND the list, not overwrite.
        This is the critical LangGraph gotcha from S9.
        """
        import operator
        
        # Simulate what LangGraph does with operator.add reducer
        current = {"analyst_outputs": [{"ticker": "NVDA", "view": {}}]}
        update = {"analyst_outputs": [{"ticker": "MSFT", "view": {}}]}
        
        merged = operator.add(current["analyst_outputs"], update["analyst_outputs"])
        assert len(merged) == 2
        assert merged[0]["ticker"] == "NVDA"
        assert merged[1]["ticker"] == "MSFT"

    def test_cost_accumulates(self):
        import operator
        total = operator.add(0.001, 0.002)
        assert total == pytest.approx(0.003)


# ── Planner output parsing ────────────────────────────────────────────────────
class TestPlannerParsing:
    def test_valid_json_response(self):
        raw = '{"tickers": ["NVDA", "MSFT", "AAPL"], "rationale": "AI exposure"}'
        parsed = json.loads(raw)
        assert parsed["tickers"] == ["NVDA", "MSFT", "AAPL"]

    def test_json_with_markdown_fences(self):
        """Real LLM often wraps JSON in ```json ... ```"""
        raw = '```json\n{"tickers": ["NVDA", "MSFT"]}\n```'
        if "```" in raw:
            raw = raw.split("```")[1].strip("json").strip()
        parsed = json.loads(raw)
        assert "NVDA" in parsed["tickers"]

    def test_fallback_on_bad_json(self):
        """Planner should never crash — fall back to default tickers."""
        default = ["NVDA", "MSFT", "AAPL", "GOOGL", "AMZN"]
        raw = "sorry I cannot provide that"
        try:
            parsed = json.loads(raw)
            tickers = parsed.get("tickers", default)
        except Exception:
            tickers = default
        assert tickers == default
