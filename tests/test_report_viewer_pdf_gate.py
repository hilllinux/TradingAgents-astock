"""Regression tests for report PDF controls with missing data."""

from __future__ import annotations

import importlib
import sys
import types
from copy import deepcopy
from typing import Any

import pytest


class _Context:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeStreamlit:
    def __init__(self) -> None:
        self.session_state: dict[str, Any] = {}
        self.buttons: list[dict[str, Any]] = []
        self.downloads: list[dict[str, Any]] = []
        self.captions: list[str] = []
        self.markdowns: list[str] = []
        self.messages: list[tuple[str, str]] = []
        self.button_returns: dict[str, bool] = {}
        self.events: list[tuple[str, str]] = []

    def columns(self, spec):
        count = len(spec) if isinstance(spec, list) else int(spec)
        return [_Context() for _ in range(count)]

    def download_button(self, label: str, **kwargs):
        self.downloads.append({"label": label, **kwargs})
        return False

    def button(self, label: str, **kwargs):
        self.buttons.append({"label": label, **kwargs})
        self.events.append(("button", label))
        return self.button_returns.get(str(kwargs.get("key")), False)

    def caption(self, text: str):
        self.captions.append(text)
        self.events.append(("caption", text))

    def markdown(self, text: str, *args, **kwargs):
        self.markdowns.append(str(text))
        self.events.append(("markdown", str(text)))
        return None

    def error(self, text: str, **kwargs):
        self.messages.append(("error", text))

    def warning(self, text: str, **kwargs):
        self.messages.append(("warning", text))

    def success(self, text: str, **kwargs):
        self.messages.append(("success", text))

    def info(self, text: str, **kwargs):
        self.messages.append(("info", text))

    def expander(self, *args, **kwargs):
        return _Context()

    def spinner(self, *args, **kwargs):
        return _Context()

    def tabs(self, labels):
        return [_Context() for _ in labels]

    def rerun(self):
        raise AssertionError("rerun should not be triggered in this test")


def _load_report_viewer(monkeypatch, fake_st: _FakeStreamlit):
    streamlit_stub = types.ModuleType("streamlit")
    for name in (
        "button",
        "caption",
        "columns",
        "download_button",
        "error",
        "expander",
        "info",
        "markdown",
        "rerun",
        "spinner",
        "success",
        "tabs",
        "warning",
    ):
        setattr(streamlit_stub, name, getattr(fake_st, name))
    streamlit_stub.session_state = fake_st.session_state

    messages_stub = types.ModuleType("langchain_core.messages")
    messages_stub.ToolMessage = type("ToolMessage", (), {})
    langchain_stub = types.ModuleType("langchain_core")
    langchain_stub.messages = messages_stub
    pdf_export_stub = types.ModuleType("web.pdf_export")
    pdf_export_stub.generate_markdown = lambda *args, **kwargs: "markdown"
    pdf_export_stub.generate_pdf = lambda *args, **kwargs: b"%PDF-stub"
    missing_data_stub = types.ModuleType("tradingagents.dataflows.missing_data")
    missing_data_stub.get_missing_tasks = lambda *args, **kwargs: []
    missing_data_stub.retry_missing_tasks = lambda *args, **kwargs: {
        "attempted": 0,
        "resolved_count": 0,
        "remaining_count": 0,
        "remaining": [],
    }
    stock_display_stub = types.ModuleType("web.stock_display")
    stock_display_stub.normalize_stock_mentions = (
        lambda text, ticker, final_state=None: text
    )
    stock_display_stub.stock_display_label = (
        lambda ticker, final_state=None: str(ticker)
    )
    dataflow_utils_stub = types.ModuleType("tradingagents.dataflows.utils")
    dataflow_utils_stub.safe_ticker_component = lambda value: str(value).strip().upper()

    monkeypatch.setitem(sys.modules, "streamlit", streamlit_stub)
    monkeypatch.setitem(sys.modules, "langchain_core", langchain_stub)
    monkeypatch.setitem(sys.modules, "langchain_core.messages", messages_stub)
    monkeypatch.setitem(sys.modules, "web.pdf_export", pdf_export_stub)
    monkeypatch.setitem(sys.modules, "tradingagents.dataflows.missing_data", missing_data_stub)
    monkeypatch.setitem(sys.modules, "tradingagents.dataflows.utils", dataflow_utils_stub)
    monkeypatch.setitem(sys.modules, "web.stock_display", stock_display_stub)
    sys.modules.pop("web.components.report_viewer", None)
    return importlib.import_module("web.components.report_viewer")


def _install_report_viewer_fakes(monkeypatch, report_viewer, pdf_calls: list[tuple[Any, ...]]) -> None:
    monkeypatch.setattr(report_viewer, "generate_markdown", lambda *args: "markdown")
    monkeypatch.setattr(
        report_viewer,
        "generate_pdf",
        lambda *args: pdf_calls.append(args) or b"%PDF-current-report",
    )
    monkeypatch.setattr(
        report_viewer,
        "get_missing_tasks",
        lambda ticker, trade_date, active_only=True: [
            {
                "status": "active",
                "stage_label": "基本面",
                "tool_label": "综合基本面",
                "missing_items": ["ROE"],
            }
        ],
    )


def _install_report_viewer_fakes_without_missing(
    monkeypatch,
    report_viewer,
    pdf_calls: list[tuple[Any, ...]],
) -> None:
    _install_report_viewer_fakes(monkeypatch, report_viewer, pdf_calls)
    monkeypatch.setattr(
        report_viewer,
        "get_missing_tasks",
        lambda ticker, trade_date, active_only=True: [],
    )


def test_pdf_is_generated_and_downloadable_when_missing_data_remains(monkeypatch):
    fake_st = _FakeStreamlit()
    report_viewer = _load_report_viewer(monkeypatch, fake_st)
    pdf_calls: list[tuple[Any, ...]] = []
    _install_report_viewer_fakes(monkeypatch, report_viewer, pdf_calls)

    report_viewer.render_report(
        {"missing_data_complete": False},
        ticker="300476",
        trade_date="2026-06-04",
        signal="HOLD",
    )

    assert len(pdf_calls) == 1
    assert any(
        download["label"] == "📄 下载 PDF"
        and download["data"] == b"%PDF-current-report"
        for download in fake_st.downloads
    )
    missing_buttons = [button for button in fake_st.buttons if button["label"] == "缺失项 (1)"]
    assert missing_buttons
    assert missing_buttons[0].get("disabled") is False
    assert any("PDF 会按当前已有内容生成" in caption for caption in fake_st.captions)


def test_requires_reanalysis_keeps_current_pdf_downloadable(monkeypatch):
    fake_st = _FakeStreamlit()
    report_viewer = _load_report_viewer(monkeypatch, fake_st)
    pdf_calls: list[tuple[Any, ...]] = []
    _install_report_viewer_fakes_without_missing(monkeypatch, report_viewer, pdf_calls)

    report_viewer.render_report(
        {"missing_data_requires_reanalysis": True},
        ticker="300476",
        trade_date="2026-06-04",
        signal="HOLD",
    )

    assert len(pdf_calls) == 1
    assert any(download["label"] == "📄 下载 PDF" for download in fake_st.downloads)
    assert any(
        button["label"] == "需重新分析" and button.get("disabled") is False
        for button in fake_st.buttons
    )
    assert any("当前 PDF 仍基于旧报告内容" in caption for caption in fake_st.captions)


def test_no_missing_data_shows_disabled_clean_status(monkeypatch):
    fake_st = _FakeStreamlit()
    report_viewer = _load_report_viewer(monkeypatch, fake_st)
    pdf_calls: list[tuple[Any, ...]] = []
    _install_report_viewer_fakes_without_missing(monkeypatch, report_viewer, pdf_calls)

    report_viewer.render_report(
        {"missing_data_complete": True, "missing_data_tasks": []},
        ticker="300476",
        trade_date="2026-06-04",
        signal="HOLD",
    )

    assert any(
        button["label"] == "✅ 无缺失项" and button.get("disabled") is True
        for button in fake_st.buttons
    )
    assert any(download["label"] == "📄 下载 PDF" for download in fake_st.downloads)


def test_retry_missing_button_is_above_missing_task_details(monkeypatch):
    fake_st = _FakeStreamlit()
    report_viewer = _load_report_viewer(monkeypatch, fake_st)
    pdf_calls: list[tuple[Any, ...]] = []
    _install_report_viewer_fakes(monkeypatch, report_viewer, pdf_calls)

    report_viewer._render_missing_data_panel(
        {"missing_data_complete": False},
        ticker="300476",
        trade_date="2026-06-04",
        key_suffix="missing-panel",
    )

    retry_index = fake_st.events.index(("button", "重新取数这些缺失项"))
    first_task_index = fake_st.events.index(("markdown", "**1. 基本面 · 综合基本面**"))

    assert retry_index < first_task_index


def test_reanalysis_starts_a_fresh_run(monkeypatch):
    fake_st = _FakeStreamlit()
    report_viewer = _load_report_viewer(monkeypatch, fake_st)
    key = "reanalyze_after_missing_300476_2026-06-04_panel"
    fake_st.button_returns[key] = True

    try:
        report_viewer._start_reanalysis_button(
            "300476", "2026-06-04", "panel"
        )
    except AssertionError as exc:
        assert str(exc) == "rerun should not be triggered in this test"

    assert fake_st.session_state["start_analysis"] == {
        "ticker": "300476",
        "trade_date": "2026-06-04",
        "fresh": True,
    }


@pytest.mark.parametrize("resolved_status", ["resolved", "consumed"])
def test_resolved_saved_gap_requires_reanalysis_in_both_exports(monkeypatch, resolved_status):
    fake_st = _FakeStreamlit()
    viewer = _load_report_viewer(monkeypatch, fake_st)
    state = {
        "missing_data_tasks": [{"id": "gap-1", "status": "active"}],
        "missing_data_complete": False,
        "data_quality_summary": "北向金额未核验，不得用于决策。",
    }
    original = deepcopy(state)
    monkeypatch.setattr(viewer, "get_missing_tasks", lambda *args, **kwargs: [
        {"id": "gap-1", "status": resolved_status},
    ])
    exports = []
    monkeypatch.setattr(viewer, "generate_markdown", lambda snapshot, *args: exports.append(snapshot) or "md")
    monkeypatch.setattr(viewer, "generate_pdf", lambda snapshot, *args: exports.append(snapshot) or b"pdf")

    viewer.render_report(state, "002044", "2026-09-21", "HOLD")

    assert state == original
    assert len(exports) == 2
    for snapshot in exports:
        assert snapshot["missing_data_tasks"] == []
        assert snapshot["missing_data_complete"] is True
        assert snapshot["missing_data_requires_reanalysis"] is True
        assert snapshot["data_quality_summary"] == original["data_quality_summary"]
    assert any(button["label"] == "需重新分析" for button in fake_st.buttons)
    assert original["data_quality_summary"] in fake_st.markdowns


def test_live_and_saved_unresolved_gaps_are_both_exported(monkeypatch):
    fake_st = _FakeStreamlit()
    viewer = _load_report_viewer(monkeypatch, fake_st)
    saved = {"id": "saved-gap", "status": "active"}
    live = {"id": "live-gap", "status": "active"}
    monkeypatch.setattr(viewer, "get_missing_tasks", lambda *args, **kwargs: [live])

    state = viewer._refresh_missing_data_state(
        {"missing_data_tasks": [saved]}, "002044", "2026-09-21",
    )

    assert state["missing_data_tasks"] == [saved, live]
    assert state["missing_data_complete"] is False
    assert state["missing_data_requires_reanalysis"] is False


def test_absent_live_index_does_not_erase_saved_gaps(monkeypatch):
    fake_st = _FakeStreamlit()
    viewer = _load_report_viewer(monkeypatch, fake_st)
    state = {"missing_data_tasks": [{"id": "saved-gap", "status": "active"}]}

    refreshed = viewer._refresh_missing_data_state(state, "002044", "2026-09-21")

    assert refreshed == state
    assert refreshed is not state


def test_legacy_report_is_not_labelled_complete(monkeypatch):
    fake_st = _FakeStreamlit()
    viewer = _load_report_viewer(monkeypatch, fake_st)

    viewer.render_report({}, "002044", "2026-09-21", "HOLD")

    assert any(button["label"] == "缺失状态待核验" for button in fake_st.buttons)
    assert not any(button["label"] == "✅ 无缺失项" for button in fake_st.buttons)
    assert any("不得视为审核通过" in text for text in fake_st.markdowns)


def test_successful_call_within_analysis_does_not_require_another_run(monkeypatch):
    viewer = _load_report_viewer(monkeypatch, _FakeStreamlit())
    monkeypatch.setattr(viewer, "get_missing_tasks", lambda *args, **kwargs: [
        {"id": "gap-1", "status": "resolved"},
    ])

    state = viewer._refresh_missing_data_state(
        {"missing_data_tasks": [], "missing_data_complete": True}, "002044", "2026-09-21",
    )

    assert state["missing_data_complete"] is True
    assert state["missing_data_requires_reanalysis"] is False


@pytest.mark.parametrize("state", [{}, {"missing_data_complete": False}])
def test_unrelated_resolved_record_does_not_certify_old_report(monkeypatch, state):
    fake_st = _FakeStreamlit()
    viewer = _load_report_viewer(monkeypatch, fake_st)
    monkeypatch.setattr(viewer, "get_missing_tasks", lambda *args, **kwargs: [
        {"id": "another-run", "status": "consumed"},
    ])

    viewer.render_report(state, "002044", "2026-09-21", "HOLD")

    assert any(button["label"] == "缺失状态待核验" for button in fake_st.buttons)
