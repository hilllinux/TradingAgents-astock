import json
from types import SimpleNamespace

from tradingagents.graph.trading_graph import TradingAgentsGraph
from web.pdf_export import _collect_sections, _missing_data_warning, generate_markdown


def test_audit_is_exported_before_reports_without_changing_them():
    state = {"data_quality_summary": "未解决：市值冲突", "market_report": "原始观点"}
    sections = _collect_sections(state, "002044")
    assert sections[0][0] == "数据质量审核与限制"
    assert "市值冲突" in sections[0][1]
    assert "不得用于决策" in sections[0][1]
    assert sections[1] == ("技术分析报告", "原始观点")
    markdown = generate_markdown(state, "002044", "2026-09-21", "HOLD")
    assert markdown.index("市值冲突") < markdown.index("原始观点")


def test_legacy_history_does_not_imply_audit_passed():
    sections = _collect_sections({"market_report": "旧报告"})
    assert "未保存或未提供" in sections[0][1]
    assert "不得视为审核通过" in sections[0][1]


def test_complete_run_persists_quality_summary(tmp_path):
    graph = SimpleNamespace(log_states_dict={}, ticker="002044", config={"results_dir": str(tmp_path)})
    state = {name: "report" for name in (
        "company_of_interest", "trade_date", "market_report", "sentiment_report", "news_report",
        "fundamentals_report", "trader_investment_plan", "investment_plan", "final_trade_decision",
    )}
    state["investment_debate_state"] = {name: "" for name in (
        "bull_history", "bear_history", "history", "current_response", "judge_decision",
    )}
    state["risk_debate_state"] = {name: "" for name in (
        "aggressive_history", "conservative_history", "neutral_history", "history", "judge_decision",
    )}
    state["data_quality_summary"] = "分段审核2/3，需人工复核"
    state["missing_data_tasks"] = [{"id": "gap", "status": "active"}]
    state["missing_data_complete"] = False
    state["missing_data_requires_reanalysis"] = True
    state["missing_data_updated_at"] = 123
    TradingAgentsGraph._log_state(graph, "2026-09-21", state)
    path = tmp_path / "002044/TradingAgentsStrategy_logs/full_states_log_2026-09-21.json"
    saved = json.loads(path.read_text())
    for field in (
        "data_quality_summary", "missing_data_tasks", "missing_data_complete",
        "missing_data_requires_reanalysis", "missing_data_updated_at",
    ):
        assert saved[field] == state[field]


def test_export_keeps_both_missing_and_reanalysis_warnings():
    state = {
        "missing_data_tasks": [{"status": "active"}],
        "missing_data_requires_reanalysis": True,
        "data_quality_summary": "质量审核未通过",
        "market_report": "旧分析内容",
    }
    warning = _missing_data_warning(state)
    assert "仍有 1 个取数缺口" in warning
    assert "尚未重新分析" in warning
    markdown = generate_markdown(state, "002044", "2026-09-21", "HOLD")
    assert markdown.index(warning) < markdown.index("质量审核未通过") < markdown.index("旧分析内容")


def test_incomplete_flag_without_details_is_not_silently_ignored():
    assert "缺失状态尚未核验" in _missing_data_warning({"missing_data_complete": False})
