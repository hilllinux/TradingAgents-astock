"""Verify that evidence rules reach each role without changing the workflow."""

from copy import deepcopy
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from tradingagents.agents.analysts.fundamentals_analyst import create_fundamentals_analyst
from tradingagents.agents.analysts.hot_money_tracker import create_hot_money_tracker
from tradingagents.agents.analysts.lockup_watcher import create_lockup_watcher
from tradingagents.agents.analysts.market_analyst import create_market_analyst
from tradingagents.agents.analysts.news_analyst import create_news_analyst
from tradingagents.agents.analysts.policy_analyst import create_policy_analyst
from tradingagents.agents.analysts.social_media_analyst import create_social_media_analyst
from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.managers.research_manager import create_research_manager
from tradingagents.agents.quality_gate import REPORT_FIELDS, _build_review_prompt
from tradingagents.agents.researchers.bear_researcher import create_bear_researcher
from tradingagents.agents.researchers.bull_researcher import create_bull_researcher
from tradingagents.agents.risk_mgmt.aggressive_debator import create_aggressive_debator
from tradingagents.agents.risk_mgmt.conservative_debator import create_conservative_debator
from tradingagents.agents.risk_mgmt.neutral_debator import create_neutral_debator
from tradingagents.agents.schemas import PortfolioDecision, ResearchPlan, TraderProposal
from tradingagents.agents.trader.trader import create_trader
from tradingagents.agents.utils.research_prompts import (
    ANALYST_REPORT_RULES,
    DEBATE_RULES,
    DECISION_RULES,
    EVIDENCE_RULES,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def research_state():
    return {
        "company_of_interest": "002044",
        "trade_date": "2026-09-21",
        "messages": [HumanMessage(content="请分析该标的。")],
        **{field: f"{field}: 原始报告片段" for field in REPORT_FIELDS.values()},
        "data_quality_summary": "关键资金日期待核验，不能据此判断个股流入。",
        "investment_plan": "Hold: 关键证据不足。",
        "trader_investment_plan": "Hold: 等待核验。",
        "investment_debate_state": {
            "history": "总市值176.66亿元；流通市值178.49亿元。",
            "bull_history": "",
            "bear_history": "",
            "current_response": "退出前十大股东，但未披露当前持股数量。",
            "count": 1,
        },
        "risk_debate_state": {
            "history": "全市场资金流入，个股流入缺失。",
            "aggressive_history": "",
            "conservative_history": "",
            "neutral_history": "",
            "current_aggressive_response": "",
            "current_conservative_response": "",
            "current_neutral_response": "",
            "count": 1,
        },
    }


@pytest.mark.parametrize(
    "factory,report_field,role_rule",
    [
        (create_market_analyst, "market_report", "底背离需给出"),
        (create_social_media_analyst, "sentiment_report", "去重后的样本"),
        (create_news_analyst, "news_report", "事件发生日、发布日期及生效日"),
        (create_fundamentals_analyst, "fundamentals_report", "不能拿涨跌幅排名推导行业估值中枢"),
        (create_policy_analyst, "policy_report", "区分草案、正式发布与已生效政策"),
        (create_hot_money_tracker, "hot_money_report", "请求日期不是数据日期"),
        (create_lockup_watcher, "lockup_report", "缺少条件不估写"),
    ],
)
@pytest.mark.parametrize("tool_call", [False, True])
def test_analysts_receive_evidence_rules(factory, report_field, role_rule, tool_call, research_state):
    captured = []
    response = AIMessage(
        content="等待核验" if not tool_call else "",
        tool_calls=[{"name": "get_news", "args": {}, "id": "call-test"}] if tool_call else [],
    )

    def capture(prompt):
        captured.append(prompt.to_messages()[0].content)
        return response

    llm = MagicMock()
    llm.bind_tools.return_value = RunnableLambda(capture)
    result = factory(llm)(research_state)

    assert ANALYST_REPORT_RULES in captured[0]
    assert role_rule in captured[0]
    assert "2026-09-21" in captured[0]
    assert "002044" in captured[0]
    assert result[report_field] == ("" if tool_call else "等待核验")
    assert result["messages"] == [response]
    llm.bind_tools.assert_called_once()


@pytest.mark.parametrize(
    "factory,state_field",
    [
        (create_bull_researcher, "investment_debate_state"),
        (create_bear_researcher, "investment_debate_state"),
        (create_aggressive_debator, "risk_debate_state"),
        (create_conservative_debator, "risk_debate_state"),
        (create_neutral_debator, "risk_debate_state"),
    ],
)
def test_debate_roles_preserve_evidence_and_counterarguments(factory, state_field, research_state):
    original_state = deepcopy(research_state)
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="证据不足，暂不支持该方假设。")

    result = factory(llm)(research_state)
    prompt = llm.invoke.call_args.args[0]

    assert DEBATE_RULES in prompt
    assert research_state["data_quality_summary"] in prompt
    assert research_state[state_field]["history"] in prompt
    for field in REPORT_FIELDS.values():
        assert research_state[field] in prompt
    for removed_claim in ("30x anchor", "30x A-stock growth anchor", "30x PE digestion", "50-100x", "80% retail"):
        assert removed_claim not in prompt
    assert result[state_field]["count"] == original_state[state_field]["count"] + 1
    assert result[state_field]["history"].startswith(original_state[state_field]["history"])
    assert research_state == original_state


@pytest.mark.parametrize(
    "factory,schema,decision,output_field,heading",
    [
        (
            create_research_manager,
            ResearchPlan,
            ResearchPlan(recommendation="Hold", rationale="低置信度，证据不足。", strategic_actions="核验后复评。"),
            "investment_plan",
            "**Recommendation**: Hold",
        ),
        (
            create_trader,
            TraderProposal,
            TraderProposal(action="Hold", reasoning="低置信度，证据不足，待核验。"),
            "trader_investment_plan",
            "FINAL TRANSACTION PROPOSAL: **HOLD**",
        ),
        (
            create_portfolio_manager,
            PortfolioDecision,
            PortfolioDecision(rating="Hold", executive_summary="低置信度。", investment_thesis="证据不足，核验后复评。"),
            "final_trade_decision",
            "**Rating**: Hold",
        ),
    ],
)
@pytest.mark.parametrize("mode", ["structured", "unsupported", "invoke_failure"])
def test_decisions_keep_rules_and_output_contract(factory, schema, decision, output_field, heading, mode, research_state):
    original_state = deepcopy(research_state)
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="Hold: 低置信度，待核验。")
    if mode == "unsupported":
        llm.with_structured_output.side_effect = NotImplementedError
    elif mode == "invoke_failure":
        llm.with_structured_output.return_value.invoke.side_effect = ValueError("invalid response")
    else:
        llm.with_structured_output.return_value.invoke.return_value = decision

    result = factory(llm)(research_state)
    llm.with_structured_output.assert_called_once_with(schema)
    invoke = llm.with_structured_output.return_value.invoke if mode == "structured" else llm.invoke
    prompt = invoke.call_args.args[0]
    prompt_text = "\n".join(message["content"] for message in prompt) if isinstance(prompt, list) else prompt

    assert DECISION_RULES in prompt_text
    assert research_state["data_quality_summary"] in prompt_text
    for field in REPORT_FIELDS.values():
        assert research_state[field] in prompt_text
    assert "不要增加新的结构化字段" in prompt_text
    assert "证据不足" in prompt_text
    assert "置信度" in prompt_text
    if mode == "structured":
        assert heading in result[output_field]
        llm.invoke.assert_not_called()
    else:
        assert result[output_field] == "Hold: 低置信度，待核验。"
    if mode == "invoke_failure":
        assert llm.with_structured_output.return_value.invoke.call_args.args[0] == prompt
    assert research_state == original_state


def test_quality_review_preserves_report_tail_and_checks_conflicts():
    reports = {
        "market_report": "成交量5280万手，另一个报告为5280万股。",
        "fundamentals_report": "总市值176.66亿元，流通市值178.49亿元。" + "长报告内容" * 800 + "不可见尾部",
    }
    original_reports = deepcopy(reports)
    prompt = _build_review_prompt(reports, "2026-09-21", "002044")

    assert EVIDENCE_RULES in prompt
    assert "跨报告冲突" in prompt
    assert "确定错误、证据不足和待外部核验" in prompt
    assert "仅审核可见片段" in prompt
    assert "truncated for review" not in prompt
    assert "不可见尾部" in prompt
    assert "176.66" in prompt and "178.49" in prompt
    assert reports == original_reports


@pytest.mark.parametrize(
    "required_rule",
    [
        "不是行为指令",
        "[机构预测]",
        "采集日期不等于数据日期",
        "1手=100股",
        "退出前十大股东=清仓",
        "全市场流入=个股流入",
        "减持预案=已减持",
        "数据缺失=可凭经验补数",
        "不设统一合理PE或PEG阈值",
        "重复引用同一来源不增加证据强度",
    ],
)
def test_shared_rules_cover_report_audit_failures(required_rule):
    assert required_rule in EVIDENCE_RULES


@pytest.mark.parametrize("factory,state_field", [
    (create_bull_researcher, "investment_debate_state"),
    (create_bear_researcher, "investment_debate_state"),
    (create_aggressive_debator, "risk_debate_state"),
    (create_conservative_debator, "risk_debate_state"),
    (create_neutral_debator, "risk_debate_state"),
])
def test_compaction_preserves_original_evidence_and_audit(factory, state_field, research_state):
    history = "\n" + "\n".join(
        f"Bull Analyst: 第{turn}轮假设。待核验的第二句。" for turn in range(7)
    )
    research_state[state_field]["history"] = history
    original = deepcopy(research_state)
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="保留限制，暂不作方向判断。")

    result = factory(llm)(research_state)
    prompt = llm.invoke.call_args.args[0]

    assert "[早期论点摘要]" in prompt
    assert "第0轮假设。待核验的第二句。" not in prompt
    assert prompt.count("第6轮假设。待核验的第二句。") == 1
    assert DEBATE_RULES in prompt
    assert research_state["data_quality_summary"] in prompt
    for field in REPORT_FIELDS.values():
        assert research_state[field] in prompt
    assert result[state_field]["history"].startswith(history)
    assert research_state == original


def test_structured_fields_unchanged_and_descriptions_require_evidence():
    assert set(ResearchPlan.model_fields) == {"recommendation", "rationale", "strategic_actions"}
    assert set(TraderProposal.model_fields) == {"action", "reasoning"}
    assert set(PortfolioDecision.model_fields) == {"rating", "executive_summary", "investment_thesis", "time_horizon"}
    assert "insufficient" in ResearchPlan.model_fields["recommendation"].description
    assert "confidence" in ResearchPlan.model_fields["rationale"].description
    assert "confidence" in TraderProposal.model_fields["reasoning"].description
    assert "confidence" in PortfolioDecision.model_fields["executive_summary"].description
    assert "conflicts" in PortfolioDecision.model_fields["investment_thesis"].description
