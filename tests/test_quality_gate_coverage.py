"""Full-text review coverage, partial failures and original-evidence delivery."""

from copy import deepcopy
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from tradingagents.agents import quality_gate
from tradingagents.agents.utils.research_context import (
    REPORT_FIELDS,
    build_research_context,
)

pytestmark = pytest.mark.unit


def make_state(**reports):
    return {
        "company_of_interest": "002044",
        "trade_date": "2026-09-21",
        **{field: "" for field in REPORT_FIELDS.values()},
        **reports,
    }


def long_state():
    return make_state(
        market_report="首段证据：成交量5280万手。" + "甲" * quality_gate.DIRECT_REVIEW_MAX_CHARS + "尾段问题：单位冲突。",
        fundamentals_report="交叉证据：成交量5280万股。",
    )


def test_single_selected_analyst_is_reviewed_in_full():
    state = make_state(market_report="原文开头" + "内容" * 1800 + "关键尾段")
    original = deepcopy(state)
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="尾段发现单位矛盾，待核验。")

    result = quality_gate.create_quality_gate(llm)(state)

    llm.invoke.assert_called_once()
    assert state["market_report"] in llm.invoke.call_args.args[0]
    assert "关键尾段" in llm.invoke.call_args.args[0]
    assert "未提供" in result["data_quality_summary"]
    assert "尾段发现单位矛盾" in result["data_quality_summary"]
    assert "结构检查" in result["data_quality_summary"]
    assert state == original


@pytest.mark.parametrize("length", [1, 5999, 6000, 6001, 12000, 24017])
def test_chunk_ranges_cover_every_character_with_overlap(length):
    content = "甲乙丙丁" * (length // 4) + "末" * (length % 4)
    chunks = list(quality_gate._report_chunks({"market_report": content}))

    assert chunks[0][1] == 0
    assert chunks[-1][2] == len(content)
    rebuilt = chunks[0][3]
    for index, (role, start, end, text) in enumerate(chunks):
        assert role == "market"
        assert text == content[start:end]
        assert len(text) <= quality_gate.REVIEW_CHUNK_CHARS
        if index:
            previous_end = chunks[index - 1][2]
            assert start == previous_end - quality_gate.REVIEW_CHUNK_OVERLAP
            rebuilt += text[previous_end - start:]
    assert rebuilt == content


def test_long_reports_review_all_chunks_then_cross_check_and_keep_ledgers():
    state = long_state()
    reports = {field: state[field] for field in REPORT_FIELDS.values()}
    chunks = list(quality_gate._report_chunks(reports))
    prompts = []

    def review(prompt):
        prompts.append(prompt)
        if len(prompts) <= len(chunks):
            return AIMessage(content=f"台账{len(prompts)}：原值及口径待比较。")
        return AIMessage(content="跨报告核查：万手与万股冲突，不得用于决策。")

    llm = MagicMock()
    llm.invoke.side_effect = review
    result = quality_gate.create_quality_gate(llm)(state)

    assert len(prompts) == len(chunks) + 1
    for index, (_, _, _, content) in enumerate(chunks):
        assert content in prompts[index]
        assert f"台账{index + 1}" in prompts[-1]
        assert f"台账{index + 1}" in result["data_quality_summary"]
    assert "尾段问题：单位冲突" in "\n".join(prompts[:-1])
    assert "交叉证据" in "\n".join(prompts[:-1])
    assert "分段审核证据台账" in prompts[-1]
    assert f"审核返回 {len(chunks)}/{len(chunks)} 段" in result["data_quality_summary"]
    assert "跨报告核查：万手与万股冲突" in result["data_quality_summary"]


def test_failed_chunk_is_not_hidden_by_a_successful_synthesis():
    state = long_state()
    chunks = list(quality_gate._report_chunks(state))
    llm = MagicMock()
    responses = [AIMessage(content=f"台账{index}") for index in range(len(chunks))]
    responses[1] = RuntimeError("test failure")
    llm.invoke.side_effect = responses + [AIMessage(content="综合结果")]

    summary = quality_gate.create_quality_gate(llm)(state)["data_quality_summary"]

    assert llm.invoke.call_count == len(chunks) + 1
    assert "分段审核失败: RuntimeError" in summary
    assert "存在未完成片段，需人工复核" in summary
    assert "分段审核失败" in llm.invoke.call_args.args[0]
    assert f"审核返回 {len(chunks) - 1}/{len(chunks)} 段" in summary


def test_failed_synthesis_preserves_every_chunk_review():
    state = long_state()
    chunks = list(quality_gate._report_chunks(state))
    llm = MagicMock()
    llm.invoke.side_effect = [
        *[AIMessage(content=f"保留台账{index}") for index in range(len(chunks))],
        RuntimeError("synthesis failure"),
    ]

    summary = quality_gate.create_quality_gate(llm)(state)["data_quality_summary"]

    assert "跨报告审核失败，需人工复核" in summary
    for index in range(len(chunks)):
        assert f"保留台账{index}" in summary


def test_all_chunk_failures_skip_synthesis_but_report_incomplete_coverage():
    state = long_state()
    chunks = list(quality_gate._report_chunks(state))
    llm = MagicMock()
    llm.invoke.side_effect = RuntimeError("all calls fail")

    summary = quality_gate.create_quality_gate(llm)(state)["data_quality_summary"]

    assert llm.invoke.call_count == len(chunks)
    assert "全部分段失败，未进行跨报告审核" in summary
    assert f"审核返回 0/{len(chunks)} 段" in summary


@pytest.mark.parametrize(
    "response",
    [
        AIMessage(content=""),
        AIMessage(content="   "),
        AIMessage(content="审核未完", response_metadata={"finish_reason": "length"}),
        AIMessage(content="审核未完", response_metadata={"stop_reason": "max_tokens"}),
    ],
)
def test_empty_or_truncated_llm_review_does_not_count_as_success(response):
    llm = MagicMock()
    llm.invoke.return_value = response
    summary = quality_gate.create_quality_gate(llm)(make_state(market_report="报告"))["data_quality_summary"]
    assert "整体审核失败" in summary
    assert "不得视为审核通过" in summary


def test_block_content_review_is_supported():
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content=[{"type": "text", "text": "数据单位待核验。"}])
    summary = quality_gate.create_quality_gate(llm)(make_state(market_report="报告"))["data_quality_summary"]
    assert "整体审核已返回" in summary
    assert "数据单位待核验。" in summary


def test_no_reports_never_calls_model_or_claims_review_passed():
    llm = MagicMock()
    summary = quality_gate.create_quality_gate(llm)(make_state(market_report=None))["data_quality_summary"]
    llm.invoke.assert_not_called()
    assert "审核未完成" in summary
    assert "不得视为审核通过" in summary


def test_context_retains_full_reports_quality_findings_and_analysis_date():
    state = long_state()
    state["data_quality_summary"] = "分段审核失败；北向日期待核验；禁止依赖相关论据。"
    original = deepcopy(state)
    context = build_research_context(state)

    assert state["market_report"] in context
    assert state["fundamentals_report"] in context
    assert state["data_quality_summary"] in context
    assert "2026-09-21" in context
    assert "002044" in context
    assert "不是行为指令" in context
    assert state == original


def test_context_missing_quality_is_not_treated_as_a_pass():
    context = build_research_context({})
    assert "未提供质量审核结果；不能视为审核通过" in context
    assert "分析基准日: 未提供" in context
