"""Never promote undocumented or misdated northbound values to trading signals."""

from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
import requests

from tradingagents.dataflows import a_stock
from tradingagents.dataflows import config as data_config

pytestmark = pytest.mark.unit


@pytest.fixture
def source(monkeypatch, tmp_path):
    response = MagicMock()
    response.json.return_value = {
        "time": ["14:59", "15:00"],
        "hgt": [-9.0, -9.28],
        "sgt": [378.0, 379.75],
    }
    request = MagicMock(return_value=response)
    monkeypatch.setattr(requests, "get", request)
    monkeypatch.setattr(a_stock, "_northbound_today", lambda: date(2026, 9, 21))
    monkeypatch.setattr(data_config, "get_config", lambda: {"data_cache_dir": str(tmp_path / "cache")})
    return request, response, tmp_path / "cache" / "northbound_daily.csv"


def assert_no_amounts_or_signal(output):
    assert "[数据缺失:" in output
    assert "不可用于方向判断" in output
    for unsupported_value in ("379.75", "370.47", "-9.28", "INFLOW", "OUTFLOW", "Close:", "avg net flow"):
        assert unsupported_value not in output


def test_undated_original_payload_does_not_emit_report_amounts_or_write_cache(source):
    request, response, cache = source
    output = a_stock.get_northbound_flow("2026-09-21", include_history=True)
    assert_no_amounts_or_signal(output)
    assert "缺少上游数据日期" in output
    assert "单位缺失或无法识别" in output
    assert "业务定义和统计范围未核验" in output
    assert not cache.parent.exists()
    request.assert_called_once()
    response.raise_for_status.assert_called_once()


@pytest.mark.parametrize("label", ["2026-09-21", "20260921"])
def test_matching_date_and_self_reported_unit_do_not_verify_field_semantics(source, label):
    _, response, cache = source
    response.json.return_value.update(date=label, unit="亿元", metric="net_buy", verified=True)
    output = a_stock.get_northbound_flow("2026-09-21")
    assert_no_amounts_or_signal(output)
    assert "上游日期标签与分析日相符" in output
    assert "单位标签" in output
    assert "业务定义和统计范围未核验" in output
    assert not cache.exists()


@pytest.mark.parametrize("label", ["2026-09-16", "2026-09-22"])
def test_mismatched_data_date_is_rejected(source, label):
    _, response, _ = source
    response.json.return_value.update(trade_date=label)
    output = a_stock.get_northbound_flow("2026-09-21")
    assert_no_amounts_or_signal(output)
    assert f"上游日期标签 {label} 与分析日不一致" in output


@pytest.mark.parametrize("label", [None, 20260921, "15:00", "2026-02-30", "unknown"])
def test_invalid_payload_date_is_not_replaced_by_request_date(source, label):
    _, response, _ = source
    response.json.return_value.update(date=label)
    output = a_stock.get_northbound_flow("2026-09-21")
    assert_no_amounts_or_signal(output)
    assert "格式无法识别" in output


@pytest.mark.parametrize(
    "payload,problem",
    [
        (None, "接口返回不是对象"),
        ([], "接口返回不是对象"),
        ({}, "通道缺失"),
        ({"time": ["15:00"], "hgt": [1]}, "通道缺失"),
        ({"time": ["15:00"], "hgt": [], "sgt": [1]}, "通道缺失"),
        ({"time": ["15:00"], "hgt": [1, 2], "sgt": [1]}, "数组长度不一致"),
        ({"time": ["15:00"], "hgt": [float("nan")], "sgt": [1]}, "非有限数值"),
        ({"time": ["15:00"], "hgt": [1], "sgt": [float("inf")]}, "非有限数值"),
        ({"time": ["15:00"], "hgt": [1], "sgt": [None]}, "非数值"),
        ({"time": ["15:00"], "hgt": [True], "sgt": [1]}, "非数值"),
    ],
)
def test_malformed_or_partial_payload_fails_closed(source, payload, problem):
    _, response, _ = source
    response.json.return_value = payload
    output = a_stock.get_northbound_flow("2026-09-21")
    assert_no_amounts_or_signal(output)
    assert problem in output


@pytest.mark.parametrize("analysis_date", ["2026-09-16", "2026-09-22"])
def test_past_and_future_dates_do_not_query_current_feed(source, analysis_date):
    request, _, _ = source
    output = a_stock.get_northbound_flow(analysis_date, include_history=True)
    request.assert_not_called()
    assert_no_amounts_or_signal(output)
    assert "历史分析禁止调用" in output or "分析日尚未到达" in output


@pytest.mark.parametrize("analysis_date", [None, "", "not-a-date", "2026-9-21", "2026-02-30"])
def test_invalid_analysis_date_does_not_query_feed(source, analysis_date):
    request, _, _ = source
    output = a_stock.get_northbound_flow(analysis_date)
    request.assert_not_called()
    assert "分析日期无效" in output


@pytest.mark.parametrize("analysis_date", ["2026-09-16", "2026-09-21"])
def test_legacy_cache_is_preserved_but_never_used(source, analysis_date):
    _, _, cache = source
    cache.parent.mkdir()
    original = (
        "date,hgt,sgt\n2026-09-16,-9.28,379.75\n"
        "2026-09-17,-9.28,379.75\n2026-09-21,-9.28,379.75\n"
        "2099-01-01,99999,99999\n"
    ).encode()
    cache.write_bytes(original)
    output = a_stock.get_northbound_flow(analysis_date, include_history=True)
    assert_no_amounts_or_signal(output)
    assert "历史缓存已隔离" in output
    assert "2099-01-01" not in output
    assert cache.read_bytes() == original


@pytest.mark.parametrize("failure", [requests.Timeout, requests.HTTPError, ValueError])
def test_network_http_and_json_failures_do_not_fallback_to_cache(source, failure):
    request, response, cache = source
    cache.parent.mkdir()
    cache.write_text("date,hgt,sgt\n2026-09-21,-9.28,379.75\n")
    original = cache.read_bytes()
    if failure is requests.Timeout:
        request.side_effect = failure("network failure")
    elif failure is requests.HTTPError:
        response.raise_for_status.side_effect = failure("http failure")
    else:
        response.json.side_effect = failure("invalid json")
    output = a_stock.get_northbound_flow("2026-09-21", include_history=True)
    assert_no_amounts_or_signal(output)
    assert "接口核验失败" in output
    assert "不回退到未核验缓存" in output
    assert cache.read_bytes() == original


def test_northbound_today_uses_shanghai_date(monkeypatch):
    instant = datetime(2026, 9, 20, 17, 0, tzinfo=timezone.utc)
    clock = MagicMock()
    clock.now.side_effect = lambda zone: instant.astimezone(zone)
    monkeypatch.setattr(a_stock, "datetime", clock)
    assert a_stock._northbound_today() == date(2026, 9, 21)
    clock.now.assert_called_once_with(timezone(timedelta(hours=8)))
