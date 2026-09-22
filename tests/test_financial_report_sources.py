from datetime import datetime, timezone, timedelta
from unittest.mock import Mock

import pandas as pd
import pytest
import requests

from tradingagents.dataflows import a_stock, financial_reports as reports


@pytest.fixture(autouse=True)
def isolated_credentials(monkeypatch):
    monkeypatch.delenv("HITHINK_FINANCE_API_KEY", raising=False)
    monkeypatch.delenv("HITHINK_FINANCE_ENABLED", raising=False)


def sina_payload():
    return {"result": {"status": {"code": 0}, "data": {"report_list": {
        "20260630": {
            "rType": "合并期末", "rCurrency": "CNY", "publish_date": "20260831",
            "data": [
                {"item_title": "货币资金", "item_value": "2066923045.66", "item_display_type": 2},
                {"item_title": "资产总计", "item_value": "18705854965.03", "item_display_type": 2},
                {"item_title": "负债合计", "item_value": "9979646711.92", "item_display_type": 2},
                {"item_title": "商誉", "item_value": "5193306878.51", "item_display_type": 2},
                {"item_title": "零值", "item_value": "0", "item_display_type": 2},
                {"item_title": "缺失", "item_value": None, "item_display_type": 2},
            ],
        },
        "20251231": {
            "rType": "合并期末", "rCurrency": "CNY", "publish_date": "20260428",
            "data": [{"item_title": "货币资金", "item_value": "3611475562.79", "item_display_type": 2}],
        },
    }}}}


def epoch(date):
    return int(datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=reports.SHANGHAI).timestamp() * 1000)


def hithink_data():
    return {"item": [{
        "thscode": "002044.SZ", "fiscal_year": 2026, "fiscal_period": "Q2",
        "period_end_ms": epoch("2026-06-30"), "report_date_ms": epoch("2026-08-31"),
        "currency": "CNY", "assets_total": 18705854965.03, "total_debt": 9979646711.92,
        "cash": 2066923045.66, "holder_equity_total": 8726208253.11,
        "operating_income": 3844337529.39, "parent_holder_net_profit": -182388732.05,
    }]}


def test_sina_parses_report_list_preserves_details_zero_and_null():
    frame = reports.parse_sina(sina_payload(), "资产负债表", "quarterly", "2026-09-21")
    assert len(frame) == 2
    assert frame.iloc[0]["报告日"] == "2026-06-30"
    assert frame.iloc[0]["商誉"] == 5193306878.51
    assert frame.iloc[0]["零值"] == 0
    assert pd.isna(frame.iloc[0]["缺失"])


def test_publication_date_not_just_period_end_is_filtered():
    frame = reports.parse_sina(sina_payload(), "资产负债表", "quarterly", "2026-08-30")
    assert list(frame["报告日"]) == ["2025-12-31"]
    assert reports.parse_sina(sina_payload(), "资产负债表", "quarterly", "2026-04-27").empty


@pytest.mark.parametrize("field,value", [("rType", "母公司期末"), ("rCurrency", "USD"), ("publish_date", None)])
def test_unverified_metadata_is_not_used(field, value):
    payload = sina_payload()
    payload["result"]["data"]["report_list"]["20260630"][field] = value
    frame = reports.parse_sina(payload, "资产负债表", "quarterly", "2026-09-21")
    assert list(frame["报告日"]) == ["2025-12-31"]
    assert frame.attrs["rejected_rows"] == 1


def test_annual_frequency_and_schema_errors():
    frame = reports.parse_sina(sina_payload(), "资产负债表", "annual", "2026-09-21")
    assert list(frame["报告日"]) == ["2025-12-31"]
    with pytest.raises(ValueError):
        reports.parse_sina({"result": {"data": {"fzb": []}}}, "资产负债表", "annual", "2026-09-21")


def test_hithink_mappings_and_publication_filter(monkeypatch):
    monkeypatch.setattr(reports, "_hithink_get", lambda *args: hithink_data())
    frame = reports.hithink_statements("002044", "sz", "资产负债表", "quarterly", "2026-09-21")
    assert frame.iloc[0]["负债合计"] == 9979646711.92
    assert "有息负债" not in frame.columns
    assert "商誉" not in frame.columns
    assert reports.hithink_statements("002044", "sz", "资产负债表", "quarterly", "2026-08-30").empty


def test_q2_income_is_explicitly_cumulative(monkeypatch):
    monkeypatch.setattr(reports, "_hithink_get", lambda *args: hithink_data())
    frame = reports.hithink_statements("002044", "sz", "利润表", "quarterly", "2026-09-21")
    assert frame.iloc[0]["营业收入"] == 3844337529.39
    assert "累计" in frame.iloc[0]["报表口径"]
    assert pd.isna(frame.iloc[0]["利息费用"])


def test_wrong_security_is_rejected(monkeypatch):
    data = hithink_data()
    data["item"][0]["thscode"] = "600519.SH"
    monkeypatch.setattr(reports, "_hithink_get", lambda *args: data)
    with pytest.raises(ValueError, match="different security"):
        reports.hithink_statements("002044", "sz", "资产负债表", "quarterly", "2026-09-21")


def test_reconciliation_preserves_conflicting_sources(monkeypatch):
    primary = reports.parse_sina(sina_payload(), "资产负债表", "quarterly", "2026-09-21")
    monkeypatch.setattr(reports, "_hithink_get", lambda *args: hithink_data())
    secondary = reports.hithink_statements("002044", "sz", "资产负债表", "quarterly", "2026-09-21")
    assert "数据冲突" not in reports.reconcile(primary, secondary)
    secondary.loc[0, "货币资金"] = 100
    before = primary.copy(deep=True)
    output = reports.reconcile(primary, secondary)
    assert "数据冲突" in output and "货币资金" in output and "HiThink=100" in output
    pd.testing.assert_frame_equal(before, primary)


def test_http_auth_never_enters_url_or_error(monkeypatch):
    monkeypatch.setenv("HITHINK_FINANCE_API_KEY", "test-secret-not-for-logging")
    response = Mock(status_code=401)
    request = Mock(return_value=response)
    monkeypatch.setattr(reports.requests, "get", request)
    with pytest.raises(RuntimeError) as exc:
        reports._hithink_get("balance-sheets", {"thscode": "002044.SZ"})
    assert "test-secret" not in str(exc.value)
    assert "test-secret" not in request.call_args.args[0]
    assert request.call_args.kwargs["allow_redirects"] is False
    assert request.call_count == 1


def test_bounded_network_retry_redacts_exception(monkeypatch):
    monkeypatch.setenv("HITHINK_FINANCE_API_KEY", "secret")
    request = Mock(side_effect=requests.ConnectionError("secret"))
    monkeypatch.setattr(reports.requests, "get", request)
    monkeypatch.setattr(reports.time, "sleep", lambda *args: None)
    with pytest.raises(RuntimeError) as exc:
        reports._hithink_get("balance-sheets", {})
    assert "secret" not in str(exc.value)
    assert request.call_count == 2


def test_disabled_source_makes_no_network_request(monkeypatch):
    request = Mock()
    monkeypatch.setattr(reports.requests, "get", request)
    with pytest.raises(RuntimeError):
        reports._hithink_get("balance-sheets", {})
    request.assert_not_called()


def test_hithink_fallback_when_sina_fails(monkeypatch):
    monkeypatch.setenv("HITHINK_FINANCE_API_KEY", "test")
    monkeypatch.setattr(a_stock, "_get_financial_report_sina", Mock(side_effect=ValueError("bad schema")))
    monkeypatch.setattr(reports, "_hithink_get", lambda *args: hithink_data())
    output = a_stock.get_balance_sheet("002044", curr_date="2026-09-21")
    assert "新浪读取失败" in output
    assert "18705854965.03" in output
    assert "交叉核验未完成" in output
    assert "53.3504" in output


def test_historical_analysis_does_not_fetch_unversioned_indicators(monkeypatch):
    monkeypatch.setenv("HITHINK_FINANCE_API_KEY", "test")
    monkeypatch.setattr(a_stock, "_get_financial_report_sina", lambda *args: pd.DataFrame())
    monkeypatch.setattr(reports, "_hithink_get", lambda *args: hithink_data())
    indicators = Mock()
    monkeypatch.setattr(reports, "hithink_indicators", indicators)
    past = (datetime.now(timezone.utc) - timedelta(days=2)).date().isoformat()
    a_stock.get_balance_sheet("002044", curr_date=past)
    indicators.assert_not_called()


def test_indicator_units_nulls_and_negative_cash_warning(monkeypatch):
    monkeypatch.setattr(reports, "_hithink_get", lambda *args: {"abilities": [{"indicators": [
        {"index_id": "index_weighted_avg_roe", "value": "-2.2400"},
        {"index_id": "assets_debt_ratio", "value": "53.3504"},
        {"index_id": "earned_interest_multiple", "value": None},
        {"index_id": "net_profit_cash_content", "value": "235.112768"},
    ]}]})
    output = reports.hithink_indicators("002044", "sz", {"报告日": "2026-06-30", "披露日": "2026-08-31"})
    assert "-2.24%" in output and "53.3504%" in output
    assert "利息保障倍数: [数据缺失" in output
    assert "235.112768" not in output
    assert "负负相除" in output


def test_conflicting_balance_data_does_not_generate_derived_ratio(monkeypatch):
    monkeypatch.setenv("HITHINK_FINANCE_API_KEY", "test")
    primary = reports.parse_sina(sina_payload(), "资产负债表", "quarterly", "2026-09-21")
    data = hithink_data()
    data["item"][0]["total_debt"] = 100
    monkeypatch.setattr(a_stock, "_get_financial_report_sina", lambda *args: primary)
    monkeypatch.setattr(reports, "_hithink_get", lambda *args: data)
    output = a_stock.get_balance_sheet("002044", curr_date="2026-09-21")
    assert "[数据冲突]" in output
    assert "不生成资产负债率" in output
    assert "[计算值/" not in output
