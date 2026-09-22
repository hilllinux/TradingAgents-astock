from unittest.mock import Mock

import pandas as pd
import pytest

from tradingagents.dataflows import a_stock
from tradingagents.dataflows.market_metrics import summarize_prices


def quote_fields():
    fields = [""] * 74
    for index, value in {
        1: "美年健康", 3: "4.56", 30: "20260921161421", 38: "2.61", 39: "55.04",
        44: "176.66", 45: "178.49", 46: "2.24", 52: "-48.93", 53: "62.56",
        72: "3874227627", 73: "3914253923",
    }.items():
        fields[index] = value
    return fields


def read_quote(monkeypatch, fields):
    raw = ('v_sz002044="' + "~".join(fields) + '";').encode("gbk")
    monkeypatch.setattr(a_stock.urllib.request, "urlopen", lambda *args, **kwargs: Mock(read=lambda: raw))
    return a_stock._tencent_quote(["002044"])["002044"]


def test_quote_cap_and_pe_field_mapping(monkeypatch):
    quote = read_quote(monkeypatch, quote_fields())
    assert quote["mcap_yi"] == 178.49
    assert quote["float_mcap_yi"] == 176.66
    assert quote["pe_static"] == 62.56
    assert quote["pe_dynamic"] == -48.93
    assert quote["validation_issues"] == []


def test_quote_cap_inversion_fails_closed(monkeypatch):
    fields = quote_fields()
    fields[44], fields[45] = fields[45], fields[44]
    quote = read_quote(monkeypatch, fields)
    assert quote["mcap_yi"] is None and quote["float_mcap_yi"] is None
    assert quote["validation_issues"]


@pytest.mark.parametrize("bad", ["", "-", "nan", "inf"])
def test_missing_quote_pe_is_not_zero(monkeypatch, bad):
    fields = quote_fields()
    fields[53] = bad
    assert read_quote(monkeypatch, fields)["pe_static"] is None


def test_cap_share_price_validation(monkeypatch):
    fields = quote_fields()
    fields[45] = "190"
    quote = read_quote(monkeypatch, fields)
    assert quote["mcap_yi"] is None
    assert quote["float_mcap_yi"] == 176.66


def prices():
    dates = pd.bdate_range("2026-08-24", "2026-09-21")
    return pd.DataFrame({"Date": dates, "Close": [4.61] * 20 + [4.56], "Volume": [100] * 20 + [200]})


def test_twenty_day_window_uses_preceding_close_and_exposes_formulas():
    output = summarize_prices(prices(), "2026-09-21")
    assert "2026-08-25至2026-09-21" in output
    assert "2026-08-24收盘4.61" in output
    assert "-1.0846%" in output
    assert "2100.00/20=105.00股" in output
    assert "近5日均量/近20日均量（均含当日）=120.00/105.00=1.1429倍" in output
    assert "不含当日" in output and "200.00/100.00=2.0000倍" in output


def test_metrics_exclude_future_rows():
    frame = prices()
    frame.loc[len(frame)] = [pd.Timestamp("2099-01-01"), 999, 99999]
    assert "99999" not in summarize_prices(frame, "2026-09-21")


def test_short_windows_are_not_presented_as_twenty_days():
    output = summarize_prices(prices().tail(5), "2026-09-21")
    assert "实际仅5条" in output
    assert "共21条" in output
    assert "近20日日均量（含当日）" not in output


def test_zero_volume_does_not_divide_by_zero():
    frame = prices()
    frame["Volume"] = 0
    assert "均量为零" in summarize_prices(frame, "2026-09-21")


def test_duplicate_dates_and_invalid_numbers_fail_closed():
    frame = prices()
    assert "日期重复" in summarize_prices(pd.concat([frame, frame.tail(1)]), "2026-09-21")
    frame.loc[20, "Close"] = float("nan")
    assert "Close含无效数值" in summarize_prices(frame, "2026-09-21")


def test_fund_history_failure_preserves_realtime(monkeypatch):
    monkeypatch.setattr(a_stock, "_is_historical", lambda date: False)

    def response(url, **kwargs):
        if "push2his" in url:
            raise ConnectionError("history unavailable")
        return Mock(json=lambda: {"data": {"klines": ["2026-09-21 15:00,15950000,0,0,-12550000,28500000"]}})

    monkeypatch.setattr(a_stock, "_em_get", response)
    output = a_stock.get_fund_flow("002044", "2026-09-21")
    assert "1595万元" in output
    assert "数据缺失" in output
    assert "bullish" not in output


def test_realtime_failure_still_fetches_history(monkeypatch):
    monkeypatch.setattr(a_stock, "_is_historical", lambda date: False)

    def response(url, **kwargs):
        if "push2his" not in url:
            raise ConnectionError("minute unavailable")
        return Mock(json=lambda: {"data": {"klines": ["2026-09-21,15950000,0,0,-12550000,28500000"]}})

    monkeypatch.setattr(a_stock, "_em_get", response)
    output = a_stock.get_fund_flow("002044", "2026-09-21")
    assert "main=1595" in output
    assert "当日分钟资金流请求失败" in output
