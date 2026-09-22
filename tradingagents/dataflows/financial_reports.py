"""Dated financial statements, optional HiThink enrichment and reconciliation."""

from datetime import datetime, timezone, timedelta
import math
import os
import time

import pandas as pd
import requests


SHANGHAI = timezone(timedelta(hours=8))
BASE_URL = "https://fuyao.aicubes.cn"
REPORT_FIELDS = {
    "资产负债表": ("balance-sheets", {
        "assets_total": "资产总计",
        "total_current_assets": "流动资产合计",
        "non_current_nets_total": "非流动资产合计",
        "cash": "货币资金",
        "accounts_receivable": "应收账款",
        "total_debt": "负债合计",
        "holder_equity_total": "所有者权益合计",
    }),
    "利润表": ("income-statements", {
        "operating_income": "营业收入",
        "operating_costs": "营业成本",
        "operating_expenses": "营业总成本",
        "sales_fee": "销售费用",
        "manage_fee": "管理费用",
        "research_and_development_expenses": "研发费用",
        "operating_profit": "营业利润",
        "interest_expenses": "利息费用",
        "profit_total": "利润总额",
        "income_tax_expense": "所得税费用",
        "net_profit": "净利润",
        "parent_holder_net_profit": "归属于母公司所有者的净利润",
        "basic_eps": "基本每股收益",
    }),
    "现金流量表": ("cash-flow-statements", {
        "act_cash_flow_net": "经营活动产生的现金流量净额",
        "invest_cash_flow_net": "投资活动产生的现金流量净额",
        "financing_cash_flow_net": "筹资活动产生的现金流量净额",
        "pay_fixed_assets_etc_cash": "购建固定资产、无形资产和其他长期资产支付的现金",
        "pay_dividends_profits_interest_cash": "分配股利、利润或偿付利息支付的现金",
        "cash_equivalents_net_addition": "现金及现金等价物净增加额",
    }),
}
ALIASES = {
    "归属于母公司股东的净利润": "归属于母公司所有者的净利润",
    "所有者权益(或股东权益)合计": "所有者权益合计",
    "所有者权益（或股东权益）合计": "所有者权益合计",
}
METADATA = {"报告日", "披露日", "币种", "报表口径", "数据来源"}


def number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def cutoff_date(curr_date):
    if curr_date is None:
        return datetime.now(SHANGHAI).date().isoformat()
    return datetime.strptime(str(curr_date), "%Y-%m-%d").date().isoformat()


def _date(value, milliseconds=False):
    if milliseconds:
        parsed = pd.to_datetime(value, unit="ms", utc=True, errors="coerce")
        if pd.isna(parsed):
            return None
        return parsed.tz_convert(SHANGHAI).date().isoformat()
    parsed = pd.to_datetime(str(value), errors="coerce")
    return None if pd.isna(parsed) else parsed.date().isoformat()


def _frame(rows, freq, curr_date):
    if freq not in {"annual", "quarterly"}:
        raise ValueError("frequency must be annual or quarterly")
    cutoff = cutoff_date(curr_date)
    valid = []
    rejected = 0
    for row in rows:
        period = row.get("报告日")
        published = row.get("披露日")
        if not period or not published or row.get("币种") != "CNY":
            rejected += 1
            continue
        if "合并" not in row.get("报表口径", ""):
            rejected += 1
            continue
        if period > cutoff or published > cutoff:
            continue
        if freq == "annual" and not period.endswith("-12-31"):
            continue
        valid.append(row)
    frame = pd.DataFrame(valid)
    if not frame.empty:
        frame = frame.sort_values(["报告日", "披露日"], ascending=False)
        frame = frame.drop_duplicates("报告日").head(8).reset_index(drop=True)
    frame.attrs["rejected_rows"] = rejected
    return frame


def parse_sina(payload, report_type, freq, curr_date):
    REPORT_FIELDS[report_type]
    result = payload.get("result", {})
    if result.get("status", {}).get("code") != 0:
        raise ValueError("Sina financial response failed")
    reports = result.get("data", {}).get("report_list")
    if not isinstance(reports, dict):
        raise ValueError("Sina financial report_list schema changed")
    rows = []
    for period, report in reports.items():
        row = {
            "报告日": _date(period),
            "披露日": _date(report.get("publish_date")),
            "币种": report.get("rCurrency"),
            "报表口径": report.get("rType", ""),
            "数据来源": "新浪财经",
        }
        for item in report.get("data", []):
            if item.get("item_display_type") == 1:
                continue
            title = item.get("item_title", "")
            if title and title not in METADATA:
                row[ALIASES.get(title, title)] = number(item.get("item_value"))
        rows.append(row)
    return _frame(rows, freq, curr_date)


def hithink_enabled():
    return bool(os.environ.get("HITHINK_FINANCE_API_KEY", "").strip()) and (
        os.environ.get("HITHINK_FINANCE_ENABLED", "1").lower() not in {"0", "false", "off"}
    )


def _hithink_get(endpoint, params):
    if not hithink_enabled():
        raise RuntimeError("HiThink未启用或服务进程未配置凭据")
    headers = {"X-api-key": os.environ["HITHINK_FINANCE_API_KEY"], "Accept": "application/json"}
    for attempt in range(2):
        try:
            response = requests.get(
                BASE_URL + "/api/a-share/financials/" + endpoint,
                params=params, headers=headers, timeout=(5, 20), allow_redirects=False,
            )
        except requests.RequestException:
            if attempt == 0:
                time.sleep(0.3)
                continue
            raise RuntimeError("HiThink网络连接失败") from None
        if response.status_code in {429, 502, 503, 504} and attempt == 0:
            time.sleep(0.3)
            continue
        if response.status_code != 200:
            raise RuntimeError(f"HiThink HTTP状态异常: {response.status_code}")
        try:
            payload = response.json()
        except ValueError:
            raise RuntimeError("HiThink返回非JSON数据") from None
        if not isinstance(payload, dict) or payload.get("code") != 0:
            raise RuntimeError("HiThink认证、权限或业务请求失败；未使用返回数值")
        data = payload.get("data")
        if not isinstance(data, dict):
            raise RuntimeError("HiThink数据结构异常")
        return data
    raise RuntimeError("HiThink请求未完成")


def hithink_statements(code, suffix, report_type, freq, curr_date):
    endpoint, fields = REPORT_FIELDS[report_type]
    cutoff = cutoff_date(curr_date)
    end = datetime.strptime(cutoff, "%Y-%m-%d").replace(tzinfo=SHANGHAI)
    start = end - timedelta(days=365 * 5)
    thscode = f"{code}.{suffix.upper()}"
    data = _hithink_get(endpoint, {
        "thscode": thscode, "period": freq,
        "start": int(start.timestamp() * 1000),
        "end": int((end + timedelta(days=1)).timestamp() * 1000) - 1,
    })
    items = data.get("item")
    if not isinstance(items, list):
        raise ValueError("HiThink financial item schema changed")
    rows = []
    for item in items:
        if item.get("thscode") != thscode:
            raise ValueError("HiThink returned a different security")
        row = {
            "报告日": _date(item.get("period_end_ms"), milliseconds=True),
            "披露日": _date(item.get("report_date_ms"), milliseconds=True),
            "币种": item.get("currency"),
            "报表口径": "合并期末" if report_type == "资产负债表" else "合并年初至报告期末累计",
            "数据来源": "HiThink Financial-API",
        }
        row.update({title: number(item.get(field)) for field, title in fields.items()})
        rows.append(row)
    return _frame(rows, freq, cutoff)


def reconcile(primary, secondary):
    lines = []
    checked = 0
    if primary.empty or secondary.empty:
        return "[交叉核验未完成] 仅单一来源有可用报表，不能视为双源验证。"
    common = (set(primary.columns) & set(secondary.columns)) - METADATA
    for _, first in primary.iterrows():
        matches = secondary[secondary["报告日"] == first["报告日"]]
        if matches.empty:
            continue
        second = matches.iloc[0]
        for field in sorted(common):
            left, right = number(first[field]), number(second[field])
            if left is None or right is None:
                continue
            checked += 1
            if not math.isclose(left, right, rel_tol=1e-10, abs_tol=0.01):
                lines.append(
                    f"[数据冲突] {first['报告日']} {field}: 新浪={left:g}; HiThink={right:g}。"
                    "不得择一使用或据此推导结论，需核对公告及版本。"
                )
        if first["披露日"] != second["披露日"]:
            lines.append(f"[版本待核验] {first['报告日']} 两源披露日不同，不能证明是同一披露版本。")
    summary = f"已对照{checked}个同报告期非空字段；数值一致不等于独立公告核验。"
    return summary + ("\n" + "\n".join(lines) if lines else "")


def hithink_indicators(code, suffix, statement):
    period = statement["报告日"]
    quarter = (int(period[5:7]) - 1) // 3 + 1
    report = f"{period[:4]}-{quarter}"
    data = _hithink_get("indicators", {"thscode": f"{code}.{suffix.upper()}", "report": report})
    definitions = {
        "index_weighted_avg_roe": ("加权平均ROE", "%"),
        "index_deduct_weighted_avg_roe": ("扣非加权平均ROE", "%"),
        "assets_debt_ratio": ("资产负债率", "%"),
        "sale_gross_margin": ("销售毛利率", "%"),
        "current_ratio": ("流动比率", "倍"),
        "quick_ratio": ("速动比率", "倍"),
        "earned_interest_multiple": ("利息保障倍数", "倍"),
    }
    values = {
        item.get("index_id"): number(item.get("value"))
        for ability in data.get("abilities", [])
        for item in ability.get("indicators", [])
    }
    lines = [f"## HiThink财务指标 | 报告期{period} | 财报披露日{statement['披露日']}"]
    for field, (label, unit) in definitions.items():
        value = values.get(field)
        lines.append(f"{label}: {value:g}{unit}" if value is not None else f"{label}: [数据缺失: 未返回有限数值]")
    lines.append("指标接口不提供自身历史版本，不能证明历史时点可得；历史分析不调用此接口。")
    lines.append("经营现金流与利润均为负时，不把负负相除的比例解释为盈利质量良好。")
    return "\n".join(lines)
