"""Deterministic daily-price statistics shared by analyst tools."""

import math

import pandas as pd


def summarize_prices(data, cutoff):
    lines = [
        "## 程序统一量价统计（计算值，优先引用，不自行重算窗口）",
        "成交量单位：股；1手=100股。固定5/20个交易记录窗口独立于下方展示的自然日区间。",
        "按输入日线记录计算；未补造停牌/缺失交易日，复权口径未独立核验，不能据此识别资金身份。",
        "若数据包含当日盘中日线，Close和成交量是暂时值，不作收盘形态确认。",
    ]
    if not {"Date", "Close", "Volume"}.issubset(data.columns):
        return "\n".join(lines + ["[数据缺失] 日期、收盘价或成交量字段不全。"])
    frame = data.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce").dt.normalize()
    if frame["Date"].isna().any():
        return "\n".join(lines + ["[数据异常] 存在无效日期，不生成统计。"])
    frame = frame[frame["Date"] <= pd.Timestamp(cutoff)].sort_values("Date")
    if frame["Date"].duplicated().any():
        return "\n".join(lines + ["[数据异常] 日期重复，不擅自选择价格或合并成交量。"])
    frame = frame.tail(21).reset_index(drop=True)
    if frame.empty:
        return "\n".join(lines + ["[数据缺失] 基准日及以前没有日线。"])
    for field in ("Close", "Volume"):
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
        if not frame[field].map(lambda value: pd.notna(value) and math.isfinite(value)).all():
            return "\n".join(lines + [f"[数据异常] {field}含无效数值，不生成统计。"])
    if (frame["Close"] <= 0).any() or (frame["Volume"] < 0).any():
        return "\n".join(lines + ["[数据异常] 收盘价非正或成交量为负。"])
    latest = frame.iloc[-1]
    date_text = lambda row: row["Date"].strftime("%Y-%m-%d")
    lines.append(f"实际数据截止日：{date_text(latest)}；请求基准日：{cutoff}。")
    lines.append(f"最新收盘价={latest['Close']:.4f}元；当日成交量={latest['Volume']:.2f}股。")
    if len(frame) >= 2:
        previous = frame.iloc[-2]
        change = (latest["Close"] / previous["Close"] - 1) * 100
        lines.append(f"当日涨跌幅=({latest['Close']:g}/{previous['Close']:g}-1)×100%={change:.4f}%；前收盘日={date_text(previous)}。")
    averages = {}
    for window in (5, 20):
        if len(frame) < window:
            lines.append(f"[数据缺失] 近{window}日均量需{window}条记录，实际仅{len(frame)}条；不使用短窗口冒充。")
            continue
        sample = frame.tail(window)
        total = sample["Volume"].sum()
        average = total / window
        averages[window] = average
        lines.append(
            f"近{window}日日均量（含当日）：{date_text(sample.iloc[0])}至{date_text(latest)}，"
            f"{window}条；总量={total:.2f}股；均量={total:.2f}/{window}={average:.2f}股={average/10000/100:.6f}万手。"
        )
        if average > 0:
            lines.append(f"当日量/近{window}日均量（含当日）={latest['Volume']:.2f}/{average:.2f}={latest['Volume']/average:.4f}倍。")
        else:
            lines.append(f"[数据缺失] 近{window}日均量为零，不计算放量倍数。")
    if averages.get(20, 0) > 0 and 5 in averages:
        lines.append(f"近5日均量/近20日均量（均含当日）={averages[5]:.2f}/{averages[20]:.2f}={averages[5]/averages[20]:.4f}倍。")
    if len(frame) >= 6:
        previous_mean = frame.iloc[-6:-1]["Volume"].mean()
        if previous_mean > 0:
            lines.append(f"当日量/前5日均量（不含当日）={latest['Volume']:.2f}/{previous_mean:.2f}={latest['Volume']/previous_mean:.4f}倍；不得混用为含当日均量。")
    if len(frame) >= 21:
        baseline = frame.iloc[-21]
        first = frame.iloc[-20]
        change = (latest["Close"] / baseline["Close"] - 1) * 100
        lines.append(
            f"近20个交易记录累计涨跌幅：{date_text(first)}至{date_text(latest)}；"
            f"基准为窗口前一交易记录{date_text(baseline)}收盘{baseline['Close']:g}元；"
            f"({latest['Close']:g}/{baseline['Close']:g}-1)×100%={change:.4f}%。"
        )
    else:
        lines.append("[数据缺失] 20日累计涨跌幅需要20条区间记录及前一收盘价，共21条；不另换基准日凑数。")
    return "\n".join(lines)
