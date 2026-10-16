"""TSMC 法說事件研究 / TSMX 觸價機率 Monte Carlo / 相關係數。

所有數字皆來自 yfinance 實際下載資料。任一代號下載失敗即在 results.md 標註並以非 0 結束，
不使用任何模擬或估計的價格資料替代。
"""

import sys
import time
import traceback
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

TICKERS = ["TSM", "TSMX", "NVDA", "AVGO", "DRAM", "2330.TW", "^SOX"]
# 3 年期間；起點提前到 2023-09-01，讓第一季（2023-10-19）的 T-6 也有資料
START = "2023-09-01"
RETRIES = 3
RETRY_WAIT = 10

EARNINGS_DATES = [
    "2023-10-19", "2024-01-18", "2024-04-18", "2024-07-18", "2024-10-17", "2025-01-16",
    "2025-04-17", "2025-07-17", "2025-10-16", "2026-01-15", "2026-04-16", "2026-07-16",
]
CURRENT_EVENT = "2026-10-15"
POST_HORIZONS = [5, 10, 20]
MIN_N = 8

N_PATHS = 10_000
MC_HORIZONS = [10, 20, 40]
UPPER = 100.0
LOWER_FIXED = 65.0
LEVERAGE = 2.0
DEFAULT_EXPENSE_RATIO = 0.0095
SEED = 20261015

CORR_TICKERS = ["TSM", "NVDA", "AVGO", "DRAM", "2330.TW"]
CORR_WINDOW = 60

TPE = timezone(timedelta(hours=8))


# ---------------------------------------------------------------- data

def fetch(ticker, log):
    """下載日線（auto_adjust=True），失敗重試；仍為空則回傳 None。"""
    for attempt in range(1, RETRIES + 1):
        try:
            df = yf.download(ticker, start=START, interval="1d", auto_adjust=True,
                             progress=False, threads=False)
        except Exception as e:  # noqa: BLE001
            log(f"[{ticker}] attempt {attempt}/{RETRIES} exception: {e!r}")
            df = None
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
            df = df.dropna(subset=["Close"])
            if not df.empty:
                log(f"[{ticker}] ok: {len(df)} rows, {df.index[0].date()} ~ {df.index[-1].date()}")
                return df
        log(f"[{ticker}] attempt {attempt}/{RETRIES} empty")
        if attempt < RETRIES:
            time.sleep(RETRY_WAIT)
    return None


def fetch_earnings_dates(log):
    """yfinance get_earnings_dates("TSM") 的日期集合；失敗回傳 None 與原因。"""
    try:
        ed = yf.Ticker("TSM").get_earnings_dates(limit=24)
        if ed is None or ed.empty:
            return None, "get_earnings_dates 回傳空值"
        idx = pd.to_datetime(ed.index)
        if idx.tz is not None:
            idx = idx.tz_convert("America/New_York").tz_localize(None)
        return sorted({d.normalize() for d in idx}), None
    except Exception as e:  # noqa: BLE001
        log(f"get_earnings_dates failed: {e!r}")
        return None, f"get_earnings_dates 失敗：{type(e).__name__}"


def fetch_expense_ratio(log):
    """回傳 (費用率小數, 說明)。"""
    t = yf.Ticker("TSMX")
    try:
        info = t.info or {}
        for key in ("netExpenseRatio", "annualReportExpenseRatio", "expenseRatio"):
            v = info.get(key)
            if v is not None and float(v) > 0:
                v = float(v)
                # yfinance 的 netExpenseRatio 以百分比表示（0.95 = 0.95%）
                ratio = v / 100 if v > 0.05 else v
                return ratio, f"yfinance info['{key}'] = {v}"
    except Exception as e:  # noqa: BLE001
        log(f"TSMX info failed: {e!r}")
    try:
        ops = t.funds_data.fund_operations
        if ops is not None and not ops.empty:
            for label in ops.index:
                if "expense" in str(label).lower():
                    v = float(ops.loc[label].iloc[0])
                    if v > 0:
                        ratio = v / 100 if v > 0.05 else v
                        return ratio, f"yfinance funds_data.fund_operations['{label}'] = {v}"
    except Exception as e:  # noqa: BLE001
        log(f"TSMX funds_data failed: {e!r}")
    return DEFAULT_EXPENSE_RATIO, "yfinance 查無費用率，使用預設值 0.0095（未經查證）"


# ---------------------------------------------------------------- formatting

def pct(x, d=2):
    return "N/A" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.{d}f}%"


def num(x, d=2):
    return "N/A" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.{d}f}"


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# ---------------------------------------------------------------- task 1

def event_metrics(close, t):
    idx = close.index
    nan = float("nan")
    m = {"date": t, "in_data": t in idx}
    keys = ["close_T", "pre", "day0"] + [f"post{n}" for n in POST_HORIZONS] + ["maxDD20"]
    m.update({k: nan for k in keys})
    if not m["in_data"]:
        return m
    i = idx.get_loc(t)
    c = close.values
    m["close_T"] = c[i]
    if i >= 6:
        m["pre"] = c[i - 1] / c[i - 6] - 1
    if i >= 1:
        m["day0"] = c[i] / c[i - 1] - 1
    for n in POST_HORIZONS:
        if i + n < len(c):
            m[f"post{n}"] = c[i + n] / c[i] - 1
    if i + 20 < len(c):
        m["maxDD20"] = min(0.0, c[i + 1:i + 21].min() / c[i] - 1)
    return m


def stats_rows(rows, cols):
    out = []
    for col in cols:
        v = np.array([r[col] for r in rows], dtype=float)
        v = v[~np.isnan(v)]
        n = len(v)
        if n == 0:
            out.append([col, 0, "N/A", "N/A", "N/A", "N/A", "無資料"])
            continue
        note = "樣本不足，結論僅供參考" if n < MIN_N else ""
        out.append([col, n, pct(v.mean()), pct(float(np.median(v))), pct((v > 0).mean(), 1),
                    pct(v.min()), note])
    return out


def task1(data, log):
    close = data["TSM"]["Close"]
    yahoo_dates, yahoo_err = fetch_earnings_dates(log)
    lines = ["## 任務 1：TSMC 法說事件研究（TSM）", ""]
    lines.append("定義：T = 法說當天（美股反應日）。pre = close(T-1)/close(T-6)-1；day0 = close(T)/close(T-1)-1；"
                 "postN = close(T+N)/close(T)-1；maxDD20 = T+1~T+20 收盤相對 close(T) 的最大跌幅（無下跌記 0）。")
    lines.append("")

    # 日期交叉驗證
    lines.append("### 法說日期交叉驗證（yfinance get_earnings_dates）")
    lines.append("")
    check_rows = []
    for d in EARNINGS_DATES + [CURRENT_EVENT]:
        dt = pd.Timestamp(d)
        if yahoo_dates is None:
            status = f"無法驗證（{yahoo_err}）"
        elif dt in yahoo_dates:
            status = "一致"
        else:
            near = [y for y in yahoo_dates if abs((y - dt).days) <= 10]
            status = ("**不一致**：Yahoo 為 " + ", ".join(str(y.date()) for y in near)) if near \
                else "Yahoo 清單無此季（未涵蓋）"
        check_rows.append([d, "是" if dt in close.index else "**否**", status])
    lines.append(md_table(["清單日期", "TSM 當日有交易資料", "Yahoo 比對"], check_rows))
    lines.append("")

    rows = [event_metrics(close, pd.Timestamp(d)) for d in EARNINGS_DATES]
    lines.append("### 逐季表格")
    lines.append("")
    tab = []
    for r in rows:
        tab.append([r["date"].date(), num(r["close_T"]), pct(r["pre"]), pct(r["day0"]),
                    pct(r["post5"]), pct(r["post10"]), pct(r["post20"]), pct(r["maxDD20"])])
    lines.append(md_table(["T", "close(T)", "pre", "day0", "post5", "post10", "post20", "maxDD20"], tab))
    lines.append("")

    cur = event_metrics(close, pd.Timestamp(CURRENT_EVENT))
    lines.append(f"### 本次 {CURRENT_EVENT}（只列 T 日數據，不納入統計）")
    lines.append("")
    if cur["in_data"]:
        tsmx = data["TSMX"]
        tx = tsmx.loc[pd.Timestamp(CURRENT_EVENT)] if pd.Timestamp(CURRENT_EVENT) in tsmx.index else None
        lines.append(md_table(["T", "TSM close(T)", "TSM pre", "TSM day0", "TSMX close(T)", "TSMX Low(T)"],
                              [[CURRENT_EVENT, num(cur["close_T"]), pct(cur["pre"]), pct(cur["day0"]),
                                num(tx["Close"]) if tx is not None else "N/A",
                                num(tx["Low"]) if tx is not None else "N/A"]]))
    else:
        lines.append(f"TSM 資料中沒有 {CURRENT_EVENT} 這一天。")
    lines.append("")

    cols = ["pre", "day0", "post5", "post10", "post20", "maxDD20"]
    hdr = ["欄位", "n", "平均", "中位數", "勝率(>0)", "最差值", "備註"]
    lines.append("### 統計：全部樣本")
    lines.append("")
    lines.append(md_table(hdr, stats_rows(rows, cols)))
    lines.append("")
    neg = [r for r in rows if not np.isnan(r["day0"]) and r["day0"] < 0]
    lines.append("### 統計：day0 < 0 子樣本")
    lines.append("")
    lines.append(f"子樣本日期：{', '.join(str(r['date'].date()) for r in neg) or '無'}")
    lines.append("")
    lines.append(md_table(hdr, stats_rows(neg, cols)))
    lines.append("")
    lines.append("註：maxDD20 以 0 為上限，勝率(>0) 恆為 0%，僅看平均/中位數/最差值。")
    lines.append("")

    def s(rs, col):
        v = np.array([r[col] for r in rs], dtype=float)
        v = v[~np.isnan(v)]
        return v

    summary = []
    a5, a20, ad = s(rows, "post5"), s(rows, "post20"), s(rows, "maxDD20")
    d0 = s(rows, "day0")
    summary.append(f"- TSM 過去 {len(d0)} 季法說 day0：平均 {pct(d0.mean())}、中位數 {pct(float(np.median(d0)))}、"
                   f"上漲比例 {pct((d0 > 0).mean(), 1)}、最差 {pct(d0.min())}。")
    summary.append(f"- post5：n={len(a5)}，平均 {pct(a5.mean())}、勝率 {pct((a5 > 0).mean(), 1)}；"
                   f"post20：n={len(a20)}，平均 {pct(a20.mean())}、勝率 {pct((a20 > 0).mean(), 1)}、最差 {pct(a20.min())}。")
    summary.append(f"- maxDD20：n={len(ad)}，平均 {pct(ad.mean())}、中位數 {pct(float(np.median(ad)))}、最差 {pct(ad.min())}。")
    n5, n20 = s(neg, "post5"), s(neg, "post20")
    if len(neg):
        summary.append(f"- day0<0 子樣本 n={len(neg)}：post5 平均 {pct(n5.mean())}、勝率 {pct((n5 > 0).mean(), 1)}；"
                       f"post20 平均 {pct(n20.mean())}、勝率 {pct((n20 > 0).mean(), 1)}"
                       + ("（n<8，樣本不足）。" if len(neg) < MIN_N else "。"))
    else:
        summary.append("- day0<0 子樣本 n=0。")
    if cur["in_data"]:
        summary.append(f"- 本次 {CURRENT_EVENT}：TSM day0 {pct(cur['day0'])}、pre {pct(cur['pre'])}，close {num(cur['close_T'])}。")
    return lines, summary


# ---------------------------------------------------------------- task 2

def realized_vol(ret, n):
    return float(ret.tail(n).std(ddof=1) * np.sqrt(252))


def simulate(tsm_daily, s0, er):
    """tsm_daily: (N_PATHS, H) 的 TSM 日簡單報酬 → TSMX 收盤價路徑。"""
    gross = 1 + LEVERAGE * tsm_daily - er / 252
    gross = np.maximum(gross, 1e-9)
    return s0 * np.cumprod(gross, axis=1)


def touch_stats(paths, h, lower):
    p = paths[:, :h]
    big = h + 1
    hit_u = p >= UPPER
    hit_l = p <= lower
    any_u, any_l = hit_u.any(1), hit_l.any(1)
    first_u = np.where(any_u, hit_u.argmax(1), big)
    first_l = np.where(any_l, hit_l.argmax(1), big)
    return {
        "P(≥100)": any_u.mean(),
        "P(≤L)": any_l.mean(),
        "P(≤65)": (p <= LOWER_FIXED).any(1).mean(),
        "P(先100再L)": (any_u & any_l & (first_u < first_l)).mean(),
        "P(僅100)": (any_u & ~any_l).mean(),
        "P(先L)": (any_l & (first_l < first_u)).mean(),
        "P(皆未)": (~any_u & ~any_l).mean(),
        "期末中位數": float(np.median(p[:, -1])),
    }


def task2(data, log):
    tsm_ret = data["TSM"]["Close"].pct_change().dropna()
    tsmx = data["TSMX"]
    s0 = float(tsmx["Close"].iloc[-1])
    s0_date = tsmx.index[-1].date()
    ev = pd.Timestamp(CURRENT_EVENT)
    if ev not in tsmx.index:
        raise RuntimeError(f"TSMX 資料中沒有 {CURRENT_EVENT}，無法取得停損參考線 L")
    L = float(tsmx.loc[ev, "Low"])
    er, er_note = fetch_expense_ratio(log)

    vol20, vol60 = realized_vol(tsm_ret, 20), realized_vol(tsm_ret, 60)
    boot_pool = tsm_ret.tail(60).values
    H = max(MC_HORIZONS)
    rng = np.random.default_rng(SEED)

    models = {}
    models["A. Bootstrap（TSM 近 60 日報酬重抽）"] = rng.choice(boot_pool, size=(N_PATHS, H), replace=True)
    for name, vol in (("B1. GBM σ=20日實現波動", vol20), ("B2. GBM σ=60日實現波動", vol60)):
        sd = vol / np.sqrt(252)
        z = rng.standard_normal((N_PATHS, H))
        models[name] = np.exp(sd * z - 0.5 * sd * sd) - 1  # drift=0：E[日報酬]=0

    lines = ["## 任務 2：TSMX 觸價機率（Monte Carlo）", ""]
    lines.append(md_table(["項目", "數值"], [
        ["起始價 S0（TSMX 最新收盤）", f"{num(s0)}（{s0_date}）"],
        [f"停損參考線 L（TSMX {CURRENT_EVENT} 日內 Low）", num(L)],
        ["TSMX 費用率", f"{pct(er, 2)}（{er_note}）"],
        ["TSM 20 日年化實現波動率", pct(vol20)],
        ["TSM 60 日年化實現波動率", pct(vol60)],
        ["Bootstrap 樣本期間", f"{tsm_ret.tail(60).index[0].date()} ~ {tsm_ret.tail(60).index[-1].date()}（60 日）"],
        ["路徑數 / 隨機種子", f"{N_PATHS:,} / {SEED}"],
    ]))
    lines.append("")
    lines.append(f"模型：TSMX 日報酬 = 2 × TSM 日報酬 − 費用率/252；觸價以收盤價判定（第 1 ~ N 個交易日）。"
                 f"GBM drift=0 指 TSM 日簡單報酬期望值為 0（log 報酬期望值 −σ²/2）。")
    lines.append("P(先100再L) + P(僅100) + P(先L) + P(皆未) = 100%。")
    if s0 >= UPPER or s0 <= L:
        lines.append(f"**注意：S0={num(s0)} 已在 100 以上或 L 以下，觸價機率解讀需調整。**")
    lines.append("")

    cols = ["P(≥100)", "P(≤L)", "P(≤65)", "P(先100再L)", "P(僅100)", "P(先L)", "P(皆未)", "期末中位數"]
    all_res = {}
    for name, rets in models.items():
        paths = simulate(rets, s0, er)
        rows = []
        for h in MC_HORIZONS:
            r = touch_stats(paths, h, L)
            all_res[(name, h)] = r
            rows.append([f"{h} 日"] + [num(r[c]) if c == "期末中位數" else pct(r[c], 1) for c in cols])
        lines.append(f"### {name}")
        lines.append("")
        lines.append(md_table(["期間"] + cols, rows))
        lines.append("")

    lines.append("### 2x 槓桿波動拖累（年化 ≈ σ²）")
    lines.append("")
    drag_rows = []
    for label, vol in (("20 日 σ", vol20), ("60 日 σ", vol60)):
        d = vol ** 2
        drag_rows.append([label, pct(vol), pct(d)] + [pct(d * h / 252) for h in MC_HORIZONS])
    lines.append(md_table(["σ 來源", "TSM 年化 σ", "年化拖累 σ²"] + [f"{h} 日拖累" for h in MC_HORIZONS],
                          drag_rows))
    lines.append("")

    summary = [f"- TSMX 起始價 {num(s0)}（{s0_date}），L = {num(L)}（{CURRENT_EVENT} Low），"
               f"費用率 {pct(er)}；TSM σ20 = {pct(vol20)}、σ60 = {pct(vol60)}。"]
    for name in models:
        short = name.split("（")[0]
        r20, r40 = all_res[(name, 20)], all_res[(name, 40)]
        summary.append(f"- {short}：20 日 P(≥100) {pct(r20['P(≥100)'], 1)}、P(≤L) {pct(r20['P(≤L)'], 1)}、"
                       f"P(≤65) {pct(r20['P(≤65)'], 1)}；40 日 P(≥100) {pct(r40['P(≥100)'], 1)}、"
                       f"P(≤L) {pct(r40['P(≤L)'], 1)}、期末中位數 {num(r40['期末中位數'])}。")
    summary.append(f"- 2x 年化波動拖累 ≈ {pct(vol20 ** 2)}（σ20）/ {pct(vol60 ** 2)}（σ60），"
                   f"20 日約 {pct(vol20 ** 2 * 20 / 252)} / {pct(vol60 ** 2 * 20 / 252)}。")
    return lines, summary


# ---------------------------------------------------------------- task 3

def task3(data):
    rets = {t: data[t]["Close"].pct_change().dropna() for t in CORR_TICKERS}
    joint = pd.concat(rets, axis=1, join="inner").tail(CORR_WINDOW)
    corr = joint.corr()

    lines = ["## 任務 3：相關係數（日報酬）", ""]
    lines.append(f"同日期對齊（五檔皆有交易的共同日期），最近 {len(joint)} 筆："
                 f"{joint.index[0].date()} ~ {joint.index[-1].date()}。")
    lines.append("")
    lines.append(md_table([""] + CORR_TICKERS,
                          [[a] + [f"{corr.loc[a, b]:.3f}" for b in CORR_TICKERS] for a in CORR_TICKERS]))
    lines.append("")

    tw = rets["2330.TW"].rename("tw").to_frame().reset_index()
    tw.columns = ["date", "tw"]
    us = rets["TSM"].rename("tsm").to_frame().reset_index()
    us.columns = ["us_date", "tsm"]
    lag = pd.merge_asof(tw.sort_values("date"), us.sort_values("us_date"), left_on="date",
                        right_on="us_date", direction="backward", allow_exact_matches=False)
    lag = lag.dropna().tail(CORR_WINDOW)
    c_lag = float(lag["tw"].corr(lag["tsm"]))
    c_same = float(corr.loc["TSM", "2330.TW"])

    lines.append("### 時差調整：TSM(t-1) vs 2330.TW(t)")
    lines.append("")
    lines.append("2330.TW 第 t 日收盤（台北 13:30）早於美股第 t 日開盤，故以「2330.TW 交易日之前最近一個美股交易日」"
                 "的 TSM 報酬對齊。")
    lines.append("")
    lines.append(md_table(["配對", "n", "期間（2330.TW 日期）", "相關係數"], [
        ["TSM(t) vs 2330.TW(t)（同日）", len(joint), f"{joint.index[0].date()} ~ {joint.index[-1].date()}",
         f"{c_same:.3f}"],
        ["TSM(t-1) vs 2330.TW(t)", len(lag), f"{lag['date'].iloc[0].date()} ~ {lag['date'].iloc[-1].date()}",
         f"{c_lag:.3f}"],
    ]))
    lines.append("")

    pairs = [(a, b) for i, a in enumerate(CORR_TICKERS) for b in CORR_TICKERS[i + 1:]]
    tsm_pairs = ", ".join(f"{b} {corr.loc['TSM', b]:.3f}" for b in CORR_TICKERS if b != "TSM")
    hi = max(pairs, key=lambda p: corr.loc[p])
    lo = min(pairs, key=lambda p: corr.loc[p])
    summary = [
        f"- 期間 {joint.index[0].date()} ~ {joint.index[-1].date()}，{len(joint)} 個共同交易日日報酬。",
        f"- TSM 對其他標的相關係數：{tsm_pairs}。",
        f"- 最高一組 {hi[0]}/{hi[1]} {corr.loc[hi]:.3f}；最低一組 {lo[0]}/{lo[1]} {corr.loc[lo]:.3f}。",
        f"- TSM vs 2330.TW：同日 {c_same:.3f}；TSM(t-1) vs 2330.TW(t) {c_lag:.3f}（n={len(lag)}）。",
    ]
    return lines, summary


# ---------------------------------------------------------------- main

def header(now, data, failed):
    lines = ["# TSMC 法說 / TSMX 量化結果", ""]
    lines.append(f"- 資料抓取時間：{now:%Y-%m-%d %H:%M:%S} UTC（台北 {now.astimezone(TPE):%Y-%m-%d %H:%M}）")
    lines.append(f"- 資料來源：yfinance {yf.__version__}，日線，auto_adjust=True，start={START}（約 3 年；提前起點以涵蓋 2023-10-19 的 T-6）")
    lines.append("")
    rows = []
    for t in TICKERS:
        df = data.get(t)
        if df is None:
            rows.append([t, "**下載失敗**", "", "", ""])
        else:
            rows.append([t, df.index[0].date(), df.index[-1].date(), len(df), num(float(df["Close"].iloc[-1]))])
    lines.append(md_table(["代號", "第一筆日期", "最後一筆日期", "筆數", "最後收盤"], rows))
    lines.append("")
    if failed:
        lines.append(f"**下載失敗（重試 {RETRIES} 次、間隔 {RETRY_WAIT} 秒後仍為空）：{', '.join(failed)}。"
                     f"依規則不以任何模擬或估計資料替代，分析未執行。**")
        lines.append("")
    return lines


def main(out_path="results.md", fetch_fn=fetch):
    def log(msg):
        print(msg, flush=True)

    now = datetime.now(timezone.utc)
    data, failed = {}, []
    for t in TICKERS:
        df = fetch_fn(t, log)
        if df is None:
            failed.append(t)
        data[t] = df

    lines = header(now, data, failed)
    if failed:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        log(f"FAILED tickers: {failed}")
        return 1

    l1, s1 = task1(data, log)
    l2, s2 = task2(data, log)
    l3, s3 = task3(data)
    lines += l1 + l2 + l3
    lines += ["## 可直接貼入研究 prompt 的摘要", "",
              "**4. 法說後回測（TSM）**", ""] + s1 + ["",
              "**5. 觸價機率（TSMX Monte Carlo）**", ""] + s2 + ["",
              "**9. 相關係數（近 60 交易日）**", ""] + s3 + [""]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    out = "results.md"
    try:
        code = main(out)
    except Exception:  # noqa: BLE001
        tb = traceback.format_exc()
        print(tb, flush=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write("# TSMC 法說 / TSMX 量化結果\n\n**執行失敗**（"
                    f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC）\n\n```\n{tb}```\n")
        code = 1
    sys.exit(code)
