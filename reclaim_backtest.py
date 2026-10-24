"""TSMC 法說後「收復法說前一日收盤」進場回測。

研究問題：法說當日 TSM 收跌後，若 K 日內收盤收復 close(T-1)，收復後進場的後續報酬
是否優於未收復組與無條件基準。主規格事先固定（見 MAIN），穩健性變體只列表。

所有數字來自 yfinance 實際資料；價格或法說日期抓取失敗即以非 0 結束，不以模擬資料替代。
"""

import re
import sys
import time
import traceback
from datetime import datetime, timezone, timedelta, time as dtime

import numpy as np
import pandas as pd
import requests
import yfinance as yf

TICKER = "TSM"
START = "2012-01-01"
RETRIES = 3
RETRY_WAIT = 10

FIRST_Q = (2015, 1)
LAST_Q = (2026, 3)          # 含樣本外的本次事件
OOS_T = pd.Timestamp("2026-10-15")
PRIOR_LIST = {  # 先前任務 1 使用的清單，僅作比對欄位
    "2023-10-19", "2024-01-18", "2024-04-18", "2024-07-18", "2024-10-17", "2025-01-16",
    "2025-04-17", "2025-07-17", "2025-10-16", "2026-01-15", "2026-04-16", "2026-07-16", "2026-10-15",
}

# 主規格（事先固定，不得依結果調整）
MAIN = dict(K=10, thr=1.0, entry="open", filter_neg=True)
HOLD = 20
FEE = 0.0099
LEV = 2.0
N_BOOT = 10_000
N_PERM = 10_000
SEED = 20261023
MIN_N = 15
INSUFFICIENT = "樣本不足，不得推論"

ROBUST = [
    ("K = 5", dict(K=5)),
    ("K = 15", dict(K=15)),
    ("門檻 = P_ref × 0.99", dict(thr=0.99)),
    ("門檻 = P_ref × 1.01", dict(thr=1.01)),
    ("進場 = close(t_s)（前視疑慮，僅供比較）", dict(entry="close")),
    ("全部法說（不限 day0<0）", dict(filter_neg=False)),
]

NY = "America/New_York"
TPE = timezone(timedelta(hours=8))


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- formatting

def isnan(x):
    return x is None or (isinstance(x, float) and np.isnan(x))


def pct(x, d=2):
    return "N/A" if isnan(x) else f"{x * 100:.{d}f}%"


def num(x, d=2):
    return "N/A" if isnan(x) else f"{x:,.{d}f}"


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# ---------------------------------------------------------------- data

def fetch_prices():
    for attempt in range(1, RETRIES + 1):
        try:
            df = yf.download(TICKER, start=START, interval="1d", auto_adjust=True,
                             progress=False, threads=False)
        except Exception as e:  # noqa: BLE001
            log(f"[{TICKER}] attempt {attempt} exception: {e!r}")
            df = None
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
            df = df.dropna(subset=["Open", "High", "Low", "Close"])
            if not df.empty:
                log(f"[{TICKER}] ok: {len(df)} rows {df.index[0].date()} ~ {df.index[-1].date()}")
                return df
        log(f"[{TICKER}] attempt {attempt} empty")
        if attempt < RETRIES:
            time.sleep(RETRY_WAIT)
    return None


def fetch_yahoo_earnings():
    """回傳 DataFrame[ts(NY tz-aware), reported]；失敗回傳 None。"""
    for attempt in range(1, RETRIES + 1):
        try:
            ed = yf.Ticker(TICKER).get_earnings_dates(limit=100)
            if ed is not None and not ed.empty:
                idx = pd.to_datetime(ed.index)
                idx = idx.tz_localize(NY) if idx.tz is None else idx.tz_convert(NY)
                rep_col = next((c for c in ed.columns if "reported" in str(c).lower()), None)
                out = pd.DataFrame({"ts": idx,
                                    "reported": ed[rep_col].values if rep_col else np.nan})
                log(f"get_earnings_dates ok: {len(out)} rows {out.ts.min()} ~ {out.ts.max()}")
                return out
        except Exception as e:  # noqa: BLE001
            log(f"get_earnings_dates attempt {attempt} exception: {e!r}")
        if attempt < RETRIES:
            time.sleep(RETRY_WAIT)
    return None


MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
DATE_RE_1 = re.compile(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(\d{1,2}),?\s+(\d{4})")
DATE_RE_2 = re.compile(r"\b(20\d{2})[/-](\d{1,2})[/-](\d{1,2})\b")


def fetch_tsmc_ir(year, q):
    """TSMC 投資人關係季度頁面。回傳 (date|None, url, 說明)。
    只接受出現在 'conference'/'earnings' 字樣 200 字內、且落在季末 +7~+45 天的唯一日期。"""
    url = f"https://investor.tsmc.com/english/quarterly-results/{year}/q{q}"
    try:
        r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    except Exception as e:  # noqa: BLE001
        return None, url, f"連線失敗 {type(e).__name__}"
    if r.status_code != 200:
        return None, url, f"HTTP {r.status_code}"
    text = re.sub(r"<[^>]+>", " ", r.text)
    text = re.sub(r"\s+", " ", text)
    q_end = pd.Timestamp(year=year, month=3 * q, day=1) + pd.offsets.MonthEnd(0)
    lo, hi = q_end + pd.Timedelta(days=7), q_end + pd.Timedelta(days=45)
    found = set()
    for m in list(DATE_RE_1.finditer(text)) + list(DATE_RE_2.finditer(text)):
        g = m.groups()
        try:
            if m.re is DATE_RE_1:
                d = pd.Timestamp(year=int(g[2]), month=MONTHS[g[0][:3].lower()], day=int(g[1]))
            else:
                d = pd.Timestamp(year=int(g[0]), month=int(g[1]), day=int(g[2]))
        except ValueError:
            continue
        ctx = text[max(0, m.start() - 200):m.end() + 200].lower()
        if lo <= d <= hi and ("conference" in ctx or "earnings" in ctx):
            found.add(d)
    if len(found) == 1:
        return found.pop(), url, "ok"
    if not found:
        return None, url, "頁面中找不到符合條件的日期"
    return None, url, "多個候選日期：" + ", ".join(str(d.date()) for d in sorted(found))


# ---------------------------------------------------------------- dates

def fiscal_quarter(d):
    """公布日 → 所公布的季度（1–3 月公布前一年 Q4，依此類推）。"""
    m, y = d.month, d.year
    if m <= 3:
        return (y - 1, 4)
    return (y, (m - 1) // 3)


def next_trading(idx, d, strictly_after=False):
    pos = idx.searchsorted(d, side="right" if strictly_after else "left")
    return idx[pos] if pos < len(idx) else None


def ts_to_T(idx, ts):
    local = ts.tz_convert(NY)
    d = pd.Timestamp(local.date())
    t = local.time()
    if t >= dtime(16, 0):
        return next_trading(idx, d, strictly_after=True), "16:00 後 → 下一交易日"
    if t < dtime(9, 30):
        return next_trading(idx, d), "09:30 前 → 當日（非交易日則下一交易日）"
    return next_trading(idx, d), "**盤中時間戳**（09:30–16:00），取當日"


def all_quarters():
    out, (y, q) = [], FIRST_Q
    while (y, q) <= LAST_Q:
        out.append((y, q))
        y, q = (y + 1, 1) if q == 4 else (y, q + 1)
    return out


def build_dates(px):
    idx = px.index
    ye = fetch_yahoo_earnings()
    rows, exclusions = {}, []
    by_q = {}
    if ye is not None:
        today = pd.Timestamp(datetime.now(timezone.utc).date())
        for ts, rep in zip(ye.ts, ye.reported):
            if pd.Timestamp(ts.tz_convert(NY).date()) > today:
                continue
            T, rule = ts_to_T(idx, ts)
            if T is None:
                continue
            by_q.setdefault(fiscal_quarter(T), []).append((ts, T, rule))
    for yq in all_quarters():
        label = f"{yq[0]}Q{yq[1]}"
        cands = by_q.get(yq, [])
        Ts = sorted({c[1] for c in cands})
        if len(Ts) == 1:
            ts, T, rule = cands[0]
            rows[yq] = dict(q=label, raw=ts.strftime("%Y-%m-%d %H:%M %Z"), T=T, rule=rule,
                            source="yfinance get_earnings_dates")
            continue
        if len(Ts) > 1:
            reason = "Yahoo 同季多筆時間戳對應不同 T：" + ", ".join(str(t.date()) for t in Ts)
        else:
            reason = "Yahoo 無此季"
        d, url, note = fetch_tsmc_ir(*yq)
        log(f"TSMC IR {label}: {note} {d}")
        if d is not None:
            # IR 日期為台北日期；14:00 台北 = 美東同日凌晨（< 09:30）→ 當日或下一交易日
            T = next_trading(idx, d)
            if len(Ts) > 1 and T not in Ts:
                exclusions.append([label, reason + f"；TSMC IR 為 {d.date()}，仍無法對應"])
                continue
            rows[yq] = dict(q=label, raw=f"{d.date()}（台北日期）", T=T,
                            rule="台北 14:00 = 美東凌晨 → 當日（非交易日則下一交易日）",
                            source=f"TSMC IR {url}" + ("（解決 Yahoo 衝突）" if len(Ts) > 1 else ""))
        else:
            exclusions.append([label, f"{reason}；TSMC IR：{note}（{url}）"])
    return rows, exclusions, ye is not None


# ---------------------------------------------------------------- event engine

def x2_path_return(daily):
    return float(np.prod(1 + LEV * np.asarray(daily) - FEE / 252) - 1)


def run_event(px, T, K, thr, entry):
    """回傳 dict；資料不足的欄位為 NaN。entry='open' → open(t_s+1)；'close' → close(t_s)。"""
    O, H, L, C = (px[c].values for c in ("Open", "High", "Low", "Close"))
    idx = px.index
    i = idx.get_loc(T)
    nan = float("nan")
    ev = dict(T=T, i=i, p_ref=C[i - 1], day0=C[i] / C[i - 1] - 1, reclaim=None, k=None, ts=None,
              e=None, entry_date=None, entry_px=nan, r5=nan, r10=nan, r20=nan, dd20=nan, x2_20=nan,
              stop=nan, exit_reason=None, exit_date=None, rule_r=nan, rule_x2=nan, complete=False)
    target = thr * ev["p_ref"]
    s = None
    for k in range(1, K + 1):
        if i + k >= len(C):
            return ev  # 觀察窗未結束
        if C[i + k] >= target:
            s = i + k
            ev.update(reclaim=True, k=k, ts=idx[s])
            break
    if s is None:
        ev["reclaim"] = False
        s = i + K  # 未收復：視同在 T+K 收盤判定
    if entry == "open":
        e = s + 1
        if e >= len(C):
            return ev
        ev.update(e=e, entry_date=idx[e], entry_px=O[e])
        closes = C[e:e + HOLD]
        first = C[e] / O[e] - 1
        day_rets = np.concatenate([[first], C[e + 1:e + HOLD] / C[e:e + HOLD - 1] - 1])
        offset = 0  # 第 n 日收盤 = C[e + n - 1]
    else:
        e = s
        ev.update(e=e, entry_date=idx[e], entry_px=C[e])
        closes = C[e + 1:e + 1 + HOLD]
        day_rets = C[e + 1:e + 1 + HOLD] / C[e:e + HOLD] - 1
        offset = 1  # 第 n 日收盤 = C[e + n]
    px0 = ev["entry_px"]
    for n in (5, 10, 20):
        j = e + n - 1 + offset
        if j < len(C):
            ev[f"r{n}"] = C[j] / px0 - 1
    if len(closes) == HOLD:
        ev["dd20"] = min(0.0, closes.min() / px0 - 1)
        ev["x2_20"] = x2_path_return(day_rets)
        ev["complete"] = True

    # 規則模擬：停損 = T ~ 進場前一日的最低 Low；收盤跌破 → 次日開盤出場；第 20 日收盤時間到
    stop_hi = e if entry == "open" else e + 1
    ev["stop"] = float(L[i:stop_hi].min())
    first_day = e + offset
    for n in range(1, HOLD + 1):
        j = first_day + n - 1
        if j >= len(C):
            return ev
        if C[j] < ev["stop"] and n < HOLD:
            if j + 1 >= len(C):
                return ev
            ev.update(exit_reason="停損", exit_date=idx[j + 1])
            rets = list(day_rets[:n]) + [O[j + 1] / C[j] - 1]
            ev["rule_r"] = O[j + 1] / px0 - 1
            ev["rule_x2"] = x2_path_return(rets)
            ev["rule_days"] = n
            return ev
    j = first_day + HOLD - 1
    ev.update(exit_reason="時間到", exit_date=idx[j], rule_r=C[j] / px0 - 1,
              rule_x2=x2_path_return(day_rets), rule_days=HOLD)
    return ev


# ---------------------------------------------------------------- statistics

def boot_ci(x, rng_seed=SEED):
    rng = np.random.default_rng(rng_seed)
    x = np.asarray(x, float)
    means = x[rng.integers(0, len(x), size=(N_BOOT, len(x)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def perm_p(a, b):
    rng = np.random.default_rng(SEED)
    a, b = np.asarray(a, float), np.asarray(b, float)
    obs = abs(a.mean() - b.mean())
    pool = np.concatenate([a, b])
    na, hits = len(a), 0
    for _ in range(N_PERM):
        rng.shuffle(pool)
        if abs(pool[:na].mean() - pool[na:].mean()) >= obs - 1e-15:
            hits += 1
    return (hits + 1) / (N_PERM + 1)


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")


def vals(evs, key):
    v = np.array([e[key] for e in evs], dtype=float)
    return v[~np.isnan(v)]


def describe_rows(group_name, evs, metrics):
    rows = []
    for key, label in metrics:
        v = vals(evs, key)
        n = len(v)
        if n == 0:
            rows.append([group_name, label, 0] + ["N/A"] * 7 + ["無資料"])
            continue
        if n >= MIN_N:
            lo, hi = boot_ci(v)
            ci, concl = f"[{pct(lo)}, {pct(hi)}]", ""
        else:
            ci, concl = "—", INSUFFICIENT
        rows.append([group_name, label, n, pct(v.mean()), pct(float(np.median(v))),
                     pct((v > 0).mean(), 1), pct(v.min()), pct(v.max()), ci, concl])
    return rows


def compare(rec, unr, key="r20"):
    a, b = vals(rec, key), vals(unr, key)
    out = dict(na=len(a), nb=len(b), diff=float("nan"), p=None, d=None, ok=False)
    if len(a) and len(b):
        out["diff"] = a.mean() - b.mean()
    if len(a) >= MIN_N and len(b) >= MIN_N:
        out.update(p=perm_p(a, b), d=cohens_d(a, b), ok=True)
    return out


# ---------------------------------------------------------------- benchmark

def benchmark(px, start_T):
    O, C = px["Open"].values, px["Close"].values
    idx = px.index
    i0 = idx.get_loc(start_T)
    r = [C[t + HOLD] / O[t + 1] - 1 for t in range(i0, len(C) - HOLD)]
    return np.array(r)


# ---------------------------------------------------------------- report

def run_spec(px, dates_in, spec):
    evs = []
    for T in dates_in:
        ev = run_event(px, T, spec["K"], spec["thr"], spec["entry"])
        if spec["filter_neg"] and not ev["day0"] < 0:
            continue
        evs.append(ev)
    return evs


def main(out_path="reclaim_backtest.md", fetch_px=fetch_prices, build_dates_fn=build_dates):
    now = datetime.now(timezone.utc)
    px = fetch_px()
    if px is None:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"# 法說後收復進場回測\n\n**TSM 價格下載失敗（重試 {RETRIES} 次），未執行。不以模擬資料替代。**\n")
        return 1
    date_rows, exclusions, yahoo_ok = build_dates_fn(px)
    if not date_rows:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("# 法說後收復進場回測\n\n**無任何可驗證的法說日期（Yahoo 與 TSMC IR 皆失敗），未執行。**\n\n"
                    + md_table(["季度", "排除原因"], exclusions) + "\n")
        return 1

    idx = px.index
    last = idx[-1]
    in_sample, status = [], {}
    for yq, r in sorted(date_rows.items()):
        T = r["T"]
        if T == OOS_T:
            status[yq] = "樣本外（本次事件）"
            continue
        if T >= OOS_T:
            status[yq] = "排除：晚於本次事件"
            exclusions.append([r["q"], "T 晚於本次事件"])
            continue
        i = idx.get_loc(T)
        need = i + max(s.get("K", MAIN["K"]) for _, s in [("", MAIN)] + ROBUST) + 1 + HOLD
        if i < 1 or need >= len(idx):
            status[yq] = "排除：資料長度不足"
            exclusions.append([r["q"], f"T={T.date()} 之後資料不足以完成 K+1+20 日觀察"])
            continue
        status[yq] = "納入"
        in_sample.append(T)
    oos = [r for r in date_rows.values() if r["T"] == OOS_T]

    # ---- 主規格
    all_events = [run_event(px, T, MAIN["K"], MAIN["thr"], MAIN["entry"]) for T in in_sample]
    neg = [e for e in all_events if e["day0"] < 0]
    rec = [e for e in neg if e["reclaim"]]
    unr = [e for e in neg if e["reclaim"] is False]
    bench = benchmark(px, in_sample[0]) if in_sample else np.array([])
    bmean = float(bench.mean()) if len(bench) else float("nan")
    cmp_main = compare(rec, unr)
    rec_r20 = vals(rec, "r20")
    excess = rec_r20.mean() - bmean if len(rec_r20) else float("nan")
    excess_ci = boot_ci(rec_r20 - bmean) if len(rec_r20) >= MIN_N else None
    any_small = min(len(neg), len(rec), len(unr)) < MIN_N

    # ---- 本次事件
    oos_lines, oos_status = [], "不在資料中"
    if OOS_T in idx:
        O, Lw, C = px["Open"].values, px["Low"].values, px["Close"].values
        i = idx.get_loc(OOS_T)
        p_ref, day0 = C[i - 1], C[i] / C[i - 1] - 1
        rows, ts = [], None
        for k in range(1, MAIN["K"] + 1):
            if i + k >= len(C):
                break
            hit = C[i + k] >= p_ref
            rows.append([f"T+{k}", idx[i + k].date(), num(C[i + k]), pct(C[i + k] / p_ref - 1),
                         "是" if hit else "否"])
            if hit and ts is None:
                ts = i + k
                break
        oos_lines += [md_table(["項目", "數值"], [
            ["T", OOS_T.date()], ["P_ref = close(T-1)", num(p_ref)], ["close(T)", num(C[i])],
            ["day0", pct(day0)], ["符合事件條件 day0<0", "是" if day0 < 0 else "否"],
            ["資料最後一日", last.date()],
        ]), "", md_table(["日", "日期", "收盤", "相對 P_ref", "≥ P_ref"], rows), ""]
        if ts is None:
            done = i + MAIN["K"] < len(C)
            stop_now = float(Lw[i:].min())
            oos_status = ("10 日內未收復（觀察窗已結束）" if done else
                          f"尚未收復（已觀察 {len(C) - 1 - i} / {MAIN['K']} 日）")
            oos_lines.append(md_table(["項目", "數值"], [
                ["收復訊號", "未觸發"], ["事件低點（T ~ 最新）", num(stop_now)],
                ["狀態", oos_status]]))
        else:
            e = ts + 1
            stop = float(Lw[i:e].min())
            info = [["收復訊號", "觸發"], ["t_s", idx[ts].date()], ["k", ts - i],
                    ["事件低點停損（T ~ t_s）", num(stop)]]
            if e < len(C):
                entry = O[e]
                broke = [idx[j].date() for j in range(e, len(C)) if C[j] < stop]
                info += [["進場日 / 價 open(t_s+1)", f"{idx[e].date()} / {num(entry)}"],
                         [f"截至 {last.date()} 收盤報酬", pct(C[-1] / entry - 1)],
                         ["已持有交易日", len(C) - e],
                         ["是否已收盤跌破停損", f"是（{broke[0]}）" if broke else "否"]]
                oos_status = (f"{idx[ts].date()} 收復，{idx[e].date()} 開盤 {num(entry)} 進場，"
                              f"至 {last.date()} 報酬 {pct(C[-1] / entry - 1)}，"
                              + ("已觸及停損" if broke else "未觸及停損"))
            else:
                info.append(["進場", f"待 {idx[ts].date()} 次一交易日開盤"])
                oos_status = f"{idx[ts].date()} 收盤收復，待次一交易日開盤進場"
            oos_lines.append(md_table(["項目", "數值"], info))

    # ---- markdown
    L = []
    L.append("# TSMC 法說後「收復」進場回測（TSM）")
    L.append("")
    L.append("## 摘要")
    L.append("")
    dm = cmp_main
    p_txt = f"p = {dm['p']:.4f}" if dm["ok"] else f"p 值不報告（{INSUFFICIENT}）"
    ex_ci = f"[{pct(excess_ci[0])}, {pct(excess_ci[1])}]" if excess_ci else f"CI 不報告（{INSUFFICIENT}）"
    L.append(f"1. 樣本：納入 {len(in_sample)} 季法說，day0<0 事件 {len(neg)} 筆；收復組 {len(rec)}、未收復組 {len(unr)}（K=10）。")
    L.append(f"2. r20 平均：收復組 {pct(vals(rec, 'r20').mean() if len(rec) else float('nan'))} vs 未收復組 "
             f"{pct(vals(unr, 'r20').mean() if len(unr) else float('nan'))}，差 {pct(dm['diff'])}；{p_txt}。")
    L.append(f"3. 收復組 r20 相對無條件基準（{pct(bmean)}）超額 {pct(excess)}，95% CI {ex_ci}。")
    L.append(f"4. 推論狀態：" + (f"至少一組 n < {MIN_N} → **{INSUFFICIENT}**，以下僅為描述統計。"
                                if any_small else f"各組 n ≥ {MIN_N}，可報告推論統計（仍需注意多重比較）。"))
    L.append(f"5. 本次事件（{OOS_T.date()}，樣本外）：{oos_status}。")
    L.append("")
    L.append(f"- 產出時間：{now:%Y-%m-%d %H:%M:%S} UTC（台北 {now.astimezone(TPE):%Y-%m-%d %H:%M}）")
    L.append(f"- 價格：yfinance {yf.__version__}，{TICKER} 日線 auto_adjust=True，{idx[0].date()} ~ {last.date()}（{len(idx)} 筆）；"
             f"報酬與收復判定皆用同一組調整後 OHLC。")
    L.append(f"- 法說日期主來源：yfinance get_earnings_dates(limit=100)" + ("" if yahoo_ok else "（**抓取失敗**）")
             + "；缺漏季度嘗試 TSMC IR 季度頁面。")
    L.append("")

    L.append("## 定義（主規格，事先固定）")
    L.append("")
    L.append(md_table(["項目", "定義"], [
        ["事件條件", "day0 = close(T)/close(T-1) − 1 < 0"],
        ["參考價", "P_ref = close(T-1)"],
        ["收復訊號", "k = 1..10 中第一個 close(T+k) ≥ P_ref 的交易日 t_s"],
        ["進場", "收復組 open(t_s+1)；未收復組 open(T+11)"],
        ["持有日計數", "進場日為第 1 日；r_n = close(進場日 + n − 1) / 進場價 − 1"],
        ["maxDD20", "第 1~20 日收盤相對進場價最大跌幅（無下跌記 0）"],
        ["2x 模擬", f"TSMX_t = 2 × r_t − {FEE}/252 逐日複利；第 1 日 r = close/open − 1"],
        ["規則模擬停損", "T ~ 進場前一日的最低 Low（進場日 Low 於開盤時未知，故不納入以免前視）"],
        ["規則模擬出場", "收盤 < 停損 → 次一交易日開盤出場；否則第 20 日收盤時間到出場"],
        ["無條件基準", f"任意交易日 t+1 開盤進場、第 20 日收盤出場（期間自第一個納入事件 T 起；重疊窗）"],
        ["推論門檻", f"任一組 n < {MIN_N}：不報告 CI / p 值 / Cohen's d，結論欄寫「{INSUFFICIENT}」"],
        ["Bootstrap / 置換", f"各 {N_BOOT:,} / {N_PERM:,} 次，seed = {SEED}；置換檢定為雙尾"],
    ]))
    L.append("")

    L.append("## 1. 法說日期")
    L.append("")
    drows = []
    for yq in all_quarters():
        r = date_rows.get(yq)
        if r is None:
            continue
        prior = ""
        if r["T"].year >= 2023:
            prior = "一致" if str(r["T"].date()) in PRIOR_LIST else "**不在先前清單**"
        drows.append([r["q"], r["raw"], r["rule"], r["T"].date(), r["source"], status.get(yq, ""), prior])
    L.append(md_table(["季度", "原始時間戳 / 日期", "T 判定規則", "T", "來源", "是否納入", "與任務1清單"], drows))
    L.append("")
    L.append(f"### 排除清單（{len(exclusions)} 季）")
    L.append("")
    L.append(md_table(["季度", "原因"], exclusions) if exclusions else "無。")
    L.append("")

    L.append("## 2. 逐事件表（主規格）")
    L.append("")
    erows = []
    for e in all_events:
        grp = ("收復" if e["reclaim"] else "未收復") if e["day0"] < 0 else "day0≥0（非主樣本）"
        if e["day0"] < 0:
            erows.append([e["T"].date(), pct(e["day0"]), num(e["p_ref"]), grp, e["k"] or "—",
                          e["ts"].date() if e["ts"] is not None else "—",
                          e["entry_date"].date() if e["entry_date"] is not None else "N/A",
                          num(e["entry_px"]), pct(e["r5"]), pct(e["r10"]), pct(e["r20"]),
                          pct(e["dd20"]), pct(e["x2_20"])])
        else:
            erows.append([e["T"].date(), pct(e["day0"]), num(e["p_ref"]), grp] + ["—"] * 9)
    L.append(md_table(["T", "day0", "P_ref", "組別", "k", "t_s", "進場日", "進場價", "r5", "r10", "r20",
                       "maxDD20", "2x r20"], erows))
    L.append("")

    L.append("## 3. 主規格統計")
    L.append("")
    metrics = [("r5", "r5"), ("r10", "r10"), ("r20", "r20"), ("dd20", "maxDD20"), ("x2_20", "2x r20")]
    srows = (describe_rows("全部 day0<0", neg, metrics) + describe_rows("收復組", rec, metrics)
             + describe_rows("未收復組", unr, metrics))
    L.append(md_table(["組別", "指標", "n", "平均", "中位數", "勝率(>0)", "最差", "最佳", "平均 95% CI", "結論"], srows))
    L.append("")
    L.append("註：maxDD20 以 0 為上限，勝率恆為 0%。")
    L.append("")
    L.append("### 收復組 vs 未收復組（r20）")
    L.append("")
    L.append(md_table(["項目", "數值"], [
        ["n（收復 / 未收復）", f"{dm['na']} / {dm['nb']}"],
        ["r20 平均差", pct(dm["diff"])],
        ["置換檢定 p 值（雙尾）", f"{dm['p']:.4f}" if dm["ok"] else f"—（{INSUFFICIENT}）"],
        ["Cohen's d", f"{dm['d']:.3f}" if dm["ok"] else f"—（{INSUFFICIENT}）"],
    ]))
    L.append("")
    L.append("### 無條件基準與超額報酬")
    L.append("")
    L.append(md_table(["項目", "數值"], [
        ["基準期間", f"{in_sample[0].date() if in_sample else 'N/A'} 起，{len(bench):,} 個（重疊）20 日窗"],
        ["基準 r20 平均 / 中位數 / 勝率",
         f"{pct(bmean)} / {pct(float(np.median(bench)) if len(bench) else float('nan'))} / "
         f"{pct((bench > 0).mean() if len(bench) else float('nan'), 1)}"],
        ["收復組 r20 超額（平均 − 基準平均）", pct(excess)],
        ["超額 95% CI", ex_ci],
        ["未收復組 r20 超額", pct(vals(unr, 'r20').mean() - bmean if len(unr) else float('nan'))],
        ["全部 day0<0 r20 超額", pct(vals(neg, 'r20').mean() - bmean if len(neg) else float('nan'))],
    ]))
    L.append("")

    L.append("## 4. 規則模擬（事件低點停損 + 20 日時間停損）")
    L.append("")
    rrows = []
    for e in rec:
        rrows.append([e["T"].date(), e["entry_date"].date() if e["entry_date"] is not None else "N/A",
                      num(e["entry_px"]), num(e["stop"]), pct(e["stop"] / e["entry_px"] - 1),
                      e["exit_reason"] or "N/A", e["exit_date"].date() if e["exit_date"] is not None else "N/A",
                      e.get("rule_days", "N/A"), pct(e["rule_r"]), pct(e["rule_x2"])])
    L.append("### 收復組逐筆")
    L.append("")
    L.append(md_table(["T", "進場日", "進場價", "停損價", "停損距離", "出場原因", "出場日", "持有日",
                       "TSM 報酬", "2x 模擬報酬"], rrows) if rrows else "收復組無事件。")
    L.append("")
    sum_rows = []
    for name, grp in (("收復組", rec), ("未收復組（參考）", unr)):
        done = [e for e in grp if e["exit_reason"]]
        v, v1 = vals(done, "rule_x2"), vals(done, "rule_r")
        n = len(done)
        stops = sum(e["exit_reason"] == "停損" for e in done)
        sum_rows.append([name, n, f"{stops} / {n - stops}",
                         pct(stops / n, 1) if n else "N/A", pct(v1.mean() if n else float("nan")),
                         pct(v.mean() if n else float("nan")),
                         pct(float(np.median(v)) if n else float("nan")),
                         pct((v > 0).mean() if n else float("nan"), 1),
                         pct(v.min() if n else float("nan")), INSUFFICIENT if n < MIN_N else ""])
    L.append("### 彙總")
    L.append("")
    L.append(md_table(["組別", "n", "停損 / 時間到", "停損率", "TSM 平均", "2x 平均", "2x 中位數",
                       "2x 勝率", "2x 最差", "結論"], sum_rows))
    L.append("")

    L.append("## 5. 穩健性（只列表，不用於選參數）")
    L.append("")
    brows = []
    for label, override in [("主規格（K=10、門檻 1.00、open(t_s+1)、day0<0）", {})] + ROBUST:
        spec = {**MAIN, **override}
        evs = run_spec(px, in_sample, spec)
        rr = [e for e in evs if e["reclaim"]]
        uu = [e for e in evs if e["reclaim"] is False]
        c = compare(rr, uu)
        brows.append([label, len(evs), c["na"], c["nb"],
                      pct(vals(rr, "r20").mean() if c["na"] else float("nan")),
                      pct(vals(uu, "r20").mean() if c["nb"] else float("nan")), pct(c["diff"]),
                      f"{c['p']:.4f}" if c["ok"] else "—",
                      "" if c["ok"] else INSUFFICIENT])
    L.append(md_table(["變體", "事件數", "收復 n", "未收復 n", "收復 r20 平均", "未收復 r20 平均",
                       "差", "p 值", "結論"], brows))
    L.append("")
    L.append("註：進場 = close(t_s) 變體中，未收復組進場價為 close(T+K)，r_n 自進場次日起算。")
    L.append("")

    L.append(f"## 6. 本次事件（{OOS_T.date()}，樣本外，只描述）")
    L.append("")
    if oos:
        L.append(f"日期來源：{oos[0]['source']}（原始 {oos[0]['raw']}）")
    else:
        L.append("**注意：法說日期來源中未找到 T = 2026-10-15 的紀錄，以下依任務指定日期描述。**")
    L.append("")
    L += oos_lines
    L.append("")

    L.append("## 7. 判讀限制")
    L.append("")
    L.append(f"1. **樣本數**：day0<0 事件 {len(neg)} 筆，收復 / 未收復 {len(rec)} / {len(unr)}；"
             f"n < {MIN_N} 的組別只能描述，不得推論。季度事件本質上樣本稀少，單一極端季度即可主導平均。")
    L.append("2. **存活者偏誤**：TSM 是事後看來的長期贏家（2015 年至今大幅上漲），任何「持有 20 日」的報酬都"
             "帶有正趨勢；無條件基準只扣除平均趨勢，無法排除「選中這檔股票」本身的偏誤。")
    L.append("3. **2x 模擬 ≠ 實際 TSMX**：TSMX 2024-10 才上市，2x 序列以 TSM 日報酬模擬，未含追蹤誤差、"
             "交換合約成本、盤中再平衡、買賣價差、0.25% 複委託手續費與整股限制；費用率固定 0.99%。")
    L.append(f"4. **多重比較**：主規格之外列了 {len(ROBUST)} 個穩健性變體與多個結果變數，"
             "若從中挑出 p 值最小者會高估顯著性；本報告只以事先固定的主規格作結論。")
    L.append("5. **重疊與相依**：基準使用重疊 20 日窗，標準誤被低估；事件之間（尤其相鄰季度）亦非完全獨立。")
    L.append("6. **日期判定**：T 依 Yahoo 時間戳換算，盤中時間戳或缺漏季度已於日期表標註；換算錯誤會直接改變 day0 與組別。")
    L.append("")
    L.append("本報告僅為歷史資料的描述與檢定，不構成任何投資建議。")
    L.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    log(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    out = "reclaim_backtest.md"
    try:
        code = main(out)
    except Exception:  # noqa: BLE001
        tb = traceback.format_exc()
        print(tb, flush=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(f"# 法說後收復進場回測\n\n**執行失敗**（{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC）\n\n```\n{tb}```\n")
        code = 1
    sys.exit(code)
