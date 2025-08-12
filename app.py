# app.py — QuantaraX Pro (v5.1)
# ---------------------------------------------------------------------------------
# pip install: streamlit yfinance pandas numpy matplotlib feedparser vaderSentiment scikit-learn
# Optional: pip install hmmlearn

import math
import datetime as dt
from typing import List, Tuple, Dict, Optional

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import feedparser

# Optional ML imports
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.cluster import KMeans
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False
    TimeSeriesSplit = None
    KMeans = None

# Optional HMM
try:
    from hmmlearn.hmm import GaussianHMM
    HMM_OK = True
except Exception:
    HMM_OK = False
    GaussianHMM = None

# ───────────────────────────── Page Setup ─────────────────────────────
st.set_page_config(page_title="QuantaraX Pro v5.1", layout="wide")
analyzer = SentimentIntensityAnalyzer()
rec_map = {1: "🟢 BUY", 0: "🟡 HOLD", -1: "🔴 SELL"}

# Sample past FOMC dates
FOMC_DATES = [
    dt.date(2024, 1, 31), dt.date(2024, 3, 20), dt.date(2024, 5, 1),
    dt.date(2024, 6, 12), dt.date(2024, 7, 31), dt.date(2024, 9, 18),
    dt.date(2024,11, 7),  dt.date(2024,12, 18),
]

# ───────────────────────────── Helpers ─────────────────────────────
def _map_symbol(sym: str) -> str:
    s = sym.strip().upper()
    if "/" in s:
        base, quote = s.split("/")
        quote = "USD" if quote in ("USDT","USD") else quote
        return f"{base}-{quote}"
    return s

@st.cache_data(show_spinner=False, ttl=900)
def load_prices(symbol: str, period: str, interval: str) -> pd.DataFrame:
    sym = _map_symbol(symbol)
    df = yf.download(sym, period=period, interval=interval, auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df.dropna()

def compute_indicators(df: pd.DataFrame, ma_w: int, rsi_p: int, mf: int, ms: int, sig: int,
                       use_bb: bool = True) -> pd.DataFrame:
    d = df.copy()
    if d.empty or not set(["Open","High","Low","Close"]).issubset(d.columns):
        return pd.DataFrame()

    d[f"MA{ma_w}"] = d["Close"].rolling(ma_w).mean()

    chg = d["Close"].diff()
    up, dn = chg.clip(lower=0), -chg.clip(upper=0)
    ema_up   = up.ewm(com=rsi_p-1, adjust=False).mean()
    ema_down = dn.ewm(com=rsi_p-1, adjust=False).mean()
    rs = ema_up / ema_down.replace(0, np.nan)
    d[f"RSI{rsi_p}"] = 100 - 100 / (1 + rs)

    ema_f = d["Close"].ewm(span=mf, adjust=False).mean()
    ema_s = d["Close"].ewm(span=ms, adjust=False).mean()
    macd_line = ema_f - ema_s
    d["MACD"] = macd_line
    d["MACD_Signal"] = macd_line.ewm(span=sig, adjust=False).mean()

    pc = d["Close"].shift(1)
    tr = pd.concat([(d["High"]-d["Low"]).abs(), (d["High"]-pc).abs(), (d["Low"]-pc).abs()], axis=1).max(axis=1)
    d["TR"] = tr
    d["ATR"] = tr.ewm(alpha=1/14, adjust=False).mean()

    if use_bb:
        w = 20; k = 2.0
        mid = d["Close"].rolling(w).mean()
        sd  = d["Close"].rolling(w).std(ddof=0)
        d["BB_M"], d["BB_U"], d["BB_L"] = mid, mid + k*sd, mid - k*sd

    klen = 14
    ll = d["Low"].rolling(klen).min(); hh = d["High"].rolling(klen).max()
    d["STO_K"] = 100 * (d["Close"] - ll) / (hh - ll)
    d["STO_D"] = d["STO_K"].rolling(3).mean()

    adx_n = 14
    up_move = d["High"].diff()
    dn_move = -d["Low"].diff()
    plus_dm  = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
    tr_sm = tr.ewm(alpha=1/adx_n, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=d.index).ewm(alpha=1/adx_n, adjust=False).mean() / tr_sm
    minus_di= 100 * pd.Series(minus_dm, index=d.index).ewm(alpha=1/adx_n, adjust=False).mean() / tr_sm
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)).replace([np.inf,-np.inf], np.nan) * 100
    d["ADX"] = dx.ewm(alpha=1/adx_n, adjust=False).mean()

    dc_n = 20
    d["DC_U"] = d["High"].rolling(dc_n).max()
    d["DC_L"] = d["Low"].rolling(dc_n).min()

    kel_n = 20
    ema_mid = d["Close"].ewm(span=kel_n, adjust=False).mean()
    d["KC_U"] = ema_mid + 2 * d["ATR"]
    d["KC_L"] = ema_mid - 2 * d["ATR"]

    return d.dropna()

def build_composite(df: pd.DataFrame, ma_w: int, rsi_p: int,
                    *, use_weighted=True, w_ma=1.0, w_rsi=1.0, w_macd=1.0, w_bb=0.5,
                    include_bb=True, threshold=1.0, allow_short=False) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    d = df.copy()
    n = len(d)
    close = d["Close"].to_numpy()
    ma    = d[f"MA{ma_w}"].to_numpy()
    rsi   = d[f"RSI{rsi_p}"].to_numpy()
    macd  = d["MACD"].to_numpy()
    sigl  = d["MACD_Signal"].to_numpy()

    ma_sig = np.zeros(n, int)
    rsi_sig = np.zeros(n, int)
    macd_sig2 = np.zeros(n, int)
    bb_sig = np.zeros(n, int)

    for i in range(1, n):
        if close[i-1] < ma[i-1] and close[i] > ma[i]:   ma_sig[i] = 1
        elif close[i-1] > ma[i-1] and close[i] < ma[i]: ma_sig[i] = -1
        if rsi[i] < 30:   rsi_sig[i] = 1
        elif rsi[i] > 70: rsi_sig[i] = -1
        if macd[i-1] < sigl[i-1] and macd[i] > sigl[i]:   macd_sig2[i] = 1
        elif macd[i-1] > sigl[i-1] and macd[i] < sigl[i]: macd_sig2[i] = -1
        if include_bb and {"BB_U","BB_L"}.issubset(d.columns):
            if close[i] < d["BB_L"].iloc[i]: bb_sig[i] = 1
            elif close[i] > d["BB_U"].iloc[i]: bb_sig[i] = -1

    comp = (w_ma*ma_sig + w_rsi*rsi_sig + w_macd*macd_sig2 + (w_bb*bb_sig if include_bb else 0)) if use_weighted \
           else (ma_sig + rsi_sig + macd_sig2)

    vol = (d["ATR"] / d["Close"]).clip(lower=1e-9)
    z = (vol - vol.rolling(100).mean()) / (vol.rolling(100).std() + 1e-9)
    z = z.clip(-1, 1).fillna(0)
    thr_adj = threshold * (1.0 + 0.25*z.values)
    size = np.clip(np.abs(comp) / 3.0, 0.0, 1.0)

    if allow_short:
        trade = np.where(comp >= thr_adj, 1, np.where(comp <= -thr_adj, -1, 0))
    else:
        trade = np.where(comp >= thr_adj, 1, 0)

    d["MA_Signal"], d["RSI_Signal"], d["MACD_Signal2"] = ma_sig, rsi_sig, macd_sig2
    if include_bb: d["BB_Signal"] = bb_sig
    d["Composite"] = comp.astype(float)
    d["ThrAdj"] = thr_adj.astype(float)
    d["Size"] = size.astype(float)
    d["Trade"] = trade.astype(int)
    return d

def get_earnings_dates(symbol: str, limit: int = 12) -> List[dt.date]:
    try:
        tk = yf.Ticker(_map_symbol(symbol))
        df = tk.get_earnings_dates(limit=limit)
        if df is None or df.empty:
            return []
        out = []
        for idx in df.index:
            if isinstance(idx, (pd.Timestamp, np.datetime64)):
                out.append(pd.Timestamp(idx).date())
            elif isinstance(idx, str):
                try: out.append(pd.to_datetime(idx).date())
                except: pass
        return sorted(list(set(out)))
    except Exception:
        return []

def apply_event_flatten(df: pd.DataFrame, dates: List[dt.date],
                        days_before: int = 1, days_after: int = 0) -> pd.DataFrame:
    if df.empty or not dates:
        df2 = df.copy()
        df2["EventMask"] = 0
        return df2
    d = df.copy()
    mask = pd.Series(0, index=d.index)
    idx_dates = [i.date() if isinstance(i, (pd.Timestamp, pd.DatetimeIndex)) else pd.Timestamp(i).date() for i in d.index]
    idx_series = pd.Series(idx_dates, index=d.index)
    date_set = set()
    for ed in dates:
        for k in range(-days_before, days_after+1):
            date_set.add(ed + dt.timedelta(days=k))
    mask[:] = idx_series.isin(date_set).astype(int).values
    d["EventMask"] = mask
    if "Trade" in d.columns:
        d.loc[d["EventMask"]==1, "Trade"] = 0
    return d

def options_lens(symbol: str) -> Optional[Dict[str, float]]:
    try:
        tk = yf.Ticker(_map_symbol(symbol))
        exps = tk.options
        if not exps:
            return None
        exp = exps[0]
        ch = tk.option_chain(exp)
        calls, puts = ch.calls.copy(), ch.puts.copy()
        spot = tk.history(period="5d", auto_adjust=True)["Close"].iloc[-1]
        k_atm = min(calls["strike"], key=lambda k: abs(k-spot))
        ivc = float(calls.loc[calls["strike"]==k_atm, "impliedVolatility"].head(1).fillna(np.nan).values[0])
        ivp = float(puts.loc[puts["strike"]==k_atm, "impliedVolatility"].head(1).fillna(np.nan).values[0])
        iv_atm = np.nanmean([ivc, ivp])
        k_put = 0.90*spot
        k_call = 1.10*spot
        p_iv = puts.iloc[(puts["strike"]-k_put).abs().argsort()[:1]]["impliedVolatility"].mean()
        c_iv = calls.iloc[(calls["strike"]-k_call).abs().argsort()[:1]]["impliedVolatility"].mean()
        skew = float(p_iv - c_iv)
        ivs = pd.concat([calls["impliedVolatility"], puts["impliedVolatility"]], axis=0).dropna()
        iv_rank_chain = float((ivs[ivs < iv_atm].count() / max(len(ivs),1)))
        return {"spot": float(spot), "iv_atm": float(iv_atm), "skew": float(skew), "iv_rank_chain": iv_rank_chain}
    except Exception:
        return None

def _stats_from_equity(d: pd.DataFrame, interval: str) -> Tuple[float,float,float,float,int,float,float]:
    ann = 252 if interval == "1d" else 252*6
    if d["CumStrat"].notna().any():
        dd = d["CumStrat"]/d["CumStrat"].cummax() - 1
        max_dd = float(dd.min()*100)
        last_cum = float(d["CumStrat"].dropna().iloc[-1])
    else:
        max_dd = 0.0; last_cum = 1.0
    mean_ann = float(d["StratRet"].mean() * ann)
    vol_ann  = float(d["StratRet"].std(ddof=0) * math.sqrt(ann))
    sharpe   = (mean_ann / vol_ann) if vol_ann > 0 else np.nan
    win_rt   = float((d["StratRet"] > 0).mean() * 100)
    pos_change = d["Position"].diff().fillna(0).abs()
    trades   = int((pos_change > 0).sum())
    tim      = float((d["Position"] != 0).mean() * 100)
    n_eff    = int(d["StratRet"].notna().sum())
    cagr     = ((last_cum ** (ann / max(n_eff, 1))) - 1) * 100 if n_eff > 0 else np.nan
    return max_dd, sharpe, win_rt, trades, tim, cagr, last_cum

def backtest(df: pd.DataFrame, *, allow_short=False, cost_bps=0.0,
             sl_atr_mult=0.0, tp_atr_mult=0.0, vol_target=0.0, interval="1d"):
    d = df.copy()
    if d.empty or "Close" not in d:
        sk = d.copy()
        for col in ["Return","Position","StratRet","CumBH","CumStrat"]:
            sk[col] = 0.0
        sk["CumBH"] = 1.0
        sk["CumStrat"] = 1.0
        return sk, 0.0, np.nan, np.nan, 0, 0.0, np.nan

    d["Return"] = d["Close"].pct_change().fillna(0.0)

    if allow_short:
        d["Position"] = d.get("Trade", 0).shift(1).fillna(0).clip(-1,1)
        base_ret = np.where(d["Position"]>=0, d["Return"], -d["Return"])
    else:
        d["Position"] = d.get("Trade", 0).shift(1).fillna(0).clip(0,1)
        base_ret = d["Position"] * d["Return"]

    if vol_target and vol_target > 0:
        look = 20
        daily_vol = d["Return"].rolling(look).std(ddof=0)
        ann = 252 if interval == "1d" else 252*6
        realized = daily_vol * math.sqrt(ann)
        scale = (vol_target / realized).clip(0, 3.0).fillna(0.0)
        base_ret = base_ret * scale

    cost = cost_bps/10000.0
    pos_change = d["Position"].diff().fillna(0).abs()
    tcost = -2.0*cost*(pos_change > 0).astype(float)
    d["StratRet"] = pd.Series(base_ret, index=d.index).fillna(0.0) + tcost

    if (sl_atr_mult>0 or tp_atr_mult>0) and "ATR" in d.columns:
        flat = np.zeros(len(d), dtype=int)
        entry = np.nan
        for i in range(len(d)):
            p, c = d["Position"].iat[i], d["Close"].iat[i]
            a = d["ATR"].iat[i] if "ATR" in d.columns else np.nan
            if p != 0 and np.isnan(entry): entry = c
            if p == 0: entry = np.nan
            if p != 0 and not np.isnan(a):
                if p == 1 and (c <= entry - sl_atr_mult*a or c >= entry + tp_atr_mult*a):
                    flat[i] = 1; entry = np.nan
                if p == -1 and (c >= entry + sl_atr_mult*a or c <= entry - tp_atr_mult*a):
                    flat[i] = 1; entry = np.nan
        if flat.any(): d.loc[flat==1, "Position"] = 0

    ret_bh = d["Return"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ret_st = d["StratRet"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    d["CumBH"] = (1 + ret_bh).cumprod()
    d["CumStrat"] = (1 + ret_st).cumprod()

    max_dd, sharpe, win_rt, trades, tim, cagr, last_cum = _stats_from_equity(d, interval)
    return d, max_dd, sharpe, win_rt, trades, tim, cagr

# ───────────────────────────── Sidebar & Tabs ─────────────────────────────
st.sidebar.header("Global Controls")

DEFAULTS = dict(ma_window=10, rsi_period=14, macd_fast=12, macd_slow=26, macd_signal=9)
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.sidebar.button("🔄 Reset to defaults", key="btn_reset_defaults"):
    for k, v in DEFAULTS.items():
        st.session_state[k] = v

# Indicator params
st.sidebar.subheader("Indicators")
ma_window   = st.sidebar.slider("MA window",      5, 60, st.session_state["ma_window"],   key="sl_ma_window")
rsi_period  = st.sidebar.slider("RSI lookback",   5, 30, st.session_state["rsi_period"],  key="sl_rsi_period")
macd_fast   = st.sidebar.slider("MACD fast span", 5, 20, st.session_state["macd_fast"],   key="sl_macd_fast")
macd_slow   = st.sidebar.slider("MACD slow span", 20, 50, st.session_state["macd_slow"],  key="sl_macd_slow")
macd_signal = st.sidebar.slider("MACD sig span",  5, 20, st.session_state["macd_signal"], key="sl_macd_signal")

# Composite weights
st.sidebar.subheader("Composite (Adaptive v2)")
use_weighted = st.sidebar.toggle("Use weighted composite", value=True, key="tg_use_weighted")
include_bb   = st.sidebar.toggle("Include Bollinger Bands", value=True, key="tg_include_bb")
w_ma   = st.sidebar.slider("Weight • MA",   0.0, 2.0, 1.0, 0.1, key="sl_w_ma")
w_rsi  = st.sidebar.slider("Weight • RSI",  0.0, 2.0, 1.0, 0.1, key="sl_w_rsi")
w_macd = st.sidebar.slider("Weight • MACD", 0.0, 2.0, 1.0, 0.1, key="sl_w_macd")
w_bb   = st.sidebar.slider("Weight • BB",   0.0, 2.0, 0.5, 0.1, key="sl_w_bb") if include_bb else 0.0
comp_thr = st.sidebar.slider("Composite trigger (base)", 0.0, 3.0, 1.0, 0.1, key="sl_comp_thr")

# Risk & costs
st.sidebar.subheader("Risk & Costs")
allow_short = st.sidebar.toggle("Allow shorts", value=False, key="tg_allow_short")
cost_bps    = st.sidebar.slider("Trading cost (bps/side)", 0.0, 25.0, 5.0, 0.5, key="sl_cost_bps")
sl_atr_mult = st.sidebar.slider("Stop • ATR ×", 0.0, 5.0, 2.0, 0.1, key="sl_sl_atr_mult")
tp_atr_mult = st.sidebar.slider("Target • ATR ×", 0.0, 8.0, 3.0, 0.1, key="sl_tp_atr_mult")
vol_target  = st.sidebar.slider("Vol targeting (annual)", 0.0, 0.5, 0.0, 0.05, key="sl_vol_target")

# Data
st.sidebar.subheader("Data")
period_sel   = st.sidebar.selectbox("History", ["6mo","1y","2y","5y"], index=1, key="sb_period_sel")
interval_sel = st.sidebar.selectbox("Interval", ["1d","1h"], index=0, key="sb_interval_sel")

# Event guards
st.sidebar.subheader("Event Guards")
guard_earn  = st.sidebar.toggle("Flatten around earnings", value=True, key="tg_guard_earn")
earn_bef    = st.sidebar.slider("Days before earnings (flat)", 0, 5, 1, key="sl_earn_bef")
earn_aft    = st.sidebar.slider("Days after earnings (flat)",  0, 5, 0, key="sl_earn_aft")
guard_fomc  = st.sidebar.toggle("Flatten around FOMC (sample dates)", value=False, key="tg_guard_fomc")

# Options lens
st.sidebar.subheader("Options Lens")
show_options = st.sidebar.toggle("Show ATM IV / Skew / IV-rank", value=False, key="tg_show_options")

# Portfolio guardrails
st.sidebar.subheader("Portfolio Guardrails")
profit_target = st.sidebar.slider("Profit target (%)", 1, 100, 10, key="sl_profit_target")
loss_limit    = st.sidebar.slider("Loss limit (%)",  1, 100, 5, key="sl_loss_limit")

# Tabs
(tab_engine, tab_ml, tab_scan, tab_regime, tab_port, tab_help) = st.tabs(
    ["🚀 Engine","🧠 ML Lab","📡 Scanner","📉 Regimes","💼 Portfolio","❓ Help"]
)

# ───────────────────────────── ENGINE ─────────────────────────────
with tab_engine:
    st.title("🚀 QuantaraX — Composite Signal Engine (v5.1)")

    ticker = st.text_input("Ticker (e.g. AAPL or BTC/USDT)", "AAPL", key="ti_engine_ticker").upper().strip()
    if ticker:
        try:
            h = yf.download(_map_symbol(ticker), period="1d", auto_adjust=True, progress=False)
            if not h.empty and "Close" in h:
                val = h["Close"].iloc[-1]
                price = float(val.item() if hasattr(val, "item") else val)
                st.subheader(f"💲 Live Price: ${price:,.2f}")
        except Exception:
            pass

        st.markdown("#### 📰 Recent News (Yahoo Finance RSS)")
        try:
            rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={_map_symbol(ticker)}&region=US&lang=en-US"
            feed = feedparser.parse(rss_url)
            shown = 0
            for entry in feed.entries[:5]:
                title = entry.get("title",""); link = entry.get("link","")
                if not (title and link): continue
                score = analyzer.polarity_scores(title).get("compound", 0.0)
                emoji = "🔺" if score>0.1 else ("🔻" if score<-0.1 else "➖")
                st.markdown(f"- [{title}]({link}) {emoji}")
                shown += 1
            if shown == 0:
                st.info("No recent items found.")
        except Exception:
            st.info("News unavailable.")

    run_btn = st.button("▶️ Run Composite Backtest", key="btn_run_backtest")

    if run_btn:
        px = load_prices(ticker, period_sel, interval_sel)
        if px.empty:
            st.error(f"No data for '{ticker}'"); st.stop()

        df_raw = compute_indicators(px, ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=include_bb)
        if df_raw.empty:
            st.error("Not enough data after indicators (try longer period or smaller windows)."); st.stop()

        df_sig = build_composite(
            df_raw, ma_window, rsi_period,
            use_weighted=use_weighted, w_ma=w_ma, w_rsi=w_rsi, w_macd=w_macd, w_bb=w_bb,
            include_bb=include_bb, threshold=comp_thr, allow_short=allow_short
        )

        if guard_earn:
            edates = get_earnings_dates(ticker, limit=12)
            df_sig = apply_event_flatten(df_sig, edates, days_before=int(earn_bef), days_after=int(earn_aft))
        if guard_fomc:
            df_sig = apply_event_flatten(df_sig, FOMC_DATES, days_before=1, days_after=0)

        df_c, max_dd, sharpe, win_rt, trades, tim, cagr = backtest(
            df_sig, allow_short=allow_short, cost_bps=cost_bps,
            sl_atr_mult=sl_atr_mult, tp_atr_mult=tp_atr_mult,
            vol_target=vol_target, interval=interval_sel
        )

        last_trade = int(df_sig["Trade"].iloc[-1]) if "Trade" in df_sig.columns and not df_sig.empty else 0
        rec = rec_map.get(1 if last_trade>0 else (-1 if last_trade<0 else 0), "🟡 HOLD")
        st.success(f"**{ticker}**: {rec}")

        last = df_sig.iloc[-1]
        ma_s  = int(last.get("MA_Signal", 0)); rsi_s = int(last.get("RSI_Signal", 0)); macd_s = int(last.get("MACD_Signal2", 0))
        rsi_v = float(last.get(f"RSI{rsi_period}", np.nan))
        ma_txt  = {1:f"Price ↑ crossed above MA{ma_window}.", 0:"No MA crossover.", -1:f"Price ↓ crossed below MA{ma_window}."}.get(ma_s, "No MA crossover.")
        rsi_txt = "RSI data unavailable." if np.isnan(rsi_v) else {
            1:f"RSI ({rsi_v:.1f}) < 30 → oversold.",
            0:f"RSI ({rsi_v:.1f}) neutral.",
           -1:f"RSI ({rsi_v:.1f}) > 70 → overbought."
        }.get(rsi_s, f"RSI ({rsi_v:.1f}) neutral.")
        macd_txt= {1:"MACD ↑ crossed above signal.", 0:"No MACD crossover.", -1:"MACD ↓ crossed below signal."}.get(macd_s, "No MACD crossover.")

        with st.expander("🔎 Why This Signal?"):
            st.write(f"- **MA:**  {ma_txt}")
            st.write(f"- **RSI:** {rsi_txt}")
            st.write(f"- **MACD:** {macd_txt}")
            if include_bb and "BB_Signal" in df_sig.columns:
                bb_s = int(last.get("BB_Signal", 0))
                bb_txt = {1:"Close under lower band (mean-revert long).",0:"Inside bands.",-1:"Close over upper band (mean-revert short)."}[bb_s]
                st.write(f"- **BB:** {bb_txt}")
            st.write(f"- **Composite (raw):** {float(last.get('Composite', 0)):.2f}")
            st.write(f"- **Adaptive Threshold now:** {float(last.get('ThrAdj', comp_thr)):.2f}")
            st.write(f"- **Position Size now:** {float(last.get('Size', 1.0)):.2f}")

        if show_options:
            snap = options_lens(ticker)
            with st.expander("🧩 Options Lens"):
                if snap:
                    st.write(f"**Spot:** {snap['spot']:.2f}")
                    st.write(f"**ATM IV:** {snap['iv_atm']*100:.1f}%")
                    st.write(f"**Skew (0.90P - 1.10C IV):** {snap['skew']*100:.1f}%")
                    st.write(f"**IV-rank (within current chain):** {snap['iv_rank_chain']*100:.1f}%")
                else:
                    st.info("No options snapshot available.")

        bh_last    = float(df_c["CumBH"].iloc[-1]) if "CumBH" in df_c else 1.0
        strat_last = float(df_c["CumStrat"].iloc[-1]) if "CumStrat" in df_c else 1.0
        cA, cB, cC, cD, cE, cF = st.columns(6)
        cA.metric("CAGR", f"{(0 if np.isnan(cagr) else cagr):.2f}%")
        cB.metric("Sharpe", f"{(0 if np.isnan(sharpe) else sharpe):.2f}")
        cC.metric("Max DD", f"{max_dd:.2f}%")
        cD.metric("Win Rate", f"{win_rt:.1f}%")
        cE.metric("Trades", f"{trades}")
        cF.metric("Time in Mkt", f"{tim:.1f}%")

        st.markdown(f"""
- **Buy & Hold:**    {(bh_last-1)*100:.2f}%  
- **Strategy:**      {(strat_last-1)*100:.2f}%  
""")

        idx = df_c.index
        fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(11,12), sharex=True)
        ax1.plot(idx, df_c["Close"], label="Close")
        if f"MA{ma_window}" in df_c: ax1.plot(idx, df_c[f"MA{ma_window}"], label=f"MA{ma_window}")
        if include_bb and {"BB_U","BB_L"}.issubset(df_c.columns):
            ax1.plot(idx, df_c["BB_U"], label="BB Upper"); ax1.plot(idx, df_c["BB_L"], label="BB Lower")
        if "EventMask" in df_c:
            mask = df_c["EventMask"]==1
            ax1.fill_between(idx, df_c["Close"].min(), df_c["Close"].max(), where=mask, alpha=0.08, label="Event flat")
        ax1.legend(); ax1.set_title("Price & Indicators")
        if "Composite" in df_c:
            ax2.bar(idx, df_c["Composite"]); ax2.set_title("Composite (raw)")
        else:
            ax2.set_title("Composite (no data)")
        ax3.plot(idx, df_c["CumBH"], ":", label="BH")
        ax3.plot(idx, df_c["CumStrat"], "-", label="Strat"); ax3.legend(); ax3.set_title("Equity")
        plt.xticks(rotation=45); plt.tight_layout()
        st.pyplot(fig)

# ───────────────────────────── ML LAB ─────────────────────────────
with tab_ml:
    st.title("🧠 ML Lab — Purged K-Fold, OOS Probabilities")
    if not SKLEARN_OK:
        st.warning("scikit-learn not installed. Run: pip install scikit-learn")

    symbol = st.text_input("Symbol (ML)", value="AAPL", key="ti_ml_symbol").upper().strip()
    horizon = st.slider("Prediction horizon (bars)", 1, 5, 1, key="sl_ml_horizon")
    n_splits = st.slider("TimeSeries splits", 3, 8, 5, key="sl_ml_splits")
    gap = st.slider("Purging gap (bars)", 0, 20, 5, key="sl_ml_gap")
    proba_enter = st.slider("Enter if P(long) ≥", 0.50, 0.90, 0.60, 0.01, key="sl_ml_enter")
    proba_exit  = st.slider("Enter short if P(long) ≤", 0.10, 0.50, 0.40, 0.01, key="sl_ml_exit")
    run_ml = st.button("🤖 Train & Backtest (Purged CV)", key="btn_ml_run")

    def _ml_features(d: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=d.index)
        out["ret1"] = d["Close"].pct_change()
        out["ret5"] = d["Close"].pct_change(5)
        out["vol20"] = d["Close"].pct_change().rolling(20).std()
        out["rsi"] = d.get(f"RSI{rsi_period}", np.nan)
        out["macd"] = d.get("MACD", np.nan)
        out["sto_k"] = d.get("STO_K", np.nan)
        out["adx"] = d.get("ADX", np.nan)
        if {"BB_U","BB_L"}.issubset(d.columns):
            rng = (d["BB_U"] - d["BB_L"]).replace(0, np.nan)
            out["bb_pos"] = (d["Close"] - d["BB_L"]) / rng
        else:
            out["bb_pos"] = np.nan
        return out.dropna()

    if run_ml and SKLEARN_OK and TimeSeriesSplit is not None:
        try:
            px = load_prices(symbol, period_sel, interval_sel)
            ind = compute_indicators(px, ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=True)
            if ind.empty: st.error("Not enough data for indicators."); st.stop()

            X = _ml_features(ind)
            y = (ind["Close"].pct_change(horizon).shift(-horizon) > 0).reindex(X.index).astype(int)
            data = pd.concat([X, y.rename("y")], axis=1).dropna()
            if len(data) < 400:
                st.warning("Not enough rows for ML. Try longer history or daily interval."); st.stop()

            tscv = TimeSeriesSplit(n_splits=int(n_splits))
            oof_proba = pd.Series(index=data.index, dtype=float)
            models = []

            for fold, (tr, te) in enumerate(tscv.split(data)):
                if gap > 0:
                    te_start = te[0]
                    tr = tr[tr < te_start - gap]
                    te = te[gap:] if len(te) > gap else te

                Xtr, ytr = data.iloc[tr].drop(columns=["y"]), data.iloc[tr]["y"]
                Xte, yte = data.iloc[te].drop(columns=["y"]), data.iloc[te]["y"]

                clf = RandomForestClassifier(n_estimators=400, max_depth=6, random_state=42, n_jobs=-1)
                clf.fit(Xtr, ytr)
                p = clf.predict_proba(Xte)[:,1]
                oof_proba.iloc[te] = p
                models.append(clf)

            y_true = data["y"].reindex(oof_proba.index).values
            mask = ~oof_proba.isna()
            acc = accuracy_score(y_true[mask], (oof_proba[mask]>0.5).astype(int))
            try:
                auc = roc_auc_score(y_true[mask], oof_proba[mask])
            except Exception:
                auc = np.nan

            c1,c2 = st.columns(2)
            c1.metric("OOF Accuracy (0.5)", f"{acc*100:.1f}%")
            c2.metric("OOF ROC-AUC", f"{(0 if np.isnan(auc) else auc):.3f}")

            sig_idx = oof_proba.dropna().index
            sig = pd.Series(0, index=sig_idx)
            if allow_short:
                sig[:] = np.where(oof_proba.loc[sig_idx] >= proba_enter, 1,
                                  np.where(oof_proba.loc[sig_idx] <= proba_exit, -1, 0))
            else:
                sig[:] = np.where(oof_proba.loc[sig_idx] >= proba_enter, 1, 0)

            ml_df = ind.loc[sig_idx].copy()
            ml_df["Trade"] = sig.astype(int)
            ml_df["Size"] = (oof_proba.loc[sig_idx] - 0.5).abs()*2.0
            ml_df["Size"] = ml_df["Size"].clip(0,1).fillna(0.0)

            bt, md, sh, wr, trd, tim, cagr = backtest(
                ml_df, allow_short=allow_short, cost_bps=cost_bps,
                sl_atr_mult=sl_atr_mult, tp_atr_mult=tp_atr_mult,
                vol_target=vol_target, interval=interval_sel
            )
            st.markdown(f"**ML OOS (OOF) Strat:** Return={(bt['CumStrat'].iloc[-1]-1)*100:.2f}% | Sharpe={sh:.2f} | MaxDD={md:.2f}% | Trades={trd}")
            fig, ax = plt.subplots(figsize=(9,3))
            ax.plot(bt.index, bt["CumBH"], ":", label="BH"); ax.plot(bt.index, bt["CumStrat"], label="ML Strat"); ax.legend(); ax.set_title("ML OOS Equity (OOF)")
            st.pyplot(fig)

            try:
                pim = permutation_importance(models[-1], Xte, yte, n_repeats=5, random_state=42)
                imp = pd.Series(pim.importances_mean, index=Xte.columns).sort_values(ascending=False)
                st.bar_chart(imp)
            except Exception:
                st.info("Permutation importance unavailable.")
        except Exception as e:
            st.error(f"ML error: {e}")
    elif run_ml and (not SKLEARN_OK or TimeSeriesSplit is None):
        st.error("scikit-learn not available.")

# ───────────────────────────── SCANNER ─────────────────────────────
with tab_scan:
    st.title("📡 Universe Scanner — Composite + Options Snapshot")
    universe = st.text_area(
        "Tickers (comma-separated)",
        "AAPL, MSFT, NVDA, TSLA, AMZN, GOOGL, META, NFLX, SPY, QQQ",
        key="ta_scan_universe"
    ).upper()

    run_scan = st.button("🔎 Scan", key="btn_scan")

    if run_scan:
        rows=[]
        tickers = [t.strip() for t in universe.split(",") if t.strip()]
        for t in tickers:
            try:
                px = load_prices(t, period_sel, interval_sel)
                if px.empty: continue
                ind = compute_indicators(px, ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=True)
                sig = build_composite(ind, ma_window, rsi_period,
                                      use_weighted=use_weighted, w_ma=w_ma, w_rsi=w_rsi, w_macd=w_macd, w_bb=w_bb,
                                      include_bb=include_bb, threshold=comp_thr, allow_short=allow_short)
                if sig.empty: continue
                if guard_earn:
                    ed = get_earnings_dates(t, limit=8)
                    sig = apply_event_flatten(sig, ed, days_before=int(earn_bef), days_after=int(earn_aft))
                if guard_fomc:
                    sig = apply_event_flatten(sig, FOMC_DATES, days_before=1, days_after=0)

                comp = float(sig["Composite"].iloc[-1]) if "Composite" in sig else 0.0
                thr  = float(sig["ThrAdj"].iloc[-1]) if "ThrAdj" in sig else comp_thr
                rec  = rec_map.get(int(np.sign(comp)), "🟡 HOLD")

                o = options_lens(t) if show_options else None
                iv_rank = (o["iv_rank_chain"]*100) if (o and "iv_rank_chain" in o) else np.nan

                ndays = np.nan
                if guard_earn:
                    eds = get_earnings_dates(t, limit=4)
                    future = [d for d in eds if d >= dt.date.today()]
                    if future:
                        ndays = (future[0] - dt.date.today()).days

                rows.append({
                    "Ticker": t,
                    "Composite": comp,
                    "Adj Thr": thr,
                    "Signal": rec,
                    "IV Rank%": iv_rank,
                    "Days→Earnings": ndays
                })
            except Exception:
                continue

        if rows:
            df = pd.DataFrame(rows).set_index("Ticker").sort_values(["Signal","Composite"], ascending=[True,False])
            st.dataframe(df, use_container_width=True)
            st.download_button("Download CSV", df.to_csv(), "scan.csv", key="dl_scan_csv")
        else:
            st.info("No results. Check tickers or increase history.")

# ───────────────────────────── REGIMES ─────────────────────────────
with tab_regime:
    st.title("📉 Regime Detection — HMM (optional) / KMeans fallback")
    sym = st.text_input("Symbol (Regime)", value="SPY", key="ti_regime_sym").upper()
    n_states = st.slider("Regime states", 2, 5, 3, key="sl_regime_states")
    run_rg = st.button("Cluster Regimes", key="btn_regimes")

    if run_rg:
        try:
            px = load_prices(sym, "3y", "1d")
            ind = compute_indicators(px, ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=False)
            if ind.empty: st.error("Not enough data."); st.stop()

            feat = pd.DataFrame(index=ind.index)
            feat["vol20"] = ind["Close"].pct_change().rolling(20).std()
            feat["mom20"] = ind["Close"].pct_change(20)
            feat["ma_slope"] = ind[f"MA{ma_window}"].diff()
            feat = feat.dropna()

            if HMM_OK:
                X = feat.values
                hmm = GaussianHMM(n_components=int(n_states), covariance_type="full", n_iter=200, random_state=42)
                hmm.fit(X)
                lab = hmm.predict(X)
            elif SKLEARN_OK and KMeans is not None:
                km = KMeans(n_clusters=int(n_states), n_init=10, random_state=42)
                lab = km.fit_predict(feat)
            else:
                q1 = feat.rank(pct=True)
                lab = (q1.mean(axis=1) > 0.66).astype(int) + (q1.mean(axis=1) < 0.33).astype(int)*2

            reg = pd.Series(lab, index=feat.index, name="Regime")
            joined = ind.join(reg, how="right")

            fwd = joined["Close"].pct_change().shift(-5)
            ret = fwd.groupby(joined["Regime"]).mean().sort_values()
            ord_map = {old:i for i, old in enumerate(ret.index)}
            joined["Regime"] = joined["Regime"].map(ord_map)

            st.dataframe(joined[["Close","Regime"]].tail(10))

            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(joined.index, joined["Close"], label="Close")
            for r in sorted(joined["Regime"].dropna().unique()):
                seg = joined[joined["Regime"]==r]
                ax.fill_between(seg.index, seg["Close"].min(), seg["Close"].max(), alpha=0.08, label=f"Regime {int(r)}")
            ax.set_title("Price with Regime Shading")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Regime error: {e}")

# ───────────────────────────── PORTFOLIO ─────────────────────────────
with tab_port:
    st.title("💼 Portfolio — Optimizers, Studio & Monte Carlo")

    st.subheader("⚖️ Risk Parity Optimizer")
    opt_tickers = st.text_input("Tickers (comma-sep)", "AAPL, MSFT, TSLA, SPY, QQQ", key="ti_opt_tickers").upper()
    if st.button("🧮 Optimize (Risk Parity)", key="btn_rp"):
        try:
            tickers = [t.strip() for t in opt_tickers.split(",") if t.strip()]
            rets = []; valid = []
            for t in tickers:
                px = load_prices(t, "1y", "1d")
                if px.empty: continue
                valid.append(t)
                rets.append(px["Close"].pct_change().dropna())
            if not rets:
                st.error("No valid tickers/data."); st.stop()
            R = pd.concat(rets, axis=1); R.columns = valid
            cov = R.cov()
            n = len(valid); w = np.ones(n)/n
            for _ in range(500):
                mrc = cov @ w
                rc  = w * mrc
                target = rc.mean()
                grad = rc - target
                w = np.clip(w - 0.05*grad, 0, None)
                s = w.sum()
                w = w / s if s > 1e-12 else np.ones(n)/n
                if np.linalg.norm(grad) < 1e-6:
                    break
            weights = pd.Series(w, index=valid, name="Weight")
            st.dataframe(weights.to_frame().T, use_container_width=True)
            fig, ax = plt.subplots(figsize=(5,5))
            weights.plot.pie(autopct="%.1f%%", ax=ax)
            ax.set_ylabel(""); ax.set_title("Risk-Parity Weights")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Optimizer error: {e}")

    st.subheader("🧭 Black–Litterman (simple views)")
    bl_tickers = st.text_input("Tickers (BL)", opt_tickers, key="ti_bl_tickers").upper()
    views_txt = st.text_area(
        "Views (one per line: e.g. 'AAPL: +2' meaning +2% vs equilibrium; or 'NVDA: -1')",
        "AAPL: +2\nMSFT: +1", key="ta_bl_views"
    )
    if st.button("🧠 Compute BL Weights", key="btn_bl"):
        try:
            tickers = [t.strip() for t in bl_tickers.split(",") if t.strip()]
            if len(tickers) < 2:
                st.error("Need 2+ tickers."); st.stop()
            R = []
            cols = []
            for t in tickers:
                px = load_prices(t, "2y", "1d")
                if px.empty: continue
                cols.append(t)
                R.append(px["Close"].pct_change().dropna())
            if not R:
                st.error("No returns for BL."); st.stop()
            R = pd.concat(R, axis=1); R.columns = cols

            mu = R.mean()*252
            Sigma = R.cov()*252
            n = len(Sigma)
            w_mkt = np.ones(n)/n
            delta = 3.0
            tau = 0.05
            Pi = delta * Sigma.values @ w_mkt

            P = []; Q = []
            for line in views_txt.splitlines():
                if ":" not in line: continue
                k, v = line.split(":", 1)
                k = k.strip().upper(); v = v.strip().replace("%","")
                try:
                    q = float(v)
                except:
                    continue
                row = np.zeros(n)
                if k in R.columns:
                    row[list(R.columns).index(k)] = 1.0
                else:
                    continue
                P.append(row); Q.append(q/100.0)
            if not P:
                st.info("No valid views parsed. Using equilibrium.")
                P = np.zeros((0,n)); Q = np.zeros((0,))
            else:
                P = np.vstack(P); Q = np.array(Q)

            Omega = np.diag(np.full(len(Q), 0.05**2)) if len(Q)>0 else np.zeros((0,0))
            Sigma_ = Sigma.values
            M = np.linalg.inv(tau*Sigma_) + P.T @ np.linalg.inv(Omega) @ P if len(Q)>0 else np.linalg.inv(tau*Sigma_)
            mu_post = np.linalg.solve(M, (np.linalg.inv(tau*Sigma_) @ Pi) + (P.T @ np.linalg.inv(Omega) @ Q if len(Q)>0 else 0))
            w_bl = np.linalg.solve(delta*Sigma_, mu_post)
            w_bl = np.maximum(w_bl, 0); w_bl = w_bl / w_bl.sum()

            w_ser = pd.Series(w_bl, index=R.columns, name="BL Weight")
            st.dataframe(w_ser.to_frame().T, use_container_width=True)
            fig, ax = plt.subplots(figsize=(5,5))
            w_ser.plot.pie(autopct="%.1f%%", ax=ax)
            ax.set_ylabel(""); ax.set_title("Black–Litterman Weights")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"BL error: {e}")

    st.subheader("🧪 Strategy Studio (sandbox)")
    ss_symbol = st.text_input("Symbol (Studio)", value="AAPL", key="ti_ss_symbol").upper()
    ss_kind = st.selectbox("Strategy type", ["MA Trend", "Donchian Breakout", "RSI Mean-Reversion"], key="sb_ss_kind")
    ss_len1 = st.number_input("Len1 (fast/RSI/DC)", 5, 200, 10, 1, key="ni_ss_len1")
    ss_len2 = st.number_input("Len2 (slow)", 5, 400, 50, 1, key="ni_ss_len2")
    if st.button("🏃 Run Studio Backtest", key="btn_studio"):
        try:
            px = load_prices(ss_symbol, "2y", "1d")
            ind = compute_indicators(px, ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=False)
            sig = ind.copy()
            sig["Trade"] = 0
            if ss_kind=="MA Trend":
                f = ind["Close"].ewm(span=int(ss_len1), adjust=False).mean()
                s = ind["Close"].ewm(span=int(ss_len2), adjust=False).mean()
                sig["Trade"] = np.where(f > s, 1, 0)
            elif ss_kind=="Donchian Breakout":
                up = ind["High"].rolling(int(ss_len1)).max()
                lo = ind["Low"].rolling(int(ss_len1)).min()
                sig["Trade"] = np.where(ind["Close"]>up.shift(1), 1, np.where(ind["Close"]<lo.shift(1), -1 if allow_short else 0, 0))
            else:
                r = ind[f"RSI{rsi_period}"]
                sig["Trade"] = np.where(r<30, 1, np.where(r>70, -1 if allow_short else 0, 0))

            if guard_earn:
                ed = get_earnings_dates(ss_symbol, limit=8)
                sig = apply_event_flatten(sig, ed, days_before=int(earn_bef), days_after=int(earn_aft))

            bt, md, sh, wr, trd, tim, cagr = backtest(sig, allow_short=allow_short, cost_bps=cost_bps,
                                                       sl_atr_mult=sl_atr_mult, tp_atr_mult=tp_atr_mult,
                                                       vol_target=vol_target, interval="1d")
            st.markdown(f"**Studio Strat:** Return={(bt['CumStrat'].iloc[-1]-1)*100:.2f}% | Sharpe={sh:.2f} | MaxDD={md:.2f}% | Trades={trd}")
            fig, ax = plt.subplots(figsize=(9,3))
            ax.plot(bt.index, bt["CumBH"], ":", label="BH"); ax.plot(bt.index, bt["CumStrat"], label="Studio"); ax.legend(); ax.set_title("Studio Equity")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Studio error: {e}")

    st.subheader("📊 Portfolio Simulator")
    st.info("Enter your positions in CSV: ticker,shares,cost_basis")
    holdings = st.text_area("e.g.\nAAPL,10,150\nMSFT,5,300", height=100, key="ta_port_holdings")
    if st.button("▶️ Simulate Portfolio", key="btn_sim_port"):
        rows = [r.strip().split(",") for r in holdings.splitlines() if r.strip()]
        data=[]
        for idx, row in enumerate(rows, 1):
            if len(row) != 3:
                st.warning(f"Skipping invalid row {idx}: {row}"); continue
            ticker_, shares, cost = row
            tkr = _map_symbol(ticker_.upper().strip())
            try:
                s=float(shares); c=float(cost)
            except:
                st.warning(f"Invalid numbers on row {idx}: {row}"); continue
            hist = yf.Ticker(tkr).history(period="1d", auto_adjust=True)
            if hist.empty:
                st.warning(f"No price for {tkr}"); continue
            price=float(hist["Close"].iloc[-1])
            invested=s*c; value=s*price; pnl=value-invested
            pnl_pct=(pnl/invested*100) if invested else np.nan

            px = load_prices(tkr, period_sel, interval_sel)
            if px.empty:
                comp_sugg="N/A"
            else:
                df_i = compute_indicators(px, ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=include_bb)
                if df_i.empty:
                    comp_sugg="N/A"
                else:
                    df_csig = build_composite(df_i, ma_window, rsi_period,
                                              use_weighted=use_weighted, w_ma=w_ma, w_rsi=w_rsi, w_macd=w_macd, w_bb=w_bb,
                                              include_bb=include_bb, threshold=comp_thr, allow_short=allow_short)
                    if df_csig.empty:
                        comp_sugg="N/A"
                    else:
                        score = float(df_csig["Composite"].iloc[-1]) if "Composite" in df_csig else 0.0
                        comp_sugg = "🟢 BUY" if score>=comp_thr else ("🔴 SELL" if score<=-comp_thr else "🟡 HOLD")

            if pnl_pct > profit_target:     suggestion="🔴 SELL"
            elif pnl_pct < -loss_limit:     suggestion="🟢 BUY"
            else:                           suggestion=comp_sugg

            data.append({
                "Ticker":tkr,"Shares":s,"Cost Basis":c,"Price":price,
                "Market Value":value,"Invested":invested,"P/L":pnl,
                "P/L %":pnl_pct,"Composite Sig":comp_sugg,"Suggestion":suggestion
            })
        if data:
            df_port=pd.DataFrame(data).set_index("Ticker")
            st.dataframe(df_port, use_container_width=True)
            c1,c2,c3 = st.columns(3)
            c1.metric("Total Market Value", f"${df_port['Market Value'].sum():,.2f}")
            c2.metric("Total Invested",     f"${df_port['Invested'].sum():,.2f}")
            c3.metric("Total P/L",          f"${df_port['Market Value'].sum()-df_port['Invested'].sum():,.2f}")
            fig, ax=plt.subplots(figsize=(5,5))
            df_port["Market Value"].plot.pie(autopct="%.1f%%", ax=ax)
            ax.set_ylabel(""); ax.set_title("Portfolio Allocation")
            st.pyplot(fig)
        else:
            st.error("No valid holdings provided.")

    st.subheader("🎲 Monte Carlo (Bootstrap) of Strategy Returns")
    mc_symbol = st.text_input("Symbol (MC)", value="AAPL", key="ti_mc_symbol").upper()
    n_paths = st.slider("Paths", 200, 3000, 800, 100, key="sl_mc_paths")
    if st.button("Run Monte Carlo", key="btn_mc"):
        try:
            px = load_prices(mc_symbol, "2y", "1d")
            ind = compute_indicators(px, ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=True)
            sig = build_composite(ind, ma_window, rsi_period,
                                  use_weighted=use_weighted, w_ma=w_ma, w_rsi=w_rsi, w_macd=w_macd, w_bb=w_bb,
                                  include_bb=include_bb, threshold=comp_thr, allow_short=allow_short)
            bt, *_ = backtest(sig, allow_short=allow_short, cost_bps=cost_bps,
                               sl_atr_mult=sl_atr_mult, tp_atr_mult=tp_atr_mult,
                               vol_target=vol_target, interval="1d")
            r = bt["StratRet"].dropna().values
            if len(r) < 50:
                st.warning("Not enough strategy bars to bootstrap."); st.stop()
            N = len(r)
            endings = []
            for _ in range(int(n_paths)):
                samp = np.random.choice(r, size=N, replace=True)
                eq = (1 + pd.Series(samp)).cumprod().iloc[-1]
                endings.append(eq)
            endings = np.array(endings)
            pct = (np.percentile(endings, [5, 25, 50, 75, 95]) - 1) * 100
            c1,c2,c3,c4,c5 = st.columns(5)
            c1.metric("P5%",  f"{pct[0]:.1f}%"); c2.metric("P25%", f"{pct[1]:.1f}%"); c3.metric("Median", f"{pct[2]:.1f}%"); c4.metric("P75%", f"{pct[3]:.1f}%"); c5.metric("P95%", f"{pct[4]:.1f}%")
            fig, ax = plt.subplots(figsize=(8,3))
            ax.hist((endings-1)*100, bins=30, alpha=0.8)
            ax.set_title("Monte Carlo: Distribution of End Returns (%)")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Monte Carlo error: {e}")

# ───────────────────────────── HELP ─────────────────────────────
with tab_help:
    st.header("How QuantaraX Pro v5.1 Works")
    st.markdown(r"""
**QuantaraX Pro v5.1** packs:
- **Adaptive Composite** (MA/RSI/MACD + optional Bollinger) with vol-aware threshold & size.
- **Event Guards**: flatten around earnings and sample FOMC dates.
- **Robust Backtester**: long/short, costs, ATR stops/targets, vol targeting.
- **Options Lens**: ATM IV, skew, within-chain IV-rank.
- **ML Lab**: Purged TimeSeries CV, OOF probabilities → backtest & feature importances.
- **Regimes**: HMM (if `hmmlearn`), else KMeans fallback.
- **Portfolio**: Risk Parity, **Black–Litterman**, Strategy Studio, Monte Carlo.

If you still see a duplicate-ID error, it means a new widget label collided elsewhere.
All widgets here now have explicit `key=` values to avoid that.
""")
