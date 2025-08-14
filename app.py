# app.py — QuantaraX Decision Engine (v24)
# ---------------------------------------------------------------------
# pip install:
#   streamlit yfinance pandas numpy matplotlib feedparser vaderSentiment scikit-learn

import math
from typing import List, Tuple
import time
import warnings
from datetime import datetime, timezone

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import feedparser

warnings.simplefilter("ignore", FutureWarning)

# ───────────────────────────── Page Setup ─────────────────────────────
st.set_page_config(page_title="QuantaraX — Decision Engine (v24)", layout="wide")
analyzer = SentimentIntensityAnalyzer()
REC_MAP = {1: "🟢 BUY", 0: "🟡 HOLD", -1: "🔴 SELL"}

# ───────────────────────────── Sidebar ─────────────────────────────
st.sidebar.header("Global Controls")

# Beginner/Pro mode (moved to sidebar)
MODE = st.sidebar.radio("Mode", ["Beginner", "Pro"], index=0, key="mode_toggle")

DEFAULTS = dict(ma_window=10, rsi_period=14, macd_fast=12, macd_slow=26, macd_signal=9)
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.sidebar.button("🔄 Reset to defaults", key="btn_reset_defaults"):
    for k, v in DEFAULTS.items():
        st.session_state[k] = v

st.sidebar.subheader("Indicator Parameters")
ma_window   = st.sidebar.slider("MA window",      5, 60, st.session_state["ma_window"],   key="ma_window")
rsi_period  = st.sidebar.slider("RSI lookback",   5, 30, st.session_state["rsi_period"],  key="rsi_period")
macd_fast   = st.sidebar.slider("MACD fast span", 5, 20, st.session_state["macd_fast"],   key="macd_fast")
macd_slow   = st.sidebar.slider("MACD slow span", 20, 50, st.session_state["macd_slow"],  key="macd_slow")
macd_signal = st.sidebar.slider("MACD sig span",  5, 20, st.session_state["macd_signal"], key="macd_signal")

st.sidebar.subheader("Composite v2 (advanced)")
use_weighted = st.sidebar.toggle("Use weighted composite", value=True, key="use_weighted")
include_bb   = st.sidebar.toggle("Include Bollinger Bands", value=True, key="include_bb")
w_ma   = st.sidebar.slider("Weight • MA",   0.0, 2.0, 1.0, 0.1, key="w_ma")
w_rsi  = st.sidebar.slider("Weight • RSI",  0.0, 2.0, 1.0, 0.1, key="w_rsi")
w_macd = st.sidebar.slider("Weight • MACD", 0.0, 2.0, 1.0, 0.1, key="w_macd")
w_bb   = st.sidebar.slider("Weight • BB",   0.0, 2.0, 0.5, 0.1, key="w_bb") if include_bb else 0.0
comp_thr = st.sidebar.slider("Composite trigger (enter/exit)", 0.0, 3.0, 1.0, 0.1, key="comp_thr")

st.sidebar.subheader("Risk & Costs")
allow_short = st.sidebar.toggle("Allow shorts", value=False, key="allow_short")
cost_bps    = st.sidebar.slider("Trading cost (bps/side)", 0.0, 25.0, 5.0, 0.5, key="cost_bps")
sl_atr_mult = st.sidebar.slider("Stop • ATR ×", 0.0, 5.0, 2.0, 0.1, key="sl_atr_mult")
tp_atr_mult = st.sidebar.slider("Target • ATR ×", 0.0, 8.0, 3.0, 0.1, key="tp_atr_mult")
vol_target  = st.sidebar.slider("Vol targeting (annual)", 0.0, 0.5, 0.0, 0.05, key="vol_target")

st.sidebar.subheader("Data")
period_sel   = st.sidebar.selectbox("History", ["6mo","1y","2y","5y"], index=1, key="period_sel")
interval_sel = st.sidebar.selectbox("Interval", ["1d","1h"], index=0, key="interval_sel")

st.sidebar.subheader("Portfolio Guardrails")
profit_target = st.sidebar.slider("Profit target (%)", 1, 100, 10, key="profit_target")
loss_limit    = st.sidebar.slider("Loss limit (%)",  1, 100, 5,  key="loss_limit")

# ───────────────────────────── Helpers ─────────────────────────────
def _map_symbol(sym: str) -> str:
    s = sym.strip().upper()
    if "/" in s:  # e.g., BTC/USDT → BTC-USD
        base, quote = s.split("/")
        quote = "USD" if quote in ("USDT", "USD") else quote
        return f"{base}-{quote}"
    return s

def _to_float(x) -> float:
    try:
        return float(x.item()) if hasattr(x, "item") else float(x)
    except Exception:
        try:
            return float(getattr(x, "iloc", [x])[0])
        except Exception:
            return float("nan")

def _utcnow_ts() -> pd.Timestamp:
    # tz-aware UTC timestamp compatible with pandas arithmetic
    return pd.Timestamp(datetime.now(timezone.utc))

@st.cache_data(show_spinner=False, ttl=900)
def load_prices(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """Robust loader with retry; auto_adjust=True to avoid warnings."""
    sym = _map_symbol(symbol)
    for attempt in range(3):
        try:
            df = yf.download(sym, period=period, interval=interval, auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            if not df.empty:
                return df.dropna()
        except Exception:
            time.sleep(0.8 * (attempt + 1))
    return pd.DataFrame()

def data_health(df: pd.DataFrame, interval: str) -> dict:
    """Return freshness label and age; handles tz-aware/naive safely."""
    if df.empty:
        return {"label": "Stale", "age_text": "n/a"}
    last_ts = df.index[-1]
    if isinstance(last_ts, pd.Timestamp) and last_ts.tzinfo is None:
        last_ts = last_ts.tz_localize("UTC")
    now = _utcnow_ts()
    fresh_hours = max(0.0, (now - last_ts).total_seconds() / 3600.0)
    label = "Fresh" if fresh_hours <= (1 if interval == "1h" else 28) else "Stale"
    return {"label": label, "age_text": f"{fresh_hours:.1f}h ago"}

def safe_get_news(symbol: str) -> list:
    """Try yfinance news; gracefully fallback to empty list if blocked/timeouts."""
    try:
        return getattr(yf.Ticker(_map_symbol(symbol)), "news", []) or []
    except Exception:
        return []

def rss_news(symbol: str, limit: int = 5) -> list:
    try:
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={_map_symbol(symbol)}&region=US&lang=en-US"
        feed = feedparser.parse(url)
        return [{"title": e.title, "link": e.link} for e in feed.entries[:limit]]
    except Exception:
        return []

def clean_earnings_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize earnings dates to a consistent 'earn_date' column (tz-aware UTC)."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        out = out.reset_index()
    # locate 'earnings' + 'date' like column or the first datetime col
    date_col = None
    for c in out.columns:
        cl = str(c).lower().replace(" ", "")
        if "earn" in cl and "date" in cl:
            date_col = c; break
    if date_col is None:
        for c in out.columns:
            if pd.api.types.is_datetime64_any_dtype(out[c]):
                date_col = c; break
    if date_col is None:
        date_col = out.columns[0]
    out = out.rename(columns={date_col: "earn_date"})
    out["earn_date"] = pd.to_datetime(out["earn_date"], errors="coerce", utc=True)
    return out.dropna(subset=["earn_date"])

def next_and_last_earnings(symbol: str) -> Tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Return (next_earnings, last_earnings). Both tz-aware UTC or None."""
    try:
        raw = yf.Ticker(_map_symbol(symbol)).get_earnings_dates(limit=12)
    except Exception:
        raw = None
    cal = clean_earnings_df(raw)
    if cal.empty:
        return None, None
    now = _utcnow_ts()
    future = cal[cal["earn_date"] >= now].sort_values("earn_date")
    past   = cal[cal["earn_date"] <  now].sort_values("earn_date")
    nxt = future["earn_date"].iloc[0] if not future.empty else None
    last = past["earn_date"].iloc[-1] if not past.empty else None
    return nxt, last

def render_next_earnings(symbol: str):
    nxt, last = next_and_last_earnings(symbol)
    if nxt is not None:
        st.info(f"📅 **Next earnings:** {nxt.date()}")
    elif last is not None:
        st.info(f"📅 **Last earnings:** {last.date()} (no upcoming date found)")
    else:
        st.info("📅 Earnings: unavailable")

# ─────────── Indicators / Composite ───────────
def compute_indicators(df: pd.DataFrame, ma_w: int, rsi_p: int, mf: int, ms: int, sig: int,
                       use_bb: bool = True) -> pd.DataFrame:
    d = df.copy()
    need = {"Open", "High", "Low", "Close"}
    if d.empty or not need.issubset(set(d.columns)):
        return pd.DataFrame()

    # MA
    d[f"MA{ma_w}"] = d["Close"].rolling(ma_w, min_periods=ma_w).mean()

    # RSI (EMA-based)
    chg = d["Close"].diff()
    up, dn = chg.clip(lower=0), -chg.clip(upper=0)
    ema_up   = up.ewm(com=rsi_p-1, adjust=False).mean()
    ema_down = dn.ewm(com=rsi_p-1, adjust=False).mean()
    rs = ema_up / ema_down.replace(0, np.nan)
    d[f"RSI{rsi_p}"] = 100 - 100 / (1 + rs)

    # MACD
    ema_f = d["Close"].ewm(span=mf, adjust=False).mean()
    ema_s = d["Close"].ewm(span=ms, adjust=False).mean()
    macd_line = ema_f - ema_s
    d["MACD"] = macd_line
    d["MACD_Signal"] = macd_line.ewm(span=sig, adjust=False).mean()

    # ATR
    pc = d["Close"].shift(1)
    tr = pd.concat([(d["High"]-d["Low"]).abs(), (d["High"]-pc).abs(), (d["Low"]-pc).abs()], axis=1).max(axis=1)
    d["ATR"] = tr.ewm(alpha=1/14, adjust=False).mean()

    # Bollinger
    if use_bb:
        w = 20; k = 2.0
        mid = d["Close"].rolling(w, min_periods=w).mean()
        sd  = d["Close"].rolling(w, min_periods=w).std(ddof=0)
        d["BB_M"], d["BB_U"], d["BB_L"] = mid, mid + k*sd, mid - k*sd

    # Stochastic
    klen = 14
    ll = d["Low"].rolling(klen, min_periods=klen).min(); hh = d["High"].rolling(klen, min_periods=klen).max()
    rng = (hh - ll).replace(0, np.nan)
    d["STO_K"] = 100 * (d["Close"] - ll) / rng
    d["STO_D"] = d["STO_K"].rolling(3, min_periods=3).mean()

    # ADX (simplified Wilder's)
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

    # Donchian Channels
    dc_n = 20
    d["DC_U"] = d["High"].rolling(dc_n, min_periods=dc_n).max()
    d["DC_L"] = d["Low"].rolling(dc_n, min_periods=dc_n).min()

    # Keltner Channels (EMA + ATR)
    kel_n = 20
    ema_mid = d["Close"].ewm(span=kel_n, adjust=False).mean()
    d["KC_U"] = ema_mid + 2 * d["ATR"]
    d["KC_L"] = ema_mid - 2 * d["ATR"]

    return d.dropna()

def build_composite(df: pd.DataFrame, ma_w: int, rsi_p: int,
                    *, use_weighted=True, w_ma=1.0, w_rsi=1.0, w_macd=1.0, w_bb=0.5,
                    include_bb=True, threshold=0.0, allow_short=False) -> pd.DataFrame:
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

    if allow_short:
        trade = np.where(comp >= threshold, 1, np.where(comp <= -threshold, -1, 0))
    else:
        trade = np.where(comp >= threshold, 1, 0)

    d["MA_Signal"], d["RSI_Signal"], d["MACD_Signal2"] = ma_sig, rsi_sig, macd_sig2
    if include_bb: d["BB_Signal"] = bb_sig
    d["Composite"] = comp.astype(float)
    d["Trade"] = trade.astype(int)
    return d

# ─────────── Backtest ───────────
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

    # Base position from trade signal
    if allow_short:
        d["Position"] = d.get("Trade", 0).shift(1).fillna(0).clip(-1, 1)
        base_ret = np.where(d["Position"] >= 0, d["Return"], -d["Return"])
    else:
        d["Position"] = d.get("Trade", 0).shift(1).fillna(0).clip(0, 1)
        base_ret = d["Position"] * d["Return"]

    # Vol targeting (rolling 20 bars)
    if vol_target and vol_target > 0:
        look = 20
        daily_vol = d["Return"].rolling(look).std(ddof=0)
        ann = 252 if interval == "1d" else 252*6
        realized = daily_vol * math.sqrt(ann)
        scale = (vol_target / realized).clip(0, 3.0).fillna(0.0)  # cap leverage
        base_ret = base_ret * scale

    # Costs on trades
    cost = cost_bps/10000.0
    pos_change = d["Position"].diff().fillna(0).abs()
    tcost = -2.0*cost*(pos_change > 0).astype(float)  # open+close
    d["StratRet"] = pd.Series(base_ret, index=d.index).fillna(0.0) + tcost

    # ATR exits → flatten next bar
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
    d["CumBH"]    = (1 + ret_bh).cumprod()
    d["CumStrat"] = (1 + ret_st).cumprod()

    max_dd, sharpe, win_rt, trades, tim, cagr, _ = _stats_from_equity(d, interval)
    return d, max_dd, sharpe, win_rt, trades, tim, cagr

# ───────────────────────────── Tabs ─────────────────────────────
tab_engine, tab_ml, tab_scan, tab_regime, tab_port, tab_help = st.tabs(
    ["🚀 Engine", "🧠 ML Lab", "📡 Scanner", "📉 Regimes", "💼 Portfolio", "❓ Help"]
)

# ───────────────────────────── ENGINE ─────────────────────────────
with tab_engine:
    st.title("QuantaraX — Decision Engine (v24)")

    c_sym, c_mode = st.columns([3, 1])
    with c_sym:
        ticker = st.text_input("Symbol (e.g., AAPL or BTC/USDT)", "AAPL", key="inp_symbol").upper()
    with c_mode:
        st.caption("Mode changes detail density throughout the app.")
        st.metric("Mode", MODE)

    # Live close & freshness
    px_live = load_prices(ticker, "5d", "1d")
    if not px_live.empty and "Close" in px_live:
        last_px = _to_float(px_live["Close"].iloc[-1])
        meta = data_health(px_live, "1d")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader(f"💲 Last Close: ${last_px:.2f}")
        with c2:
            st.subheader(f"✅ {meta['label']} • {meta['age_text']}")

    # Earnings
    render_next_earnings(ticker)

    # News (safe → RSS fallback)
    with st.expander("📰 Recent News & Sentiment"):
        news = safe_get_news(ticker)
        shown = 0
        if news:
            for art in news:
                t_ = art.get("title",""); l_ = art.get("link","")
                if not (t_ and l_): continue
                txt = art.get("summary", t_)
                score = analyzer.polarity_scores(txt)["compound"]
                emoji = "🔺" if score>0.1 else ("🔻" if score<-0.1 else "➖")
                st.markdown(f"- [{t_}]({l_}) {emoji}")
                shown += 1
                if shown >= 5: break
        if shown == 0:
            rss = rss_news(ticker, limit=5)
            if rss:
                for r in rss:
                    st.markdown(f"- [{r['title']}]({r['link']})")
            else:
                st.info("No recent news found.")

    if st.button("▶️ Run Composite Backtest", key="btn_engine_backtest"):
        px = load_prices(ticker, period_sel, interval_sel)
        if px.empty:
            st.error(f"No data for '{ticker}'"); st.stop()

        df_raw = compute_indicators(px, ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=include_bb)
        if df_raw.empty:
            st.error("Not enough data after indicators (try longer period or smaller windows)."); st.stop()

        df_sig = build_composite(df_raw, ma_window, rsi_period,
                                 use_weighted=use_weighted, w_ma=w_ma, w_rsi=w_rsi, w_macd=w_macd, w_bb=w_bb,
                                 include_bb=include_bb, threshold=comp_thr, allow_short=allow_short)
        if df_sig.empty:
            st.error("Composite could not be built (insufficient rows)."); st.stop()

        df_c, max_dd, sharpe, win_rt, trades, tim, cagr = backtest(
            df_sig, allow_short=allow_short, cost_bps=cost_bps,
            sl_atr_mult=sl_atr_mult, tp_atr_mult=tp_atr_mult, vol_target=vol_target, interval=interval_sel
        )

        last_trade = int(df_sig["Trade"].tail(1).iloc[0]) if "Trade" in df_sig.columns and not df_sig.empty else 0
        rec = REC_MAP.get(1 if last_trade>0 else (-1 if last_trade<0 else 0), "🟡 HOLD")
        st.success(f"**{ticker}**: {rec}")

        # Reasoning
        last = df_sig.tail(1).iloc[0]
        ma_s  = int(last.get("MA_Signal", 0))
        rsi_s = int(last.get("RSI_Signal", 0))
        macd_s= int(last.get("MACD_Signal2", 0))
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
            st.write(f"- **Composite (weighted):** {float(last.get('Composite', 0)):.2f}  (threshold={comp_thr:.1f})")

        # Metrics
        bh_last    = float(df_c["CumBH"].tail(1).iloc[0])  if "CumBH" in df_c and not df_c["CumBH"].empty else 1.0
        strat_last = float(df_c["CumStrat"].tail(1).iloc[0]) if "CumStrat" in df_c and not df_c["CumStrat"].empty else 1.0
        colA, colB, colC, colD, colE, colF = st.columns(6)
        colA.metric("CAGR", f"{(0 if np.isnan(cagr) else cagr):.2f}%")
        colB.metric("Sharpe", f"{(0 if np.isnan(sharpe) else sharpe):.2f}")
        colC.metric("Max DD", f"{max_dd:.2f}%")
        colD.metric("Win Rate", f"{win_rt:.1f}%")
        colE.metric("Trades", f"{trades}")
        colF.metric("Time in Mkt", f"{tim:.1f}%")

        st.markdown(f"- **Buy & Hold:** {(bh_last-1)*100:.2f}%  \n- **Strategy:** {(strat_last-1)*100:.2f}%")

        # Plots
        idx = df_c.index
        fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(11,12), sharex=True)
        ax1.plot(idx, df_c["Close"], label="Close")
        if f"MA{ma_window}" in df_c: ax1.plot(idx, df_c[f"MA{ma_window}"], label=f"MA{ma_window}")
        if include_bb and {"BB_U","BB_L"}.issubset(df_c.columns):
            ax1.plot(idx, df_c["BB_U"], label="BB Upper"); ax1.plot(idx, df_c["BB_L"], label="BB Lower")
        ax1.legend(); ax1.set_title("Price & Indicators")
        if "Composite" in df_c:
            ax2.bar(idx, df_c["Composite"]); ax2.set_title("Composite (weighted)")
        else:
            ax2.set_title("Composite (no data)")
        ax3.plot(idx, df_c["CumBH"], ":", label="BH")
        ax3.plot(idx, df_c["CumStrat"], "-", label="Strat"); ax3.legend(); ax3.set_title("Equity")
        plt.xticks(rotation=45); plt.tight_layout()
        st.pyplot(fig)

    # Tools
    st.markdown("---")
    with st.expander("⏱️ Multi-Timeframe Confirmation", expanded=False):
        mtf_symbol = st.text_input("Symbol (MTF)", value=ticker or "AAPL", key="inp_mtf_symbol")
        if st.button("🔍 Check MTF", key="btn_mtf"):
            try:
                d1 = compute_indicators(load_prices(mtf_symbol, "1y", "1d"), ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=True)
                dH = compute_indicators(load_prices(mtf_symbol, "30d", "1h"), ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=True)
                if d1.empty or dH.empty:
                    st.warning("Insufficient data for MTF.")
                else:
                    c1 = build_composite(d1, ma_window, rsi_period, use_weighted=True, w_ma=1.0, w_rsi=1.0, w_macd=1.0, w_bb=0.5, include_bb=True, threshold=1.0)
                    cH = build_composite(dH, ma_window, rsi_period, use_weighted=True, w_ma=1.0, w_rsi=1.0, w_macd=1.0, w_bb=0.5, include_bb=True, threshold=1.0)
                    daily  = float(c1["Composite"].iloc[-1]); hourly = float(cH["Composite"].iloc[-1])
                    st.write(f"**Daily composite:** {daily:.2f}")
                    st.write(f"**Hourly composite:** {hourly:.2f}")
                    if np.sign(daily) == np.sign(hourly):
                        st.success("✅ Signals agree")
                    else:
                        st.warning("⚠️ Signals disagree")
            except Exception as e:
                st.error(f"MTF error: {e}")

    with st.expander("🧪 Walk-Forward Optimization (OOS)", expanded=False):
        wf_symbol = st.text_input("Symbol (WFO)", value=ticker or "AAPL", key="inp_wfo_symbol")
        c1, c2 = st.columns(2)
        with c1:
            ins_bars = st.number_input("In-sample bars", 60, 252*3, 126, 1, key="wfo_ins")
            oos_bars = st.number_input("OOS bars", 20, 252, 63, 1, key="wfo_oos")
        with c2:
            w_thr = st.slider("Composite trigger (WFO)", 0.0, 3.0, 1.0, 0.1, key="wfo_thr")
            wf_allow_short = st.toggle("Allow shorts (WFO)", value=False, key="wfo_short")
        if st.button("🏃 Run Walk-Forward", key="btn_wfo"):
            try:
                pxW = load_prices(wf_symbol, "2y", "1d")
                if pxW.empty: st.warning("No data for WFO."); st.stop()

                def run_eq(ma_list, rsi_list, mf_list, ms_list, sig_list,
                           insample_bars, oos_bars,
                           w_ma=1.0, w_rsi=1.0, w_macd=1.0, w_bb=0.5, threshold=1.0,
                           allow_short=False, cost_bps=5.0):
                    oos_curves = []; summary = []
                    start=200; i=start
                    while i + insample_bars + oos_bars <= len(pxW):
                        ins = pxW.iloc[i : i+insample_bars]
                        oos = pxW.iloc[i+insample_bars : i+insample_bars+oos_bars]
                        best=None; best_score=-1e9
                        for mw in ma_list:
                            for rp in rsi_list:
                                for mf in mf_list:
                                    for ms in ms_list:
                                        for s in sig_list:
                                            ins_ind = compute_indicators(ins, mw, rp, mf, ms, s, use_bb=True)
                                            if ins_ind.empty: continue
                                            ins_sig = build_composite(ins_ind, mw, rp, use_weighted=True, w_ma=w_ma, w_rsi=w_rsi, w_macd=w_macd, w_bb=w_bb, include_bb=True, threshold=threshold, allow_short=allow_short)
                                            ins_bt, md, sh, wr, tr, ti, cg = backtest(ins_sig, allow_short=allow_short, cost_bps=cost_bps)
                                            perf = (ins_bt["CumStrat"].iloc[-1]-1)*100 if "CumStrat" in ins_bt else -1e9
                                            score = perf - abs(md)
                                            if score > best_score:
                                                best_score = score
                                                best = (mw, rp, mf, ms, s, sh, perf, md)
                        if best is None:
                            i += oos_bars; continue
                        mw, rp, mf, ms, s, sh, perf, mdd = best
                        oos_ind = compute_indicators(oos, mw, rp, mf, ms, s, use_bb=True)
                        if oos_ind.empty:
                            i += oos_bars; continue
                        oos_sig = build_composite(oos_ind, mw, rp, use_weighted=True, w_ma=w_ma, w_rsi=w_rsi, w_macd=w_macd, w_bb=w_bb, include_bb=True, threshold=threshold, allow_short=allow_short)
                        oos_bt, mo_dd, mo_sh, *_ = backtest(oos_sig, allow_short=allow_short, cost_bps=cost_bps)
                        if "CumStrat" in oos_bt:
                            oos_curves.append(oos_bt[["CumStrat"]].rename(columns={"CumStrat":"Equity"}))
                        summary.append({
                            "Window": f"{oos.index[0].date()} → {oos.index[-1].date()}",
                            "MA": mw, "RSI": rp, "MACDf": mf, "MACDs": ms, "SIG": s,
                            "OOS %": ((oos_bt["CumStrat"].iloc[-1]-1)*100) if "CumStrat" in oos_bt else np.nan,
                            "OOS Sharpe": mo_sh, "OOS MaxDD%": mo_dd
                        })
                        i += oos_bars
                    eq = pd.concat(oos_curves, axis=0) if oos_curves else pd.DataFrame()
                    sm = pd.DataFrame(summary)
                    return eq, sm

                eq, sm = run_eq(
                    ma_list=[ma_window, max(5, ma_window-5), min(60, ma_window+5)],
                    rsi_list=[rsi_period, max(5, rsi_period-7), min(30, rsi_period+7)],
                    mf_list=[macd_fast, max(5, macd_fast-4), min(20, macd_fast+4)],
                    ms_list=[macd_slow, max(20, macd_slow-6), min(50, macd_slow+6)],
                    sig_list=[macd_signal, max(5, macd_signal-4), min(20, macd_signal+4)],
                    insample_bars=int(ins_bars),
                    oos_bars=int(oos_bars),
                    w_ma=1.0, w_rsi=1.0, w_macd=1.0, w_bb=0.5,
                    threshold=w_thr, allow_short=wf_allow_short, cost_bps=5.0
                )
                if not sm.empty:
                    st.dataframe(sm, use_container_width=True)
                if not eq.empty and "Equity" in eq:
                    fig, ax = plt.subplots(figsize=(10,3))
                    ax.plot(eq.index, eq["Equity"]); ax.set_title("Walk-Forward OOS Equity (stitched)")
                    st.pyplot(fig)
                else:
                    st.info("WFO produced no OOS segments (not enough data).")
            except Exception as e:
                st.error(f"WFO error: {e}")

    # Batch Backtest
    st.markdown("---")
    st.markdown("### Batch Backtest")
    batch = st.text_area("Tickers (comma-separated)", "AAPL, MSFT, TSLA, SPY, QQQ", key="ta_batch").upper()
    if st.button("▶️ Run Batch Backtest", key="btn_batch"):
        perf=[]
        for t in [x.strip() for x in batch.split(",") if x.strip()]:
            px = load_prices(t, period_sel, interval_sel)
            if px.empty: continue
            df_t = compute_indicators(px, ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=include_bb)
            if df_t.empty: continue
            df_tc = build_composite(df_t, ma_window, rsi_period,
                                    use_weighted=use_weighted, w_ma=w_ma, w_rsi=w_rsi, w_macd=w_macd, w_bb=w_bb,
                                    include_bb=include_bb, threshold=comp_thr, allow_short=allow_short)
            if df_tc.empty: continue
            bt, md, sh, wr, trd, tim, cagr = backtest(df_tc, allow_short=allow_short, cost_bps=cost_bps,
                                                      sl_atr_mult=sl_atr_mult, tp_atr_mult=tp_atr_mult,
                                                      vol_target=vol_target, interval=interval_sel)
            comp_last = float(df_tc["Composite"].tail(1).iloc[0]) if "Composite" in df_tc else 0.0
            bh_last = float(bt["CumBH"].tail(1).iloc[0]) if "CumBH" in bt and not bt["CumBH"].empty else 1.0
            strat_last = float(bt["CumStrat"].tail(1).iloc[0]) if "CumStrat" in bt and not bt["CumStrat"].empty else 1.0
            perf.append({
                "Ticker":t,
                "Composite":comp_last,
                "Signal": REC_MAP.get(int(np.sign(comp_last)), "🟡 HOLD"),
                "Buy & Hold %": (bh_last-1)*100,
                "Strategy %":   (strat_last-1)*100,
                "Sharpe":       sh,
                "Max Drawdown": md,
                "Win Rate":     wr,
                "Trades":       trd,
                "Time in Mkt %": tim,
                "CAGR %":       cagr
            })
        if perf:
            df_perf = pd.DataFrame(perf).set_index("Ticker").sort_values("Strategy %", ascending=False)
            st.dataframe(df_perf, use_container_width=True)
            st.download_button("Download CSV", df_perf.to_csv(), "batch.csv", key="dl_batch")
        else:
            st.error("No valid data for batch tickers.")

    # Portfolio Simulator
    st.markdown("---")
    st.markdown("### 📊 Portfolio Simulator")
    st.info("Enter your positions in CSV: ticker,shares,cost_basis")
    holdings = st.text_area("Positions CSV", "AAPL,10,150\nMSFT,5,300", height=100, key="ta_portfolio")
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
            except Exception:
                st.warning(f"Invalid numbers on row {idx}: {row}"); continue
            hist = load_prices(tkr, "5d", "1d")
            if hist.empty:
                st.warning(f"No price for {tkr}"); continue
            price=_to_float(hist["Close"].iloc[-1])
            invested=s*c; value=s*price; pnl=value-invested
            pnl_pct=(pnl/invested*100) if invested else np.nan

            # Composite suggestion
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
                        score = float(df_csig["Composite"].tail(1).iloc[0]) if "Composite" in df_csig else 0.0
                        comp_sugg = "🟢 BUY" if score>=comp_thr else ("🔴 SELL" if score<=-comp_thr else "🟡 HOLD")

            # Guardrails override
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

# ───────────────────────────── ML LAB ─────────────────────────────
with tab_ml:
    st.title("🧠 ML Lab — Probabilistic Signals")
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, roc_auc_score
        from sklearn.inspection import permutation_importance
        SKLEARN_OK = True
    except Exception:
        SKLEARN_OK = False
    if not SKLEARN_OK:
        st.warning("scikit-learn not installed. Run: pip install scikit-learn")
    symbol = st.text_input("Symbol (ML)", value="AAPL", key="inp_ml_symbol").upper()
    horizon = st.slider("Prediction horizon (bars)", 1, 5, 1, key="ml_horizon")
    train_frac = st.slider("Train fraction", 0.5, 0.95, 0.8, key="ml_train_frac")
    proba_enter = st.slider("Enter if P(long) ≥", 0.50, 0.80, 0.55, 0.01, key="ml_p_enter")
    proba_exit  = st.slider("Enter short if P(long) ≤", 0.20, 0.50, 0.45, 0.01, key="ml_p_exit")
    run_ml = st.button("🤖 Train & Backtest", key="btn_ml_run")

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

    if run_ml:
        try:
            if not SKLEARN_OK:
                st.stop()
            px = load_prices(symbol, period_sel, interval_sel)
            ind = compute_indicators(px, ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=True)
            if ind.empty: st.error("Not enough data for indicators."); st.stop()
            X = _ml_features(ind)
            y = (ind["Close"].pct_change(horizon).shift(-horizon) > 0).reindex(X.index).astype(int)
            data = pd.concat([X, y.rename("y")], axis=1).dropna()
            if len(data) < 200:
                st.warning("Not enough rows for ML. Try longer history or daily interval."); st.stop()
            split = int(len(data) * float(train_frac))
            train, test = data.iloc[:split], data.iloc[split:]
            clf = RandomForestClassifier(n_estimators=400, max_depth=6, random_state=42, n_jobs=-1)
            clf.fit(train.drop(columns=["y"]), train["y"])
            proba = clf.predict_proba(test.drop(columns=["y"]))[:,1]
            y_true= test["y"].values
            acc = accuracy_score(y_true, (proba>0.5).astype(int))
            try:
                auc = roc_auc_score(y_true, proba)
            except Exception:
                auc = np.nan

            st.subheader("Out-of-sample performance")
            c1,c2 = st.columns(2)
            c1.metric("Accuracy (0.5)", f"{acc*100:.1f}%")
            c2.metric("ROC-AUC", f"{(0 if np.isnan(auc) else auc):.3f}")

            # Permutation importance
            try:
                pim = permutation_importance(clf, test.drop(columns=["y"]), y_true, n_repeats=5, random_state=42)
                imp = pd.Series(pim.importances_mean, index=test.drop(columns=["y"]).columns).sort_values(ascending=False)
                st.bar_chart(imp)
            except Exception:
                st.info("Permutation importance unavailable.")

            # Convert ML probs to trades and backtest
            if allow_short:
                sig = np.where(proba >= proba_enter, 1, np.where(proba <= proba_exit, -1, 0))
            else:
                sig = np.where(proba >= proba_enter, 1, 0)
            ml_df = ind.loc[test.index].copy()
            ml_df["Trade"] = pd.Series(sig, index=ml_df.index, dtype=int)
            bt, md, sh, wr, trd, tim, cagr = backtest(ml_df, allow_short=allow_short, cost_bps=cost_bps,
                                                       sl_atr_mult=sl_atr_mult, tp_atr_mult=tp_atr_mult,
                                                       vol_target=vol_target, interval=interval_sel)
            st.markdown(f"**ML Strategy OOS:** Return={(bt['CumStrat'].iloc[-1]-1)*100:.2f}% | Sharpe={sh:.2f} | MaxDD={md:.2f}% | Trades={trd}")
            fig, ax = plt.subplots(figsize=(9,3))
            ax.plot(bt.index, bt["CumBH"], ":", label="BH"); ax.plot(bt.index, bt["CumStrat"], label="ML Strat"); ax.legend(); ax.set_title("ML OOS Equity")
            st.pyplot(fig)

            latest_p = clf.predict_proba(data.drop(columns=["y"]).tail(1))[:,1][0]
            st.info(f"Latest P(long) = {latest_p:.3f}")
        except Exception as e:
            st.error(f"ML error: {e}")

# ───────────────────────────── SCANNER ─────────────────────────────
with tab_scan:
    st.title("📡 Universe Scanner — Composite + (optional) ML")
    universe = st.text_area("Tickers (comma-separated)",
                            "AAPL, MSFT, NVDA, TSLA, AMZN, GOOGL, META, NFLX, SPY, QQQ",
                            key="ta_scan_universe").upper()
    use_ml_scan = st.toggle("Include ML probability (needs scikit-learn)", value=False, key="tg_ml_scan")
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
                comp = float(sig["Composite"].tail(1).iloc[0]) if "Composite" in sig else 0.0
                rec = REC_MAP.get(int(np.sign(comp)), "🟡 HOLD")
                mlp = np.nan
                if use_ml_scan and SKLEARN_OK:
                    from sklearn.ensemble import RandomForestClassifier
                    X = pd.DataFrame(index=ind.index)
                    X["ret1"] = ind["Close"].pct_change()
                    X["rsi"]  = ind.get(f"RSI{rsi_period}", np.nan)
                    X["macd"] = ind.get("MACD", np.nan)
                    X = X.dropna()
                    y = (ind["Close"].pct_change().shift(-1) > 0).reindex(X.index).astype(int)
                    if len(X) > 200 and y.notna().sum() > 100:
                        split = int(len(X)*0.8)
                        clf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=0)
                        clf.fit(X.iloc[:split], y.iloc[:split])
                        mlp = float(clf.predict_proba(X.iloc[split:])[-1,1])
                rows.append({"Ticker":t, "Composite":comp, "Signal":rec, "ML P(long)":mlp})
            except Exception:
                continue
        if rows:
            df = pd.DataFrame(rows).set_index("Ticker").sort_values(["Signal","Composite"], ascending=[True,False])
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No results. Check tickers or increase history.")

# ───────────────────────────── REGIMES ─────────────────────────────
with tab_regime:
    st.title("📉 Regime Detection — Vol/Momentum Clusters")
    sym = st.text_input("Symbol (Regime)", value="SPY", key="inp_regime_symbol").upper()
    run_rg = st.button("Cluster Regimes", key="btn_regimes")

    if run_rg:
        try:
            px = load_prices(sym, "2y", "1d")
            ind = compute_indicators(px, ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=False)
            if ind.empty: st.error("Not enough data."); st.stop()
            feat = pd.DataFrame(index=ind.index)
            feat["vol20"] = ind["Close"].pct_change().rolling(20).std()
            feat["mom20"] = ind["Close"].pct_change(20)
            feat["ma_slope"] = ind[f"MA{ma_window}"].diff()
            feat = feat.dropna()
            try:
                from sklearn.cluster import KMeans
                km = KMeans(n_clusters=3, n_init=10, random_state=42)
                lab = km.fit_predict(feat)
            except Exception:
                q1 = feat.rank(pct=True)
                lab = (q1.mean(axis=1) > 0.66).astype(int) + (q1.mean(axis=1) < 0.33).astype(int)*2
            reg = pd.Series(lab, index=feat.index, name="Regime")
            joined = ind.join(reg, how="right")
            ret = joined["Close"].pct_change().groupby(joined["Regime"]).mean().sort_values()
            ord_map = {old:i for i, old in enumerate(ret.index)}
            joined["Regime"] = joined["Regime"].map(ord_map)
            st.dataframe(joined[["Close","Regime"]].tail(10))
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(joined.index, joined["Close"], label="Close", linewidth=1.2)
            ymin, ymax = joined["Close"].min()*0.98, joined["Close"].max()*1.02
            for r in sorted([x for x in joined["Regime"].dropna().unique()]):
                seg = joined[joined["Regime"]==r]
                ax.fill_between(seg.index, ymin, ymax, alpha=0.08)
            ax.set_ylim([ymin, ymax])
            ax.set_title("Price with Regime Shading")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Regime error: {e}")

# ───────────────────────────── PORTFOLIO ─────────────────────────────
with tab_port:
    st.title("💼 Portfolio — Optimizers & Monte Carlo")

    st.subheader("⚖️ Risk Parity Optimizer")
    opt_tickers = st.text_input("Tickers (comma-sep)", "AAPL, MSFT, TSLA, SPY, QQQ", key="inp_opt_tickers").upper()
    if st.button("🧮 Optimize (Risk Parity)", key="btn_opt_rp"):
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

    st.subheader("🎲 Monte Carlo (Bootstrap) of Strategy Returns")
    mc_symbol = st.text_input("Symbol (MC)", value="AAPL", key="inp_mc_symbol").upper()
    n_paths = st.slider("Paths", 200, 3000, 800, 100, key="mc_paths")
    run_mc = st.button("Run Monte Carlo", key="btn_mc")

    if run_mc:
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
    st.header("How QuantaraX Works (quick tour)")
    st.markdown("""
**Concept:** QuantaraX fuses classic technicals with a robust backtester and optional ML probabilities.
You can use it at two depths:

- **Beginner mode (sidebar toggle):** conservative defaults, minimal noise, plain-English reasoning.
- **Pro mode:** exposes more controls & shows extra diagnostics.

**Workflow**
1) Pick a symbol in **Engine** → run the composite backtest to view the latest recommendation and equity curve.  
2) Use **MTF** to confirm signal alignment across daily & 1-hour bars.  
3) Explore **WFO** for simple walk-forward robustness.  
4) **Scanner** ranks many tickers by composite (and optional ML probability).  
5) **ML Lab** trains a small RandomForest on indicator features for directional probability.  
6) **Regimes** clusters market states (vol/momentum/slope) to understand context.  
7) **Portfolio** provides a risk-parity allocator and Monte-Carlo distribution of strategy returns.

**Composite Signal (v2)**
- Signals from: **MA cross**, **RSI (30/70)**, **MACD cross**, optional **Bollinger mean reversion**.
- Weighted sum → thresholded into trades (long only, or long/short).
- Risk controls: trading costs (bps), ATR stop/target exits, and volatility targeting.

**Backtest Metrics**
- **CAGR, Sharpe, MaxDD, Win rate, Trades, Time-in-market** are shown under the chart.
- “Buy & Hold vs Strategy” compares end returns over the selected period/interval.

**Freshness badge**
- “Fresh” if the last bar is recent for the chosen interval; otherwise “Stale,” with time since last bar.

**Earnings**
- We clean & sort the provider’s calendar, show **next earnings** if it’s in the future; otherwise the last date.
- If the provider is unavailable, the section clearly states it.

**Notes**
- Market data & news are third-party and can occasionally time out; the app degrades gracefully and keeps running.
- For crypto pairs, use `BTC/USDT` (auto-maps to `BTC-USD`).
""")
