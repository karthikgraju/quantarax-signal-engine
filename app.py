# app.py — QuantaraX Pro (v17, FULL hardened)
# ---------------------------------------------------------------------------------
# pip install:
#   streamlit yfinance pandas numpy matplotlib feedparser vaderSentiment scikit-learn

import math
from typing import List, Tuple, Optional
import time
import warnings

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import feedparser

warnings.simplefilter("ignore", FutureWarning)

# Optional ML imports
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.inspection import permutation_importance
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

# ───────────────────────────── Page Setup ─────────────────────────────
st.set_page_config(page_title="QuantaraX Pro v17", layout="wide")
analyzer = SentimentIntensityAnalyzer()
rec_map = {1: "🟢 BUY", 0: "🟡 HOLD", -1: "🔴 SELL"}

# ───────────────────────────── Tabs ─────────────────────────────
TAB_TITLES = [
    "🚀 Engine",
    "🧠 ML Lab",
    "📡 Scanner",
    "📉 Regimes",
    "💼 Portfolio",
    "❓ Help",
]
(tab_engine, tab_ml, tab_scan, tab_regime, tab_port, tab_help) = st.tabs(TAB_TITLES)

# ───────────────────────────── Sidebar (unique keys) ─────────────────────────────
st.sidebar.header("Global Controls")
DEFAULTS = dict(ma_window=10, rsi_period=14, macd_fast=12, macd_slow=26, macd_signal=9)
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.sidebar.button("🔄 Reset to defaults", key="btn_reset_defaults"):
    for k, v in DEFAULTS.items():
        st.session_state[k] = v

# Modes
mode = st.sidebar.radio("Mode", ["Beginner", "Pro"], index=0, key="mode_sel")

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

st.sidebar.subheader("Capital & Sizing (for Playbook)")
capital_usd = st.sidebar.number_input("Account capital ($)", min_value=0.0, value=10000.0, step=100.0, key="acct_cap")
risk_per_trade_pct = st.sidebar.slider("Risk budget per trade (%)", 0.1, 5.0, 1.0, 0.1, key="risk_budget")

st.sidebar.subheader("Data")
period_sel   = st.sidebar.selectbox("History", ["6mo","1y","2y","5y"], index=1, key="period_sel")
interval_sel = st.sidebar.selectbox("Interval", ["1d","1h"], index=0, key="interval_sel")

st.sidebar.subheader("Portfolio Guardrails")
profit_target = st.sidebar.slider("Profit target (%)", 1, 100, 10, key="profit_target")
loss_limit    = st.sidebar.slider("Loss limit (%)",  1, 100, 5,  key="loss_limit")

st.sidebar.subheader("Earnings (override, optional)")
earn_override = st.sidebar.text_input("YYYY-MM-DD (optional)", value="", key="earn_override")

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
            return float(getattr(x, "iloc", lambda *a, **k: [np.nan])[-1])
        except Exception:
            return float("nan")

def _dtindex_to_utc(idx: pd.Index) -> pd.DatetimeIndex:
    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is None:
            return idx.tz_localize("UTC")
        return idx.tz_convert("UTC")
    return pd.DatetimeIndex(pd.to_datetime(idx, errors="coerce"), tz="UTC")

def _ts_to_utc(ts: pd.Timestamp) -> pd.Timestamp:
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")

def data_health(df: pd.DataFrame, interval: str) -> dict:
    """Basic data freshness/continuity check (returns dict with metrics)."""
    if df is None or df.empty:
        return {"ok": False, "reason": "empty"}
    idx = _dtindex_to_utc(df.index)
    now = pd.Timestamp.now(tz="UTC")
    last_ts = idx[-1]
    fresh_hours = max(0.0, (now - last_ts).total_seconds() / 3600.0)
    exp_h = 24.0 if interval == "1d" else 1.0
    diffs = idx.to_series().diff().dt.total_seconds().div(3600.0).dropna()
    gaps = int((diffs > 2.5 * exp_h).sum())
    return {
        "ok": True,
        "rows": int(len(df)),
        "start": idx[0],
        "end": last_ts,
        "fresh_hours": float(fresh_hours),
        "gaps": gaps,
    }

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
                df = df.dropna()
                df.index = _dtindex_to_utc(df.index)
                return df
        except Exception:
            time.sleep(0.8 * (attempt + 1))
    return pd.DataFrame()

def safe_get_news(symbol: str) -> list:
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

def parse_date_str(s: str) -> Optional[pd.Timestamp]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        dt = pd.to_datetime(s, errors="raise")
        return _ts_to_utc(pd.Timestamp(dt))
    except Exception:
        return None

def next_earnings_date(symbol: str, manual_override: Optional[str] = None) -> Tuple[Optional[pd.Timestamp], str]:
    """
    Resolve next earnings UTC date with sanity checks & optional manual override.
    Returns (date_or_none, note).
    """
    ov = parse_date_str(manual_override)
    if ov is not None:
        return ov.normalize(), "manual-override"

    try:
        cal = yf.Ticker(_map_symbol(symbol)).get_earnings_dates(limit=12)
        if not isinstance(cal, pd.DataFrame) or cal.empty:
            return None, "unavailable"
        df = cal.copy()
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
        # detect date column
        date_col = None
        for c in df.columns:
            cl = str(c).lower().replace(" ", "")
            if "earn" in cl and "date" in cl:
                date_col = c; break
        if date_col is None:
            for c in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[c]):
                    date_col = c; break
        if date_col is None:
            date_col = df.columns[0]

        df = df.rename(columns={date_col: "earn_date"})
        df["earn_date"] = pd.to_datetime(df["earn_date"], errors="coerce")
        df = df.dropna(subset=["earn_date"]).sort_values("earn_date")
        if df.empty:
            return None, "unavailable"

        today = pd.Timestamp.now(tz="UTC").normalize()
        future = df[df["earn_date"].dt.tz_localize("UTC", nonexistent="NaT", ambiguous="NaT", errors="coerce")
                    if df["earn_date"].dt.tz is None else df["earn_date"]].copy()
        # normalize tz
        future["earn_date"] = future["earn_date"].apply(lambda t: _ts_to_utc(pd.Timestamp(t)).normalize())
        nxt = future[future["earn_date"] >= today]["earn_date"]
        ed = nxt.iloc[0] if not nxt.empty else future["earn_date"].iloc[-1]

        # Sanity: if > 200 days away, flag low confidence (some tickers are odd)
        if abs((ed - today).days) > 200:
            return ed, "low-confidence"
        # If in the past by > 5 days, possibly stale
        if (today - ed).days > 5:
            return ed, "stale"
        return ed, "ok"
    except Exception:
        return None, "error"

def render_next_earnings(symbol: str, manual_override: Optional[str] = None) -> None:
    ed, note = next_earnings_date(symbol, manual_override=manual_override)
    if ed is None:
        st.info("📅 Earnings: unavailable", icon="🗓️")
        return
    badge = {"ok":"✅", "manual-override":"✏️", "stale":"⚠️", "low-confidence":"❓", "unavailable":"—", "error":"—"}.get(note, "—")
    st.info(f"{badge} Next Earnings: **{ed.date()}**  *(source: yfinance; note: {note})*", icon="🗓️")

# ─────────── Indicators / Composite ───────────
def compute_indicators(df: pd.DataFrame, ma_w: int, rsi_p: int, mf: int, ms: int, sig: int,
                       use_bb: bool = True) -> pd.DataFrame:
    d = df.copy()
    if d.empty or not set(["Open", "High", "Low", "Close"]).issubset(d.columns):
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

# ─────────── Backtest & Analytics ───────────
def perf_stats(d: pd.DataFrame, interval: str) -> dict:
    ann = 252 if interval == "1d" else 252*6
    ret = d["StratRet"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    bh  = d["Return"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    eq  = (1 + ret).cumprod()
    # Sharpe
    mu  = float(ret.mean() * ann)
    sig = float(ret.std(ddof=0) * math.sqrt(ann))
    sharpe = (mu / sig) if sig > 0 else np.nan
    # Sortino
    dn = ret.clip(upper=0)
    dn_std = float(dn.std(ddof=0) * math.sqrt(ann))
    sortino = (mu / dn_std) if dn_std > 0 else np.nan
    # Max DD
    dd = eq/eq.cummax() - 1.0
    max_dd = float(dd.min() * 100.0)
    # CAGR
    n_eff = int(ret.notna().sum())
    last_cum = float(eq.iloc[-1]) if len(eq) else 1.0
    cagr = ((last_cum ** (ann / max(n_eff,1))) - 1) * 100 if n_eff>0 else np.nan
    # MAR
    mar = (cagr / abs(max_dd)) if max_dd != 0 else np.nan
    # Win rate / trades / exposure
    win_rt = float((ret > 0).mean() * 100)
    pos_change = d["Position"].diff().fillna(0).abs()
    trades   = int((pos_change > 0).sum())
    exposure = float((d["Position"] != 0).mean() * 100)
    # Bh cum
    bh_eq = (1 + bh).cumprod()
    return dict(
        sharpe=sharpe, sortino=sortino, max_dd=max_dd, cagr=cagr, mar=mar,
        win_rt=win_rt, trades=trades, exposure=exposure,
        strat_last=last_cum, bh_last=float(bh_eq.iloc[-1]) if len(bh_eq) else 1.0
    )

def backtest(df: pd.DataFrame, *, allow_short=False, cost_bps=0.0,
             sl_atr_mult=0.0, tp_atr_mult=0.0, vol_target=0.0, interval="1d"):
    d = df.copy()
    if d.empty or "Close" not in d:
        sk = d.copy()
        for col in ["Return","Position","StratRet","CumBH","CumStrat"]:
            sk[col] = 0.0
        sk["CumBH"] = 1.0
        sk["CumStrat"] = 1.0
        return sk, perf_stats(sk.assign(Position=0, Return=0, StratRet=0, CumBH=1, CumStrat=1), interval)

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

    return d, perf_stats(d, interval)

def build_trade_log(df: pd.DataFrame) -> pd.DataFrame:
    """Extract trades from Position series (entry/exit, PnL)."""
    if df.empty or "Position" not in df or "Close" not in df:
        return pd.DataFrame()
    pos = df["Position"].fillna(0).astype(int)
    price = df["Close"]
    trades = []
    cur_side = 0
    entry_idx = None
    entry_price = None
    for i in range(len(df)):
        p = pos.iat[i]
        if cur_side == 0 and p != 0:
            cur_side = p
            entry_idx = df.index[i]
            entry_price = price.iat[i]
        elif cur_side != 0 and p == 0:
            exit_idx = df.index[i]
            exit_price = price.iat[i]
            ret = (exit_price - entry_price) / entry_price * (1 if cur_side>0 else -1)
            trades.append(dict(
                Entry=entry_idx, Exit=exit_idx, Side=("Long" if cur_side>0 else "Short"),
                EntryPx=float(entry_price), ExitPx=float(exit_price),
                ReturnPct=float(ret*100)
            ))
            cur_side = 0; entry_idx=None; entry_price=None
    return pd.DataFrame(trades)

def drawdowns_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "StratRet" not in df:
        return pd.DataFrame()
    eq = (1 + df["StratRet"].fillna(0)).cumprod()
    peak = eq.cummax()
    dd = (eq/peak - 1.0)
    in_dd = dd < 0
    rows=[]
    start=None
    for i, (ts, flag) in enumerate(in_dd.items()):
        if flag and start is None:
            start = ts
        if not flag and start is not None:
            end = dd.loc[start:ts].idxmin()
            depth = dd.loc[start:ts].min()
            recover = ts
            rows.append(dict(Start=start, Trough=end, End=recover, DepthPct=float(depth*100)))
            start=None
    return pd.DataFrame(rows)

# ───────────────────────────── ENGINE ─────────────────────────────
with tab_engine:
    st.title("🚀 QuantaraX — Composite Signal Engine (v17)")

    st.markdown("### Single‐Ticker Backtest")
    ticker = st.text_input("Ticker (e.g. AAPL or BTC/USDT)", "AAPL", key="inp_engine_ticker").upper()

    # Live Price & Data Health
    px_live = load_prices(ticker, "5d", "1d")
    if not px_live.empty and "Close" in px_live:
        last_px = _to_float(px_live["Close"].iloc[-1])
        st.subheader(f"💲 Live (last close): ${last_px:.2f}")
        meta = data_health(px_live, "1d")
        if meta.get("ok"):
            st.caption(
                f"Data {meta['rows']} rows • {meta['start'].date()} → {meta['end'].date()} • "
                f"freshness: {meta['fresh_hours']:.1f}h • gaps: {meta['gaps']}"
            )

    # Earnings (robust + override)
    render_next_earnings(ticker, manual_override=earn_override)

    # News (safe → RSS fallback)
    news = safe_get_news(ticker)
    if news:
        st.markdown("#### 📰 Recent News & Sentiment (YFinance)")
        shown = 0
        for art in news:
            t_ = art.get("title",""); l_ = art.get("link","")
            if not (t_ and l_): continue
            txt = art.get("summary", t_)
            score = analyzer.polarity_scores(txt)["compound"]
            emoji = "🔺" if score>0.1 else ("🔻" if score<-0.1 else "➖")
            st.markdown(f"- [{t_}]({l_}) {emoji}")
            shown += 1
            if shown >= 5: break
    else:
        rss = rss_news(ticker, limit=5)
        if rss:
            st.markdown("#### 📰 Recent News (RSS Fallback)")
            for r in rss:
                st.markdown(f"- [{r['title']}]({r['link']})")
        else:
            st.info("No recent news found.")

    run_btn = st.button("▶️ Run Composite Backtest", key="btn_engine_backtest")
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
        if df_sig.empty:
            st.error("Composite could not be built (insufficient rows)."); st.stop()

        df_c, stats = backtest(
            df_sig, allow_short=allow_short, cost_bps=cost_bps,
            sl_atr_mult=sl_atr_mult, tp_atr_mult=tp_atr_mult, vol_target=vol_target, interval=interval_sel
        )

        last_trade = int(df_sig["Trade"].tail(1).iloc[0]) if "Trade" in df_sig.columns and not df_sig.empty else 0
        rec = rec_map.get(1 if last_trade>0 else (-1 if last_trade<0 else 0), "🟡 HOLD")
        st.success(f"**{ticker}**: {rec}")

        # Reasoning + Playbook
        last = df_sig.tail(1).iloc[0]
        ma_s  = int(last.get("MA_Signal", 0))
        rsi_s = int(last.get("RSI_Signal", 0))
        macd_s= int(last.get("MACD_Signal2", 0))
        rsi_v = float(last.get(f"RSI{rsi_period}", np.nan))
        atr_v = float(last.get("ATR", np.nan))
        px_v  = float(last.get("Close", np.nan))

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

        # Beginner / Pro Playbook (position sizing)
        if mode == "Beginner":
            st.info("**Playbook (Beginner)** — If signal = BUY and you want exposure:\n"
                    f"- Consider buying small (e.g., 1–2% of capital) when price closes above MA{ma_window}.\n"
                    f"- Suggested stop: ~{sl_atr_mult:.1f}×ATR ({'n/a' if np.isnan(atr_v) else f'≈ ${atr_v*sl_atr_mult:.2f} below entry'})\n"
                    f"- Take profit idea: ~{tp_atr_mult:.1f}×ATR move.\n"
                    f"- Avoid holding through earnings unless you’re comfortable with gap risk.\n", icon="🧭")
        else:
            risk_usd = capital_usd * (risk_per_trade_pct/100.0)
            per_share_risk = atr_v*sl_atr_mult if not np.isnan(atr_v) and sl_atr_mult>0 else (0.02*px_v if not np.isnan(px_v) else np.nan)
            shares = (risk_usd / per_share_risk) if per_share_risk and per_share_risk>0 else 0
            st.info(f"**Playbook (Pro)** — Risk ${risk_usd:,.2f} ({risk_per_trade_pct:.1f}% of ${capital_usd:,.0f}). "
                    f"Est. per-share risk ≈ ${0 if np.isnan(per_share_risk) else per_share_risk:.2f} → "
                    f"size ≈ **{int(shares):,} shares**.\n"
                    f"- Entry filter: only take longs if daily+hourly composites agree (see MTF tool).\n"
                    f"- Stop/TP: {sl_atr_mult}×ATR / {tp_atr_mult}×ATR; trail after +1×ATR.\n"
                    f"- Scale out 50% at +1.5×ATR, run the rest to signal flip or earnings.", icon="🎯")

        # Metrics
        colA, colB, colC, colD, colE, colF = st.columns(6)
        colA.metric("CAGR",   f"{(0 if np.isnan(stats['cagr']) else stats['cagr']):.2f}%")
        colB.metric("Sharpe", f"{(0 if np.isnan(stats['sharpe']) else stats['sharpe']):.2f}")
        colC.metric("Sortino", f"{(0 if np.isnan(stats['sortino']) else stats['sortino']):.2f}")
        colD.metric("Max DD", f"{stats['max_dd']:.2f}%")
        colE.metric("Win Rate", f"{stats['win_rt']:.1f}%")
        colF.metric("MAR", f"{(0 if np.isnan(stats['mar']) else stats['mar']):.2f}")

        st.markdown(f"- **Buy & Hold:** {(stats['bh_last']-1)*100:.2f}%  \n- **Strategy:** {(stats['strat_last']-1)*100:.2f}%  \n- **Exposure:** {stats['exposure']:.1f}%  \n- **Trades:** {stats['trades']}")

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

        # Trade log & drawdowns
        with st.expander("📒 Trade Log & Drawdowns"):
            tl = build_trade_log(df_c)
            if not tl.empty:
                st.dataframe(tl, use_container_width=True)
                st.markdown(f"**Avg Trade Return:** {tl['ReturnPct'].mean():.2f}% | **Median:** {tl['ReturnPct'].median():.2f}% | **#Trades:** {len(tl)}")
            else:
                st.info("No completed trades to show.")
            dd = drawdowns_table(df_c)
            if not dd.empty:
                st.dataframe(dd, use_container_width=True)
            else:
                st.info("No drawdowns table available.")

    # Extra tools (safe)
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
                    daily  = float(c1["Composite"].iloc[-1])
                    hourly = float(cH["Composite"].iloc[-1])
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
                px = load_prices(wf_symbol, "2y", "1d")
                if px.empty:
                    st.warning("No data for WFO.")
                else:
                    def run_eq(
                        ma_list: List[int], rsi_list: List[int],
                        mf_list: List[int], ms_list: List[int], sig_list: List[int],
                        insample_bars: int, oos_bars: int,
                        w_ma=1.0, w_rsi=1.0, w_macd=1.0, w_bb=0.5, threshold=1.0,
                        allow_short=False, cost_bps=5.0
                    ):
                        oos_curves = []; summary = []
                        start=200; i=start
                        while i + insample_bars + oos_bars <= len(px):
                            ins = px.iloc[i : i+insample_bars]
                            oos = px.iloc[i+insample_bars : i+insample_bars+oos_bars]
                            best=None; best_score=-1e9
                            for mw in ma_list:
                                for rp in rsi_list:
                                    for mf in mf_list:
                                        for ms in ms_list:
                                            for s in sig_list:
                                                ins_ind = compute_indicators(ins, mw, rp, mf, ms, s, use_bb=True)
                                                if ins_ind.empty: continue
                                                ins_sig = build_composite(
                                                    ins_ind, mw, rp,
                                                    use_weighted=True, w_ma=w_ma, w_rsi=w_rsi, w_macd=w_macd, w_bb=w_bb,
                                                    include_bb=True, threshold=threshold, allow_short=allow_short
                                                )
                                                ins_bt, _ = backtest(ins_sig, allow_short=allow_short, cost_bps=cost_bps)
                                                perf = (ins_bt["CumStrat"].iloc[-1]-1)*100 if "CumStrat" in ins_bt else -1e9
                                                dd = ((ins_bt["CumStrat"]/ins_bt["CumStrat"].cummax())-1).min()*100
                                                score = perf - abs(dd)
                                                if score > best_score:
                                                    best_score = score
                                                    best = (mw, rp, mf, ms, s)
                            if best is None:
                                i += oos_bars; continue
                            mw, rp, mf, ms, s = best
                            oos_ind = compute_indicators(oos, mw, rp, mf, ms, s, use_bb=True)
                            if oos_ind.empty:
                                i += oos_bars; continue
                            oos_sig = build_composite(
                                oos_ind, mw, rp,
                                use_weighted=True, w_ma=w_ma, w_rsi=w_rsi, w_macd=w_macd, w_bb=w_bb,
                                include_bb=True, threshold=threshold, allow_short=allow_short
                            )
                            oos_bt, oos_stats = backtest(oos_sig, allow_short=allow_short, cost_bps=cost_bps)
                            if "CumStrat" in oos_bt:
                                oos_curves.append(oos_bt[["CumStrat"]].rename(columns={"CumStrat":"Equity"}))
                            summary.append({
                                "Window": f"{oos.index[0].date()} → {oos.index[-1].date()}",
                                "MA": mw, "RSI": rp, "MACDf": mf, "MACDs": ms, "SIG": s,
                                "OOS %": ((oos_bt["CumStrat"].iloc[-1]-1)*100) if "CumStrat" in oos_bt else np.nan,
                                "OOS Sharpe": oos_stats["sharpe"], "OOS MaxDD%": oos_stats["max_dd"]
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

    # Parameter Heatmap (quick robustness feel)
    with st.expander("🧭 Parameter Heatmap (mini grid)", expanded=False):
        ph_symbol = st.text_input("Symbol (Heatmap)", value=ticker or "AAPL", key="ph_sym")
        if st.button("Run Heatmap", key="btn_heatmap"):
            try:
                px = load_prices(ph_symbol, period_sel, interval_sel)
                if px.empty: st.warning("No data."); st.stop()
                ma_list  = [max(5, ma_window-5), ma_window, min(60, ma_window+5)]
                rsi_list = [max(5, rsi_period-7), rsi_period, min(30, rsi_period+7)]
                rows=[]
                for mw in ma_list:
                    for rp in rsi_list:
                        ind = compute_indicators(px, mw, rp, macd_fast, macd_slow, macd_signal, use_bb=include_bb)
                        if ind.empty: continue
                        sig = build_composite(ind, mw, rp, use_weighted=use_weighted, w_ma=w_ma, w_rsi=w_rsi, w_macd=w_macd, w_bb=w_bb,
                                              include_bb=include_bb, threshold=comp_thr, allow_short=allow_short)
                        bt, stt = backtest(sig, allow_short=allow_short, cost_bps=cost_bps,
                                           sl_atr_mult=sl_atr_mult, tp_atr_mult=tp_atr_mult,
                                           vol_target=vol_target, interval=interval_sel)
                        rows.append({"MA":mw, "RSI":rp, "CAGR%":stt["cagr"], "MaxDD%":stt["max_dd"], "Sharpe":stt["sharpe"]})
                if rows:
                    df_hm = pd.DataFrame(rows).pivot(index="MA", columns="RSI", values="CAGR%").round(2)
                    st.dataframe(df_hm, use_container_width=True)
                else:
                    st.info("No cells computed.")
            except Exception as e:
                st.error(f"Heatmap error: {e}")

    # Watchlist quick monitor
    st.markdown("---")
    with st.expander("👀 Watchlist Monitor (signals + earnings)"):
        wl = st.text_area("Tickers (comma‐separated)", "AAPL, MSFT, NVDA, TSLA, AMZN, SPY, QQQ", key="wl_ta").upper()
        if st.button("Scan Watchlist", key="btn_wl"):
            rows=[]
            for t in [x.strip() for x in wl.split(",") if x.strip()]:
                try:
                    px = load_prices(t, period_sel, interval_sel)
                    if px.empty: continue
                    ind = compute_indicators(px, ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=True)
                    sig = build_composite(ind, ma_window, rsi_period, use_weighted=use_weighted, w_ma=w_ma, w_rsi=w_rsi, w_macd=w_macd, w_bb=w_bb,
                                          include_bb=include_bb, threshold=comp_thr, allow_short=allow_short)
                    comp = float(sig["Composite"].iloc[-1]) if not sig.empty else 0.0
                    rec = rec_map.get(int(np.sign(comp)), "🟡 HOLD")
                    last = float(ind["Close"].iloc[-1]) if not ind.empty else np.nan
                    ed, note = next_earnings_date(t, manual_override=None)
                    rows.append({"Ticker":t, "Last":last, "Composite":comp, "Signal":rec, "NextEarnings":(ed.date() if ed else None), "E_Note":note})
                except Exception:
                    continue
            if rows:
                df = pd.DataFrame(rows).set_index("Ticker").sort_values(["Signal","Composite"], ascending=[True,False])
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No results.")

# ───────────────────────────── ML LAB ─────────────────────────────
with tab_ml:
    st.title("🧠 ML Lab — Probabilistic Signals")
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
            bt, stt = backtest(ml_df, allow_short=allow_short, cost_bps=cost_bps,
                               sl_atr_mult=sl_atr_mult, tp_atr_mult=tp_atr_mult,
                               vol_target=vol_target, interval=interval_sel)
            st.markdown(f"**ML Strategy OOS:** Return={(bt['CumStrat'].iloc[-1]-1)*100:.2f}% | Sharpe={stt['sharpe']:.2f} | MaxDD={stt['max_dd']:.2f}% | Trades={stt['trades']}")
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
                rec = rec_map.get(int(np.sign(comp)), "🟡 HOLD")
                mlp = np.nan
                if use_ml_scan and SKLEARN_OK:
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
            if SKLEARN_OK:
                from sklearn.cluster import KMeans
                km = KMeans(n_clusters=3, n_init=10, random_state=42)
                lab = km.fit_predict(feat)
            else:
                q1 = feat.rank(pct=True)
                lab = (q1.mean(axis=1) > 0.66).astype(int) + (q1.mean(axis=1) < 0.33).astype(int)*2
            reg = pd.Series(lab, index=feat.index, name="Regime")
            joined = ind.join(reg, how="right")
            ret = joined["Close"].pct_change().groupby(joined["Regime"]).mean().sort_values()
            ord_map = {old:i for i, old in enumerate(ret.index)}
            joined["Regime"] = joined["Regime"].map(ord_map)
            st.dataframe(joined[["Close","Regime"]].tail(10))
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(joined.index, joined["Close"], label="Close")
            for r in sorted(joined["Regime"].dropna().unique()):
                seg = joined[joined["Regime"]==r]
                ax.fill_between(seg.index, seg["Close"].min(), seg["Close"].max(), alpha=0.08)
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
            bt, _ = backtest(sig, allow_short=allow_short, cost_bps=cost_bps,
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
    st.header("How to Use QuantaraX Pro — Beginner → Pro (Investor-Ready)")
    st.markdown(r"""
### 0) What this app *is*
QuantaraX Pro fuses classic technical signals (MA/RSI/MACD/Bollinger) with a robust backtester, optional ML probabilities, multi-timeframe confirmation, and portfolio tools. It’s designed for:
- **Beginners**: a guided “Playbook” and simple traffic-light recommendations.
- **Pros**: full control of parameters, ATR-based position sizing, regime views, WFO, metrics (Sharpe, Sortino, MAR), trade logs, and risk parity.

---

### 1) The Composite Signal (Engine tab)
We compute:
- **MA crossover** (price vs. MA{N})
- **RSI** (oversold <30, overbought >70)
- **MACD cross** (MACD vs. signal)
- **Bollinger mean reversion** (optional)

Each module votes ±1 or 0; the **weighted composite** sums these. If the composite crosses your **threshold**, we set **Trade = 1 (long)** or **–1 (short)** if enabled (else 0).

**Why this matters:** multiple weak signals become strong when aligned. You can tune **weights** and **threshold**. Use the **MTF** tool to ensure alignment across daily/hourly bars.

---

### 2) Risk, Stops, and Position Sizing
- **ATR-based stops**: SL = *ATR × multiplier*, TP = *ATR × multiplier*. ATR scales with volatility.
- **Sizing** (Pro mode): define **capital** and **risk % per trade**. We estimate **shares ≈ (risk$)/(per-share risk)**. Per-share risk defaults to *ATR × SL-multiplier*.
- **Exposure control**: optional **volatility targeting** limits leverage when realized vol is high.

**For beginners**: Use small sizes (1–2% of capital) and avoid earnings risk unless you understand gaps.

---

### 3) Backtesting Stats (what they mean)
- **CAGR**: geometric annual growth.
- **Sharpe**: risk-adjusted returns vs. total volatility.
- **Sortino**: penalizes downside volatility only (cleaner than Sharpe for skewed returns).
- **Max Drawdown**: peak-to-trough pain.
- **MAR**: CAGR / |MaxDD| (capital efficiency).
- **Win Rate / Trades / Exposure**: execution profile.
Use the **Trade Log** to inspect entries/exits and **Drawdown Table** to see major selloffs.

---

### 4) Multi-Timeframe Confirmation (MTF)
We compute the composite on **daily** and **hourly**. If both have the same sign, you get “✅ agree.” Many pros only trade when MTF agrees to reduce whipsaw.

---

### 5) Walk-Forward Optimization (WFO)
WFO simulates a realistic process:
1) Optimize parameters on a rolling **in-sample** window,
2) Apply them to the next **out-of-sample** window,
3) Stitch OOS equity.
You want **stable** performance across windows, not just one perfect setting.

---

### 6) Scanner & Watchlist
- **Scanner**: run composite (and optional ML probability) across a universe to surface ideas quickly.
- **Watchlist Monitor**: shows **Last**, **Composite**, **Signal**, and **Next Earnings** (with a confidence note). Avoid initiating new positions right before earnings unless planned.

---

### 7) Portfolio Tools
- **Risk Parity**: equalize risk contributions across assets using covariance.
- **Monte Carlo (bootstrap)**: resample strategy daily returns to approximate outcome bands (P5/P25/Median/P75/P95).

---

### 8) Earnings Dates (accuracy & overrides)
We pull from yfinance’s `get_earnings_dates`, **normalize to UTC**, and apply sanity checks:
- If date looks too far out (>200 days) → **low-confidence**.
- If date is in the past by >5 days → **stale**.
- You can **override** dates in the sidebar (YYYY-MM-DD) for mission-critical tickers.

---

### 9) Suggested Workflows
**Beginner**
1. Pick a ticker → “Run Composite Backtest”.
2. Read the “Why this signal?” explainer.
3. In Beginner mode, follow the **Playbook** sizing & stops.
4. Confirm with **MTF** (prefer agreement).
5. Avoid earnings week unless deliberate.
6. Track with **Watchlist**.

**Pro**
1. Tune weights/thresholds, use **vol targeting** if desired.
2. Use **MTF** gating for entries.
3. Size via **ATR** & risk budget.
4. Validate robustness: **WFO** and **Parameter Heatmap**.
5. Build a **risk-parity** sleeve and run **Monte Carlo** on the strategy curve.

---

### 10) Good Hygiene / Caveats
- Data can be delayed or incomplete. We show **freshness** and **gaps**.
- ML is optional and should augment—not replace—risk management.
- Past performance ≠ future results. Use stops and size conservatively.

You're set. Beginners get a guided, plain-English Playbook; pros get sizing, WFO, MTF gating, and deeper stats. If you want broker integration, alerts, or order templates next, we can add those too.
""")
