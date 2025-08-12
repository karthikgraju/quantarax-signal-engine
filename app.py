# app.py — QuantaraX Pro (v13, "Insane Edition": options, sizing, risk, patterns)
# ---------------------------------------------------------------------------------
# pip install:
#   streamlit yfinance pandas numpy matplotlib feedparser vaderSentiment scikit-learn

import math
from typing import List, Tuple, Optional
import time
import warnings
from io import StringIO

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
st.set_page_config(page_title="QuantaraX Pro v13", layout="wide")
analyzer = SentimentIntensityAnalyzer()
rec_map = {1: "🟢 BUY", 0: "🟡 HOLD", -1: "🔴 SELL"}

# ───────────────────────────── Tabs ─────────────────────────────
TAB_TITLES = [
    "🚀 Engine", "🧠 ML Lab", "📡 Scanner",
    "📉 Regimes", "💼 Portfolio",
    "📈 Options", "🎯 Sizing", "🛡️ Risk", "🕯️ Patterns",
    "❓ Help"
]
(tab_engine, tab_ml, tab_scan, tab_regime, tab_port,
 tab_options, tab_sizing, tab_risk, tab_patterns, tab_help) = st.tabs(TAB_TITLES)

# ───────────────────────────── Sidebar (unique keys) ─────────────────────────────
st.sidebar.header("Global Controls")
DEFAULTS = dict(ma_window=10, rsi_period=14, macd_fast=12, macd_slow=26, macd_signal=9)
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

colA, colB = st.sidebar.columns([1,1])
with colA:
    mode = st.selectbox("Mode", ["Retail (simple)", "Investor (advanced)"], index=1, key="v13_mode")
with colB:
    benchmark = st.text_input("Benchmark", "SPY", key="v13_bmk").upper().strip() or "SPY"

if st.sidebar.button("🔄 Reset to defaults", key="v13_btn_reset_defaults"):
    for k, v in DEFAULTS.items():
        st.session_state[k] = v

st.sidebar.subheader("Indicator Parameters")
ma_window   = st.sidebar.slider("MA window",      5, 60, st.session_state["ma_window"],   key="v13_ma_window")
rsi_period  = st.sidebar.slider("RSI lookback",   5, 30, st.session_state["rsi_period"],  key="v13_rsi_period")
macd_fast   = st.sidebar.slider("MACD fast span", 5, 20, st.session_state["macd_fast"],   key="v13_macd_fast")
macd_slow   = st.sidebar.slider("MACD slow span", 20, 50, st.session_state["macd_slow"],  key="v13_macd_slow")
macd_signal = st.sidebar.slider("MACD sig span",  5, 20, st.session_state["macd_signal"], key="v13_macd_signal")

st.sidebar.subheader("Composite Preset")
preset = st.sidebar.selectbox(
    "Select style",
    ["Custom", "Trend (MA+MACD)", "Mean-Revert (RSI+BB)", "Balanced"],
    key="v13_preset",
)
use_weighted = st.sidebar.toggle("Use weighted composite", value=True, key="v13_use_weighted")
include_bb   = st.sidebar.toggle("Include Bollinger Bands", value=True, key="v13_include_bb")

def preset_weights(name:str):
    if name == "Trend (MA+MACD)":
        return dict(w_ma=1.2, w_rsi=0.5, w_macd=1.5, w_bb=0.2, thr=1.0)
    if name == "Mean-Revert (RSI+BB)":
        return dict(w_ma=0.5, w_rsi=1.4, w_macd=0.6, w_bb=1.2, thr=0.8)
    if name == "Balanced":
        return dict(w_ma=1.0, w_rsi=1.0, w_macd=1.0, w_bb=0.5, thr=1.0)
    return None

if preset != "Custom":
    P = preset_weights(preset)
    w_ma_default, w_rsi_default, w_macd_default, w_bb_default, thr_default = P["w_ma"], P["w_rsi"], P["w_macd"], P["w_bb"], P["thr"]
else:
    w_ma_default, w_rsi_default, w_macd_default, w_bb_default, thr_default = 1.0, 1.0, 1.0, 0.5, 1.0

w_ma   = st.sidebar.slider("Weight • MA",   0.0, 2.0, w_ma_default,   0.1, key="v13_w_ma")
w_rsi  = st.sidebar.slider("Weight • RSI",  0.0, 2.0, w_rsi_default,  0.1, key="v13_w_rsi")
w_macd = st.sidebar.slider("Weight • MACD", 0.0, 2.0, w_macd_default, 0.1, key="v13_w_macd")
w_bb   = st.sidebar.slider("Weight • BB",   0.0, 2.0, w_bb_default,   0.1, key="v13_w_bb") if include_bb else 0.0
comp_thr = st.sidebar.slider("Composite trigger (enter/exit)", 0.0, 3.0, thr_default, 0.1, key="v13_comp_thr")

st.sidebar.subheader("Risk & Costs")
allow_short = st.sidebar.toggle("Allow shorts", value=False, key="v13_allow_short")
cost_bps    = st.sidebar.slider("Trading cost (bps/side)", 0.0, 25.0, 5.0, 0.5, key="v13_cost_bps")
sl_atr_mult = st.sidebar.slider("Stop • ATR ×", 0.0, 5.0, 2.0, 0.1, key="v13_sl_atr_mult")
tp_atr_mult = st.sidebar.slider("Target • ATR ×", 0.0, 8.0, 3.0, 0.1, key="v13_tp_atr_mult")
vol_target  = st.sidebar.slider("Vol targeting (annual)", 0.0, 0.5, 0.0, 0.05, key="v13_vol_target")
earn_guard_days = st.sidebar.slider("Earnings guard (± days)", 0, 10, 0, key="v13_earn_guard")

st.sidebar.subheader("Data")
period_sel   = st.sidebar.selectbox("History", ["6mo","1y","2y","5y"], index=1, key="v13_period_sel")
interval_sel = st.sidebar.selectbox("Interval", ["1d","1h"], index=0, key="v13_interval_sel")

st.sidebar.subheader("Portfolio Guardrails")
profit_target = st.sidebar.slider("Profit target (%)", 1, 100, 10, key="v13_profit_target")
loss_limit    = st.sidebar.slider("Loss limit (%)",  1, 100, 5,  key="v13_loss_limit")

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
            return float(x.iloc[0])
        except Exception:
            return float("nan")

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

def safe_earnings(symbol: str) -> pd.DataFrame:
    """Return DataFrame with normalized 'earn_date' (UTC)."""
    try:
        cal = yf.Ticker(_map_symbol(symbol)).get_earnings_dates(limit=12)
        if isinstance(cal, pd.DataFrame) and not cal.empty:
            df = cal.copy()
            if isinstance(df.index, pd.DatetimeIndex) or (
                df.index.name and "earn" in str(df.index.name).lower() and "date" in str(df.index.name).lower()
            ):
                df = df.reset_index()
            date_col = None
            for c in df.columns:
                cl = c.lower().replace(" ", "")
                if "earn" in cl and "date" in cl:
                    date_col = c; break
            if date_col is None:
                for c in df.columns:
                    if pd.api.types.is_datetime64_any_dtype(df[c]):
                        date_col = c; break
            if date_col is None:
                date_col = df.columns[0]
            df = df.rename(columns={date_col: "earn_date"})
            df["earn_date"] = pd.to_datetime(df["earn_date"], errors="coerce", utc=True)
            cols = ["earn_date"] + [c for c in df.columns if c != "earn_date"]
            return df[cols].dropna(subset=["earn_date"])
    except Exception:
        pass
    return pd.DataFrame()

def next_earnings_date(symbol: str):
    """Return (date, row_dict) for the next earnings on/after today (UTC), else last known."""
    df = safe_earnings(symbol)
    if df.empty:
        return None
    today_utc = pd.Timestamp.now(tz="UTC").date()
    df = df.copy()
    df["date_only"] = df["earn_date"].dt.date
    future = df[df["date_only"] >= today_utc].sort_values("date_only")
    if not future.empty:
        row = future.iloc[0]
        return row["date_only"], row.to_dict()
    row = df.sort_values("date_only").iloc[-1]
    return row["date_only"], row.to_dict()

def render_next_earnings(symbol: str):
    nxt = next_earnings_date(symbol)
    if nxt is None:
        st.info("📅 Earnings: unavailable")
    else:
        dt, _row = nxt
        st.info(f"📅 Next Earnings (UTC date): **{dt}**")

def apply_earnings_guard(sig_df: pd.DataFrame, symbol: str, days: int) -> pd.DataFrame:
    """Zero out Trade within ±days around earnings dates."""
    if days <= 0 or sig_df.empty:
        return sig_df
    er = safe_earnings(symbol)
    if er.empty or "earn_date" not in er:
        return sig_df
    d = sig_df.copy()
    idx_dates = pd.to_datetime(d.index).tz_localize(None).date if hasattr(d.index, "tz") else pd.to_datetime(d.index).date
    mask = np.zeros(len(d), dtype=bool)
    edates = er["earn_date"].dt.tz_convert("UTC").dt.date if er["earn_date"].dt.tz is not None else er["earn_date"].dt.date
    for e in edates:
        lo = e - pd.Timedelta(days=days)
        hi = e + pd.Timedelta(days=days)
        mask |= (idx_dates >= lo) & (idx_dates <= hi)
    if "Trade" in d:
        d.loc[mask, "Trade"] = 0
    return d

# Analytics helpers
def rolling_sharpe(ret: pd.Series, window: int = 63, ann: int = 252) -> pd.Series:
    r = ret.rolling(window)
    mu = r.mean()
    sd = r.std(ddof=0)
    rs = (mu / sd.replace(0, np.nan)) * math.sqrt(ann)
    return rs

def drawdown_curve(eq: pd.Series) -> pd.Series:
    peak = eq.cummax().replace(0, np.nan)
    dd = eq/peak - 1
    return dd

def calc_beta_alpha(strategy_ret: pd.Series, bench_ret: pd.Series, ann: int = 252):
    df = pd.concat([strategy_ret, bench_ret], axis=1).dropna()
    if df.empty or df.iloc[:,1].std(ddof=0) == 0:
        return np.nan, np.nan
    x = df.iloc[:,1].values  # benchmark
    y = df.iloc[:,0].values  # strategy
    X = np.column_stack([np.ones_like(x), x])
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    alpha_daily, beta = coef[0], coef[1]
    alpha_ann = (1 + alpha_daily) ** ann - 1
    return beta, alpha_ann

def trade_ledger(df: pd.DataFrame) -> pd.DataFrame:
    """Create trade list from Position & StratRet."""
    if df.empty or "Position" not in df or "StratRet" not in df:
        return pd.DataFrame()
    pos = df["Position"].fillna(0).astype(int).values
    idx = df.index
    out = []
    start = None
    for i in range(len(df)):
        if pos[i] != 0 and start is None:
            start = i
        if (pos[i] == 0 and start is not None) or (i == len(df)-1 and start is not None):
            end = i if pos[i] == 0 else i
            segment = df.iloc[start:end+1]
            direction = int(np.sign(segment["Position"].iloc[0]))
            ret = (1 + segment["StratRet"].fillna(0)).prod() - 1
            out.append({
                "Entry": segment.index[0],
                "Exit": segment.index[-1],
                "Bars": len(segment),
                "Dir": "LONG" if direction>=0 else "SHORT",
                "Return %": ret*100.0
            })
            start = None
    return pd.DataFrame(out)

def profit_factor_and_kelly(trades: pd.Series, daily_ret: pd.Series):
    """trades: trade-level % returns (as decimals); daily_ret: strategy daily returns."""
    if trades is None or len(trades) == 0:
        return np.nan, np.nan
    g = trades[trades > 0].sum()
    l = -trades[trades < 0].sum()
    pf = (g / l) if l > 0 else np.nan
    mu = daily_ret.mean()
    var = daily_ret.var(ddof=0)
    kelly = (mu / var) if var and var > 0 else np.nan
    return pf, kelly

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
        scale = (vol_target / realized).clip(0, 3.0).fillna(0.0)
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

# ─────────── Quick ML for Engine (blending) ───────────
def _ml_features_from_indicators(ind: pd.DataFrame, rsi_period: int) -> pd.DataFrame:
    out = pd.DataFrame(index=ind.index)
    out["ret1"] = ind["Close"].pct_change()
    out["ret5"] = ind["Close"].pct_change(5)
    out["vol20"] = ind["Close"].pct_change().rolling(20).std()
    out["rsi"] = ind.get(f"RSI{rsi_period}", np.nan)
    out["macd"] = ind.get("MACD", np.nan)
    out["sto_k"] = ind.get("STO_K", np.nan)
    out["adx"] = ind.get("ADX", np.nan)
    if {"BB_U","BB_L"}.issubset(ind.columns):
        rng = (ind["BB_U"] - ind["BB_L"]).replace(0, np.nan)
        out["bb_pos"] = (ind["Close"] - ind["BB_L"]) / rng
    else:
        out["bb_pos"] = np.nan
    return out.dropna()

def quick_ml_signal(ind: pd.DataFrame, rsi_period: int, horizon: int = 1,
                    proba_enter: float = 0.55, proba_exit: float = 0.45) -> pd.Series:
    """Fast OOS-ish ML: 80/20 split → predict last 20%; earlier bars = 0."""
    if not SKLEARN_OK:
        return pd.Series(0, index=ind.index)
    X = _ml_features_from_indicators(ind, rsi_period)
    if X.empty or len(X) < 250:
        return pd.Series(0, index=ind.index)
    y = (ind["Close"].pct_change(horizon).shift(-horizon) > 0).reindex(X.index).astype(int)
    data = pd.concat([X, y.rename("y")], axis=1).dropna()
    split = int(len(data)*0.8)
    train, test = data.iloc[:split], data.iloc[split:]
    if train["y"].nunique() < 2:
        return pd.Series(0, index=ind.index)
    clf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=7, n_jobs=-1)
    clf.fit(train.drop(columns=["y"]), train["y"])
    proba = pd.Series(clf.predict_proba(test.drop(columns=["y"]))[:,1], index=test.index)
    sig = pd.Series(0, index=ind.index)
    sig.loc[proba.index] = np.where(proba >= proba_enter, 1, np.where(proba <= proba_exit, -1, 0))
    return sig.astype(int)

# ───────────────────────────── ENGINE ─────────────────────────────
with tab_engine:
    st.title("🚀 QuantaraX — Composite Signal Engine (v13)")
    st.caption("For day traders, swing traders, options traders, and portfolio builders — one cockpit, many modes.")

    st.markdown("### Single‐Ticker Backtest")
    ticker = st.text_input("Ticker (e.g. AAPL or BTC/USDT)", "AAPL", key="v13_inp_engine_ticker").upper()

    # Live Price
    px_live = load_prices(ticker, "5d", "1d")
    if not px_live.empty and "Close" in px_live:
        last_px = _to_float(px_live["Close"].iloc[-1])
        st.subheader(f"💲 Last Close: ${last_px:.2f}")

    # News
    news = safe_get_news(ticker)
    if news:
        st.markdown("#### 📰 Recent News & Sentiment")
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

    # Earnings (UTC-safe)
    def render_next_earnings(symbol: str):
        nxt = next_earnings_date(symbol)
        if nxt is None:
            st.info("📅 Earnings: unavailable")
        else:
            dt, _row = nxt
            st.info(f"📅 Next Earnings (UTC date): **{dt}**")
    render_next_earnings(ticker)

    # Signal source controls
    st.markdown("#### Signal Source")
    signal_src = st.radio("Pick signal type", ["Composite", "ML (quick)", "Blended"], horizontal=True, key="v13_signal_src")
    ml_enter = st.slider("ML: Enter if P(long) ≥", 0.50, 0.80, 0.55, 0.01, key="v13_ml_enter")
    ml_exit  = st.slider("ML: Enter short if P(long) ≤", 0.20, 0.50, 0.45, 0.01, key="v13_ml_exit")
    blend_w  = st.slider("Blend weight (Composite ↔ ML)", 0.0, 1.0, 0.6, 0.05, key="v13_blend_w")

    # Benchmark for factor stats
    px_bmk = load_prices(benchmark, period_sel, interval_sel)

    if st.button("▶️ Run Backtest", key="v13_btn_engine_backtest"):
        px = load_prices(ticker, period_sel, interval_sel)
        if px.empty:
            st.error(f"No data for '{ticker}'"); st.stop()

        df_ind = compute_indicators(px, ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=include_bb)
        if df_ind.empty:
            st.error("Not enough data after indicators (try longer period or smaller windows)."); st.stop()

        df_comp = build_composite(
            df_ind, ma_window, rsi_period,
            use_weighted=use_weighted, w_ma=w_ma, w_rsi=w_rsi, w_macd=w_macd, w_bb=w_bb,
            include_bb=include_bb, threshold=comp_thr, allow_short=allow_short
        )

        # Optional ML and blending
        df_sig = df_comp.copy()
        if signal_src in ("ML (quick)", "Blended"):
            if SKLEARN_OK:
                ml_sig = quick_ml_signal(df_ind, rsi_period, horizon=1, proba_enter=ml_enter, proba_exit=ml_exit)
                df_sig["ML_Trade"] = ml_sig.reindex(df_sig.index).fillna(0).astype(int)
                if signal_src == "ML (quick)":
                    df_sig["Trade"] = df_sig["ML_Trade"]
                else:
                    comp_vote = df_sig["Trade"].astype(float)
                    ml_vote   = df_sig["ML_Trade"].astype(float)
                    blended = np.sign(blend_w*comp_vote + (1.0-blend_w)*ml_vote)
                    df_sig["Trade"] = blended.astype(int)
            else:
                st.warning("scikit-learn not installed; falling back to Composite.")
        # Earnings guard
        df_sig = apply_earnings_guard(df_sig, ticker, earn_guard_days)

        df_c, max_dd, sharpe, win_rt, trades, tim, cagr = backtest(
            df_sig, allow_short=allow_short, cost_bps=cost_bps,
            sl_atr_mult=sl_atr_mult, tp_atr_mult=tp_atr_mult, vol_target=vol_target, interval=interval_sel
        )

        last_trade = int(df_sig["Trade"].tail(1).iloc[0]) if "Trade" in df_sig.columns and not df_sig.empty else 0
        rec = rec_map.get(1 if last_trade>0 else (-1 if last_trade<0 else 0), "🟡 HOLD")
        st.success(f"**{ticker}**: {rec}  ({signal_src})")

        # Reasoning
        last = df_comp.tail(1).iloc[0]
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
            if include_bb and "BB_Signal" in df_comp.columns:
                bb_s = int(last.get("BB_Signal", 0))
                bb_txt = {1:"Close under lower band (mean-revert long).",0:"Inside bands.",-1:"Close over upper band (mean-revert short)."}[bb_s]
                st.write(f"- **BB:** {bb_txt}")
            st.write(f"- **Composite (weighted):** {float(last.get('Composite', 0)):.2f}  (threshold={comp_thr:.1f})")
            if signal_src in ("ML (quick)", "Blended") and "ML_Trade" in df_sig:
                st.write("- **ML:** RandomForest on technical factors (80/20 split). For strict OOS, use the ML tab.")

        # Performance Cards
        bh_last    = float(df_c["CumBH"].tail(1).iloc[0])  if "CumBH" in df_c and not df_c["CumBH"].empty else 1.0
        strat_last = float(df_c["CumStrat"].tail(1).iloc[0]) if "CumStrat" in df_c and not df_c["CumStrat"].empty else 1.0
        cA, cB, cC, cD, cE, cF = st.columns(6)
        cA.metric("CAGR", f"{(0 if np.isnan(cagr) else cagr):.2f}%")
        cB.metric("Sharpe", f"{(0 if np.isnan(sharpe) else sharpe):.2f}")
        cC.metric("Max DD", f"{max_dd:.2f}%")
        cD.metric("Win Rate", f"{win_rt:.1f}%")
        cE.metric("Trades", f"{trades}")
        cF.metric("Time in Mkt", f"{tim:.1f}%")

        st.markdown(f"- **Buy & Hold:** {(bh_last-1)*100:.2f}%  \n- **Strategy:** {(strat_last-1)*100:.2f}%")

        # Investor mode extras
        if mode == "Investor (advanced)":
            st.markdown("#### 📊 Advanced Analytics")
            px_bmk = load_prices(benchmark, period_sel, interval_sel)
            if not px_bmk.empty:
                df_align = pd.DataFrame(index=df_c.index)
                df_align["strat"] = df_c["StratRet"].fillna(0.0)
                df_align["bmk"]   = px_bmk["Close"].pct_change().reindex(df_align.index).fillna(0.0)
                beta, alpha_ann = calc_beta_alpha(df_align["strat"], df_align["bmk"])
                c1, c2 = st.columns(2)
                c1.metric("Beta vs " + benchmark, f"{(0 if np.isnan(beta) else beta):.2f}")
                c2.metric("Alpha (ann.)", f"{(0 if np.isnan(alpha_ann) else alpha_ann*100):.2f}%")
            else:
                st.info("Benchmark unavailable; factor stats skipped.")

            rs = rolling_sharpe(df_c["StratRet"].replace([np.inf,-np.inf], np.nan).fillna(0.0))
            fig_rs, ax_rs = plt.subplots(figsize=(9,2.8))
            ax_rs.plot(rs.index, rs.values); ax_rs.set_title("Rolling Sharpe (63 bars)"); ax_rs.axhline(0, ls="--", alpha=0.5)
            st.pyplot(fig_rs)

            dd = drawdown_curve(df_c["CumStrat"].replace(0, np.nan).fillna(method="ffill"))
            fig_dd, ax_dd = plt.subplots(figsize=(9,2.8))
            ax_dd.fill_between(dd.index, dd.values, 0, step=None, alpha=0.5)
            ax_dd.set_title("Drawdown (%)"); ax_dd.set_ylim(dd.min()*1.1, 0)
            st.pyplot(fig_dd)

            ledger = trade_ledger(df_c)
            if not ledger.empty:
                st.dataframe(ledger.tail(15), use_container_width=True)
                trade_rets = ledger["Return %"].astype(float)/100.0
                pf, kelly = profit_factor_and_kelly(trade_rets, df_c["StratRet"].fillna(0.0))
                c1, c2 = st.columns(2)
                c1.metric("Profit Factor", f"{(0 if np.isnan(pf) else pf):.2f}")
                c2.metric("Kelly (approx)", f"{(0 if np.isnan(kelly) else kelly):.2f}")
                st.download_button("⬇️ Export Trades (CSV)", ledger.to_csv(index=False), "trades.csv", key="v13_dl_trades")
            else:
                st.info("No closed trades to list yet.")

            eq = df_c[["CumBH","CumStrat"]].copy()
            st.download_button("⬇️ Export Equity (CSV)", eq.to_csv(), "equity.csv", key="v13_dl_equity")

        # Core Plots
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

        # —— Seasonality —
        with st.expander("📆 Seasonality (Month-of-year returns)"):
            r = px["Close"].pct_change().dropna()
            if not r.empty:
                mo = r.groupby(px.index.month).mean()*100
                figm, axm = plt.subplots(figsize=(8,2.6))
                axm.bar(mo.index.astype(int), mo.values)
                axm.set_xticks(range(1,13)); axm.set_title("Average Monthly Return (%)")
                st.pyplot(figm)
            else:
                st.info("Not enough data for seasonality.")

        # —— Parameter Surface —
        with st.expander("🗺️ Parameter Surface (MA × RSI)"):
            s1, s2, s3 = st.columns(3)
            with s1:
                ma_min = st.number_input("MA min", 5, 60, max(5, ma_window-5), key="v13_surf_mamin")
                ma_max = st.number_input("MA max", 6, 60, min(60, ma_window+5), key="v13_surf_mamax")
            with s2:
                rsi_min = st.number_input("RSI min", 5, 30, max(5, rsi_period-5), key="v13_surf_rsimin")
                rsi_max = st.number_input("RSI max", 6, 30, min(30, rsi_period+5), key="v13_surf_rsimax")
            with s3:
                steps = st.slider("Steps per axis", 3, 10, 5, key="v13_surf_steps")
            if st.button("Plot Surface", key="v13_btn_surf"):
                ma_vals  = np.linspace(ma_min, ma_max, steps, dtype=int)
                rsi_vals = np.linspace(rsi_min, rsi_max, steps, dtype=int)
                Z = np.full((len(rsi_vals), len(ma_vals)), np.nan)
                for i, rp in enumerate(rsi_vals):
                    for j, mw in enumerate(ma_vals):
                        ind_ = compute_indicators(px, mw, rp, macd_fast, macd_slow, macd_signal, use_bb=include_bb)
                        if ind_.empty: continue
                        sig_ = build_composite(
                            ind_, mw, rp,
                            use_weighted=use_weighted, w_ma=w_ma, w_rsi=w_rsi, w_macd=w_macd, w_bb=w_bb,
                            include_bb=include_bb, threshold=comp_thr, allow_short=allow_short
                        )
                        sig_ = apply_earnings_guard(sig_, ticker, earn_guard_days)
                        bt_, *_ = backtest(sig_, allow_short=allow_short, cost_bps=cost_bps, sl_atr_mult=sl_atr_mult, tp_atr_mult=tp_atr_mult, vol_target=vol_target, interval=interval_sel)
                        Z[i,j] = (bt_["CumStrat"].iloc[-1]-1)*100 if "CumStrat" in bt_ else np.nan
                figS, axS = plt.subplots(figsize=(6.5,4.2))
                im = axS.imshow(Z, origin="lower", aspect="auto", extent=[ma_vals[0], ma_vals[-1], rsi_vals[0], rsi_vals[-1]])
                axS.set_xlabel("MA window"); axS.set_ylabel("RSI period"); axS.set_title("Strategy % vs (MA, RSI)")
                plt.colorbar(im, ax=axS, label="% Return")
                st.pyplot(figS)

        # —— Stress Tester —
        with st.expander("🧨 Stress Test (gap + vol scaling)"):
            shock_pct = st.slider("Single-day gap shock (%)", -20.0, 20.0, -8.0, 0.5, key="v13_stress_gap")
            scale_win = st.slider("Scale volatility last N bars", 20, 200, 60, 5, key="v13_stress_window")
            vol_scale = st.slider("Vol scale factor", 0.5, 3.0, 1.8, 0.1, key="v13_stress_scale")
            if st.button("Run Stress", key="v13_btn_stress"):
                r = df_c["StratRet"].fillna(0).copy()
                if len(r) > scale_win+5:
                    r2 = r.copy()
                    r2.iloc[-scale_win:] = r2.iloc[-scale_win:] * vol_scale
                    r2.iloc[-scale_win]  = r2.iloc[-scale_win] + shock_pct/100.0
                    eq_base = (1+r).cumprod()
                    eq_stress = (1+r2).cumprod()
                    figT, axT = plt.subplots(figsize=(8,3))
                    axT.plot(eq_base.index, eq_base.values, label="Base")
                    axT.plot(eq_stress.index, eq_stress.values, label="Stress")
                    axT.legend(); axT.set_title("Stress Test — Equity")
                    st.pyplot(figT)
                else:
                    st.info("Not enough data for stress test.")

        # —— One-Pager —
        with st.expander("🧾 Export One-Pager"):
            notes = st.text_area("Add notes (optional)", "", key="v13_notes")
            if st.button("Download One-Pager (Markdown)", key="v13_btn_onepager"):
                btxt = StringIO()
                print(f"# QuantaraX One-Pager — {ticker}", file=btxt)
                print(f"\n**Mode:** {mode} | **Signal:** {signal_src}", file=btxt)
                print(f"\n**Settings**", file=btxt)
                print(f"- MA={ma_window}, RSI={rsi_period}, MACD=({macd_fast},{macd_slow},{macd_signal}), BB={include_bb}", file=btxt)
                print(f"- Weights: MA={w_ma}, RSI={w_rsi}, MACD={w_macd}, BB={w_bb}, Threshold={comp_thr}", file=btxt)
                print(f"- Risk: allow_short={allow_short}, cost_bps={cost_bps}, ATR SL={sl_atr_mult}, ATR TP={tp_atr_mult}, vol_target={vol_target}", file=btxt)
                print(f"- Earnings guard: ±{earn_guard_days}d", file=btxt)
                print("\n**Key Metrics**", file=btxt)
                print(f"- CAGR: {(0 if np.isnan(cagr) else cagr):.2f}%", file=btxt)
                print(f"- Sharpe: {(0 if np.isnan(sharpe) else sharpe):.2f}", file=btxt)
                print(f"- Max Drawdown: {max_dd:.2f}%", file=btxt)
                print(f"- Win Rate: {win_rt:.1f}%", file=btxt)
                print(f"- Trades: {trades} | Time in Market: {tim:.1f}%", file=btxt)
                print(f"- Strategy Return: {(strat_last-1)*100:.2f}% | Buy&Hold: {(bh_last-1)*100:.2f}%", file=btxt)
                nxt = next_earnings_date(ticker)
                if nxt:
                    print(f"\n**Next Earnings (UTC):** {nxt[0]}", file=btxt)
                if notes:
                    print("\n**Notes**", file=btxt)
                    print(notes, file=btxt)
                md = btxt.getvalue()
                st.download_button("⬇️ Save .md", md, f"quantarax_{ticker}_onepager.md", key="v13_dl_onepager")

    # ── Tools: MTF & WFO (kept, with unique keys) ──
    st.markdown("---")
    with st.expander("⏱️ Multi-Timeframe Confirmation", expanded=False):
        mtf_symbol = st.text_input("Symbol (MTF)", value=ticker or "AAPL", key="v13_inp_mtf_symbol").upper()
        if st.button("🔍 Check MTF", key="v13_btn_mtf"):
            try:
                d1 = compute_indicators(
                    load_prices(mtf_symbol, "1y", "1d"),
                    ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=True
                )
                dH = compute_indicators(
                    load_prices(mtf_symbol, "30d", "1h"),
                    ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=True
                )
                if d1.empty or dH.empty:
                    st.warning("Insufficient data for MTF."); st.stop()
                c1 = build_composite(
                    d1, ma_window, rsi_period,
                    use_weighted=True, w_ma=1.0, w_rsi=1.0, w_macd=1.0, w_bb=0.5,
                    include_bb=True, threshold=1.0, allow_short=allow_short
                )
                cH = build_composite(
                    dH, ma_window, rsi_period,
                    use_weighted=True, w_ma=1.0, w_rsi=1.0, w_macd=1.0, w_bb=0.5,
                    include_bb=True, threshold=1.0, allow_short=allow_short
                )
                daily  = float(c1["Composite"].iloc[-1]); hourly = float(cH["Composite"].iloc[-1])
                st.write(f"**Daily composite:** {daily:+.2f}")
                st.write(f"**Hourly composite:** {hourly:+.2f}")
                if np.sign(daily) == np.sign(hourly):
                    st.success("✅ Signals agree")
                else:
                    st.warning("⚠️ Signals disagree")
            except Exception as e:
                st.error(f"MTF error: {e}")

    with st.expander("🧪 Walk-Forward Optimization (OOS)", expanded=False):
        wf_symbol = st.text_input("Symbol (WFO)", value=ticker or "AAPL", key="v13_inp_wfo_symbol").upper()
        c1c, c2c = st.columns(2)
        with c1c:
            ins_bars = st.number_input("In-sample bars", 60, 252*3, 126, 1, key="v13_wfo_ins")
            oos_bars = st.number_input("OOS bars", 20, 252, 63, 1, key="v13_wfo_oos")
        with c2c:
            w_thr = st.slider("Composite trigger (WFO)", 0.0, 3.0, 1.0, 0.1, key="v13_wfo_thr")
            wf_allow_short = st.toggle("Allow shorts (WFO)", value=False, key="v13_wfo_short")
        if st.button("🏃 Run Walk-Forward", key="v13_btn_wfo"):
            try:
                px_all = load_prices(wf_symbol, "2y", "1d")
                if px_all.empty: st.warning("No data for WFO."); st.stop()

                def run_eq(ma_list: List[int], rsi_list: List[int],
                           mf_list: List[int], ms_list: List[int], sig_list: List[int],
                           insample_bars: int, oos_bars: int,
                           w_ma=1.0, w_rsi=1.0, w_macd=1.0, w_bb=0.5, threshold=1.0,
                           allow_short=False, cost_bps=5.0):
                    oos_curves = []; summary = []
                    start=200; i=start
                    while i + insample_bars + oos_bars <= len(px_all):
                        ins = px_all.iloc[i : i+insample_bars]
                        oos = px_all.iloc[i+insample_bars : i+insample_bars+oos_bars]
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
                                            ins_bt, md, sh, *_ = backtest(ins_sig, allow_short=allow_short, cost_bps=cost_bps)
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
                        oos_sig = build_composite(
                            oos_ind, mw, rp,
                            use_weighted=True, w_ma=w_ma, w_rsi=w_rsi, w_macd=w_macd, w_bb=w_bb,
                            include_bb=True, threshold=threshold, allow_short=allow_short
                        )
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
    batch = st.text_area("Tickers (comma-separated)", "AAPL, MSFT, TSLA, SPY, QQQ", key="v13_ta_batch").upper()
    if st.button("▶️ Run Batch Backtest", key="v13_btn_batch"):
        perf=[]
        for t in [x.strip() for x in batch.split(",") if x.strip()]:
            px = load_prices(t, period_sel, interval_sel)
            if px.empty: continue
            df_t = compute_indicators(px, ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=include_bb)
            if df_t.empty: continue
            df_tc = build_composite(df_t, ma_window, rsi_period,
                                    use_weighted=use_weighted, w_ma=w_ma, w_rsi=w_rsi, w_macd=w_macd, w_bb=w_bb,
                                    include_bb=include_bb, threshold=comp_thr, allow_short=allow_short)
            df_tc = apply_earnings_guard(df_tc, t, earn_guard_days)
            bt, md, sh, wr, trd, tim, cagr = backtest(df_tc, allow_short=allow_short, cost_bps=cost_bps,
                                                      sl_atr_mult=sl_atr_mult, tp_atr_mult=tp_atr_mult,
                                                      vol_target=vol_target, interval=interval_sel)
            comp_last = float(bt["Composite"].tail(1).iloc[0]) if "Composite" in bt and not bt["Composite"].empty else 0.0
            bh_last = float(bt["CumBH"].tail(1).iloc[0]) if "CumBH" in bt and not bt["CumBH"].empty else 1.0
            strat_last = float(bt["CumStrat"].tail(1).iloc[0]) if "CumStrat" in bt and not bt["CumStrat"].empty else 1.0
            perf.append({
                "Ticker":t,
                "Composite":comp_last,
                "Signal": rec_map.get(int(np.sign(comp_last)), "🟡 HOLD"),
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
            st.download_button("⬇️ Download CSV", df_perf.to_csv(), "batch.csv", key="v13_dl_batch")
        else:
            st.error("No valid data for batch tickers.")

# ───────────────────────────── OPTIONS DESK ─────────────────────────────
with tab_options:
    st.title("📈 Options Desk — Chains, Greeks, & Payoffs")
    opt_sym = st.text_input("Underlying", "AAPL", key="v13_opt_sym").upper()
    px = load_prices(opt_sym, "1mo", "1d")
    spot = float(px["Close"].iloc[-1]) if not px.empty else np.nan
    st.write(f"Spot: {spot:.2f}" if not np.isnan(spot) else "Spot unavailable")

    def load_chain(sym: str):
        try:
            tk = yf.Ticker(_map_symbol(sym))
            exps = tk.options or []
            if not exps:
                return [], None, None
            sel_exp = st.selectbox("Expiration", exps, index=0, key="v13_opt_exp")
            ch = tk.option_chain(sel_exp)
            calls = ch.calls.copy() if hasattr(ch, "calls") else pd.DataFrame()
            puts  = ch.puts.copy()  if hasattr(ch, "puts")  else pd.DataFrame()
            return exps, calls, puts
        except Exception:
            return [], None, None

    exps, calls, puts = load_chain(opt_sym)
    if calls is None or puts is None or calls.empty and puts.empty:
        st.info("Option chain unavailable.")
    else:
        def clean_chain(df):
            if df is None or df.empty:
                return pd.DataFrame()
            out = df.copy()
            # robust mid
            if {"bid","ask"}.issubset(out.columns):
                out["mid"] = (out["bid"].fillna(0) + out["ask"].fillna(0))/2
            elif "lastPrice" in out:
                out["mid"] = out["lastPrice"]
            else:
                out["mid"] = np.nan
            # implied vol column name
            ivc = None
            for c in out.columns:
                if "implied" in c.lower() and "vol" in c.lower():
                    ivc = c
            if ivc is not None:
                out["iv"] = out[ivc]
            else:
                out["iv"] = np.nan
            return out

        calls = clean_chain(calls)
        puts  = clean_chain(puts)

        st.subheader("Near-the-money")
        def near_atm(df, S):
            if df.empty or np.isnan(S): return df
            return df.iloc[(df["strike"]-S).abs().argsort()[:10]].sort_values("strike")
        c_sub = near_atm(calls, spot)
        p_sub = near_atm(puts, spot)
        col1, col2 = st.columns(2)
        with col1: st.dataframe(c_sub[["contractSymbol","strike","mid","iv","volume","openInterest"]].reset_index(drop=True), use_container_width=True)
        with col2: st.dataframe(p_sub[["contractSymbol","strike","mid","iv","volume","openInterest"]].reset_index(drop=True), use_container_width=True)

        st.subheader("Strategy Payoff (Expiration)")
        strat = st.selectbox("Strategy",
                             ["Long Call", "Long Put", "Vertical Call (Debit)", "Vertical Put (Debit)",
                              "Long Strangle", "Covered Call"],
                             key="v13_opt_strat")
        K1 = st.number_input("K1 (lower strike)", value=float(np.nan if np.isnan(spot) else round(spot)), step=1.0, key="v13_opt_k1")
        K2 = st.number_input("K2 (upper strike)", value=float(np.nan if np.isnan(spot) else round(spot*1.05)), step=1.0, key="v13_opt_k2")
        prem1 = st.number_input("Premium 1 ($)", value=0.0, step=0.01, key="v13_opt_p1")
        prem2 = st.number_input("Premium 2 ($)", value=0.0, step=0.01, key="v13_opt_p2")

        def payoff_grid(S0, strat, K1, K2, p1, p2):
            Ss = np.linspace(0.6*S0, 1.4*S0, 200) if S0 and S0>0 else np.linspace(50, 200, 200)
            P = np.zeros_like(Ss)
            if strat == "Long Call":
                P = np.maximum(Ss-K1, 0) - p1
            elif strat == "Long Put":
                P = np.maximum(K1-Ss, 0) - p1
            elif strat == "Vertical Call (Debit)":
                P = (np.maximum(Ss-K1, 0) - np.maximum(Ss-K2, 0)) - (p1 - p2)
            elif strat == "Vertical Put (Debit)":
                P = (np.maximum(K1-Ss, 0) - np.maximum(K2-Ss, 0)) - (p1 - p2)
            elif strat == "Long Strangle":
                P = np.maximum(Ss-K2, 0) + np.maximum(K1-Ss, 0) - (p1 + p2)
            elif strat == "Covered Call":
                # Long stock + short call @K1 for p1 received (so p1 is credit)
                P = (Ss - S0) - np.maximum(Ss-K1, 0) + p1
            return Ss, P

        if st.button("Plot Payoff", key="v13_opt_plot"):
            Ss, P = payoff_grid(spot, strat, K1, K2, prem1, prem2)
            figP, axP = plt.subplots(figsize=(8,3))
            axP.plot(Ss, P); axP.axhline(0, ls="--", alpha=0.6)
            axP.set_title(f"{strat} payoff at expiration")
            axP.set_xlabel("Underlying Price at Expiry"); axP.set_ylabel("P/L per 1x")
            st.pyplot(figP)

# ───────────────────────────── SIZING ─────────────────────────────
with tab_sizing:
    st.title("🎯 Position Sizing — Risk-first")
    sz_sym = st.text_input("Symbol", "AAPL", key="v13_sz_sym").upper()
    acct = st.number_input("Account Equity ($)", 10000.0, step=100.0, key="v13_sz_equity")
    risk_pct = st.slider("Risk per trade (%)", 0.1, 5.0, 1.0, 0.1, key="v13_sz_riskpct")
    entry = st.number_input("Entry ($)", 100.0, step=0.1, key="v13_sz_entry")
    stop  = st.number_input("Stop ($)", 95.0, step=0.1, key="v13_sz_stop")
    atr_mult = st.slider("OR ATR stop (× ATR)", 0.0, 5.0, 0.0, 0.1, key="v13_sz_atr_mult")

    use_atr = atr_mult > 0
    if st.button("Calculate Size", key="v13_sz_calc"):
        px = load_prices(sz_sym, "6mo", "1d")
        if px.empty:
            st.error("No data."); 
        else:
            ind = compute_indicators(px, ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=False)
            atr = ind["ATR"].iloc[-1] if "ATR" in ind else np.nan
            eff_stop = stop
            if use_atr and not np.isnan(atr):
                eff_stop = entry - atr_mult*atr
            risk_per_share = abs(entry - eff_stop)
            risk_dollars = acct * (risk_pct/100.0)
            qty = math.floor(risk_dollars / risk_per_share) if risk_per_share>0 else 0
            st.metric("ATR (14)", f"{(0 if np.isnan(atr) else atr):.2f}")
            st.metric("Position Size (shares)", f"{qty}")
            st.metric("Risk for trade ($)", f"{risk_dollars:.2f}")

# ───────────────────────────── RISK HUB ─────────────────────────────
with tab_risk:
    st.title("🛡️ Risk Hub — VaR / CVaR / Ruin")
    rk_sym = st.text_input("Symbol", "AAPL", key="v13_risk_sym").upper()
    if st.button("Run Risk", key="v13_risk_run"):
        px = load_prices(rk_sym, period_sel, interval_sel)
        if px.empty:
            st.error("No data.")
        else:
            ind = compute_indicators(px, ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=True)
            sig = build_composite(ind, ma_window, rsi_period,
                                  use_weighted=use_weighted, w_ma=w_ma, w_rsi=w_rsi, w_macd=w_macd, w_bb=w_bb,
                                  include_bb=include_bb, threshold=comp_thr, allow_short=allow_short)
            bt, *_ = backtest(sig, allow_short=allow_short, cost_bps=cost_bps,
                               sl_atr_mult=sl_atr_mult, tp_atr_mult=tp_atr_mult,
                               vol_target=vol_target, interval=interval_sel)
            r = bt["StratRet"].dropna()
            if r.empty:
                st.info("Not enough returns.")
            else:
                q = st.slider("Confidence (VaR/CVaR)", 0.80, 0.99, 0.95, 0.01, key="v13_risk_q")
                cutoff = np.quantile(r, 1-q)
                cvar = r[r <= cutoff].mean()
                st.metric("VaR (daily, %)", f"{cutoff*100:.2f}")
                st.metric("CVaR (daily, %)", f"{cvar*100:.2f}")

                # DoW/HOD
                st.subheader("Seasonality — Day of Week")
                dow = r.groupby(bt.index.dayofweek).mean()*100
                figD, axD = plt.subplots(figsize=(6,2.6))
                axD.bar(["Mon","Tue","Wed","Thu","Fri"], [dow.get(i, np.nan) for i in range(5)])
                axD.set_title("Avg. Strategy Return by Day (%)")
                st.pyplot(figD)

                # Risk of Ruin via bootstrap (prob reaching -50% peak drawdown path)
                st.subheader("Risk of Ruin (bootstrap)")
                N = len(r)
                trials = 1000
                ruin = 0
                for _ in range(trials):
                    samp = np.random.choice(r.values, size=N, replace=True)
                    eq = (1 + pd.Series(samp)).cumprod()
                    dd = eq/eq.cummax() - 1
                    if dd.min() <= -0.5: ruin += 1
                st.metric("Prob(≥50% DD) approx", f"{ruin/trials*100:.1f}%")

# ───────────────────────────── PATTERN RADAR ─────────────────────────────
with tab_patterns:
    st.title("🕯️ Pattern Radar — Candles & Breakouts")
    pr_sym = st.text_input("Symbol", "AAPL", key="v13_pr_sym").upper()
    lookback = st.slider("Lookback bars", 50, 300, 120, 10, key="v13_pr_lb")
    if st.button("Scan", key="v13_pr_scan"):
        px = load_prices(pr_sym, "2y", "1d")
        if px.empty:
            st.error("No data.")
        else:
            df = px.copy()
            df = df.tail(int(lookback)+50)
            O,H,L,C = df["Open"], df["High"], df["Low"], df["Close"]

            def is_hammer(i):
                body = abs(C.iloc[i]-O.iloc[i])
                lower = O.iloc[i] if C.iloc[i]>=O.iloc[i] else C.iloc[i]
                upper = C.iloc[i] if C.iloc[i]>=O.iloc[i] else O.iloc[i]
                wick_low = lower - L.iloc[i]
                wick_up  = H.iloc[i] - upper
                return wick_low > 2*body and wick_up < body
            def is_shooting_star(i):
                body = abs(C.iloc[i]-O.iloc[i])
                lower = O.iloc[i] if C.iloc[i]>=O.iloc[i] else C.iloc[i]
                upper = C.iloc[i] if C.iloc[i]>=O.iloc[i] else O.iloc[i]
                wick_low = lower - L.iloc[i]
                wick_up  = H.iloc[i] - upper
                return wick_up > 2*body and wick_low < body
            def engulfing_bull(i):
                if i==0: return False
                return (C.iloc[i] > O.iloc[i] and C.iloc[i-1] < O.iloc[i-1] and
                        O.iloc[i] < C.iloc[i-1] and C.iloc[i] > O.iloc[i-1])
            def engulfing_bear(i):
                if i==0: return False
                return (C.iloc[i] < O.iloc[i] and C.iloc[i-1] > O.iloc[i-1] and
                        O.iloc[i] > C.iloc[i-1] and C.iloc[i] < O.iloc[i-1])
            def is_doji(i):
                body = abs(C.iloc[i]-O.iloc[i])
                rng = H.iloc[i]-L.iloc[i]
                return body < 0.1*rng if rng>0 else False

            pat = []
            for i in range(len(df)):
                lab=[]
                if is_hammer(i): lab.append("Hammer")
                if is_shooting_star(i): lab.append("ShootingStar")
                if engulfing_bull(i): lab.append("BullEngulf")
                if engulfing_bear(i): lab.append("BearEngulf")
                if is_doji(i): lab.append("Doji")
                pat.append(",".join(lab))
            out = df.copy()
            out["Patterns"] = pat
            # Breakouts
            out["BO20"] = (C > C.rolling(20).max().shift(1)).astype(int)
            out["BD20"] = (C < C.rolling(20).min().shift(1)).astype(int)
            st.dataframe(out[["Open","High","Low","Close","Patterns","BO20","BD20"]].tail(30), use_container_width=True)

# ───────────────────────────── ML LAB (kept) ─────────────────────────────
with tab_ml:
    st.title("🧠 ML Lab — Probabilistic Signals")
    if not SKLEARN_OK:
        st.warning("scikit-learn not installed. Run: pip install scikit-learn")
    symbol = st.text_input("Symbol (ML)", value="AAPL", key="v13_inp_ml_symbol").upper()
    horizon = st.slider("Prediction horizon (bars)", 1, 5, 1, key="v13_ml_horizon")
    train_frac = st.slider("Train fraction", 0.5, 0.95, 0.8, key="v13_ml_train_frac")
    proba_enter = st.slider("Enter if P(long) ≥", 0.50, 0.80, 0.55, 0.01, key="v13_ml_p_enter")
    proba_exit  = st.slider("Enter short if P(long) ≤", 0.20, 0.50, 0.45, 0.01, key="v13_ml_p_exit")
    run_ml = st.button("🤖 Train & Backtest", key="v13_btn_ml_run")

    def _ml_features(d: pd.DataFrame) -> pd.DataFrame:
        return _ml_features_from_indicators(d, rsi_period)

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

# ───────────────────────────── SCANNER + CORR (kept) ─────────────────────────────
with tab_scan:
    st.title("📡 Universe Scanner — Composite + (optional) ML")
    universe = st.text_area("Tickers (comma-separated)",
                            "AAPL, MSFT, NVDA, TSLA, AMZN, GOOGL, META, NFLX, SPY, QQQ",
                            key="v13_ta_scan_universe").upper()
    use_ml_scan = st.toggle("Include ML probability (needs scikit-learn)", value=False, key="v13_tg_ml_scan")
    run_scan = st.button("🔎 Scan", key="v13_btn_scan")

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
                sig = apply_earnings_guard(sig, t, earn_guard_days)
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

    st.markdown("---")
    st.markdown("### 🔗 Correlation Heatmap (daily returns)")
    if st.button("Build Correlation", key="v13_btn_corr"):
        tickers = [t.strip() for t in st.session_state.get("v13_ta_scan_universe","").split(",") if t.strip()]
        if len(tickers) >= 2:
            rets = []
            valid = []
            for t in tickers:
                px = load_prices(t, "1y", "1d")
                if px.empty: continue
                valid.append(t); rets.append(px["Close"].pct_change().rename(t))
            if valid:
                R = pd.concat(rets, axis=1).dropna(how="any")
                if not R.empty and len(R.columns) >= 2:
                    C = R.corr()
                    st.dataframe(C.style.format("{:.2f}"), use_container_width=True)
                    figC, axC = plt.subplots(figsize=(6,5))
                    im = axC.imshow(C.values, origin="lower", aspect="equal")
                    axC.set_xticks(range(len(C.columns))); axC.set_xticklabels(C.columns, rotation=45, ha="right")
                    axC.set_yticks(range(len(C.index)));   axC.set_yticklabels(C.index)
                    plt.colorbar(im, ax=axC, label="Corr")
                    axC.set_title("Correlation Heatmap")
                    st.pyplot(figC)
                else:
                    st.info("Not enough overlapping data.")
        else:
            st.info("Provide ≥2 tickers in the text box above.")

# ───────────────────────────── PORTFOLIO (kept) ─────────────────────────────
with tab_port:
    st.title("💼 Portfolio — Optimizers & Monte Carlo")

    st.subheader("⚖️ Risk Parity Optimizer")
    opt_tickers = st.text_input("Tickers (comma-sep)", "AAPL, MSFT, TSLA, SPY, QQQ", key="v13_inp_opt_tickers").upper()
    if st.button("🧮 Optimize (Risk Parity)", key="v13_btn_opt_rp"):
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
    mc_symbol = st.text_input("Symbol (MC)", value="AAPL", key="v13_inp_mc_symbol").upper()
    n_paths = st.slider("Paths", 200, 3000, 800, 100, key="v13_mc_paths")
    mc_mode = st.selectbox("Resampling", ["IID (simple)", "Block bootstrap"], index=1, key="v13_mc_mode")
    block_len = st.slider("Block length (if block bootstrap)", 5, 60, 20, key="v13_mc_block")
    run_mc = st.button("Run Monte Carlo", key="v13_btn_mc")

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
            if mc_mode == "IID (simple)":
                for _ in range(int(n_paths)):
                    samp = np.random.choice(r, size=N, replace=True)
                    eq = (1 + pd.Series(samp)).cumprod().iloc[-1]
                    endings.append(eq)
            else:
                bl = int(block_len)
                for _ in range(int(n_paths)):
                    seq = []
                    k = 0
                    while k < N:
                        start = np.random.randint(0, max(1, N - bl))
                        chunk = r[start:start+bl]
                        seq.extend(chunk)
                        k += bl
                    seq = np.array(seq[:N])
                    eq = (1 + pd.Series(seq)).cumprod().iloc[-1]
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
    st.header("How to use QuantaraX Pro (v13)")
    st.markdown(r"""
**New in v13**
- **Options Desk** (chains, IV when provided by Yahoo, Greeks-ready payoff plots for common strategies).
- **Position Sizing** (fixed risk %, ATR stops, Kelly approx).
- **Risk Hub** (VaR / CVaR, day-of-week returns, bootstrap risk-of-ruin).
- **Pattern Radar** (hammer, shooting star, doji, engulfings, 20-day breakouts).

**Tips for every trader**
- **Day trader**: MTF + Pattern Radar + ATR sizing. Use correlation heatmap to avoid stacking the same bet.
- **Swing trader**: Blended signals + Earnings guard + Seasonality. Use Parameter Surface to avoid overfit.
- **Options trader**: Options Desk to sketch payoffs; combine with Composite/ML bias and Earnings guard.
- **Investor**: Beta/Alpha, Rolling Sharpe, Profit Factor, One-Pager export for IC memos.

Everything is guarded for timeouts and missing fields. If a provider fails, modules degrade gracefully.
""")
