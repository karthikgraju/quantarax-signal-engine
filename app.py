# app.py — QuantaraX Pro v20 (investor-ready, hardened)
# ---------------------------------------------------------------------------------
# pip install: streamlit yfinance pandas numpy matplotlib feedparser vaderSentiment scikit-learn

import math
import time
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.simplefilter("ignore", FutureWarning)

# Optional ML imports (graceful degradation)
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.inspection import permutation_importance
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

# ───────────────────────────── Page Setup ─────────────────────────────
st.set_page_config(page_title="QuantaraX — Decision Engine (v20)", layout="wide")
analyzer = SentimentIntensityAnalyzer()
rec_map = {1: "🟢 BUY", 0: "🟡 HOLD", -1: "🔴 SELL"}

# Tabs
TAB_TITLES = ["🚀 Engine", "🧠 ML Lab", "📡 Scanner", "📉 Regimes", "💼 Portfolio", "❓ Help"]
tab_engine, tab_ml, tab_scan, tab_regime, tab_port, tab_help = st.tabs(TAB_TITLES)

# ───────────────────────────── Sidebar (unique keys) ─────────────────────────────
st.sidebar.header("Global Controls")
DEFAULTS = dict(ma_window=10, rsi_period=14, macd_fast=12, macd_slow=26, macd_signal=9)
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.sidebar.button("🔄 Reset to defaults", key="btn_reset_defaults_v20"):
    for k, v in DEFAULTS.items():
        st.session_state[k] = v

st.sidebar.subheader("Indicator Parameters")
ma_window   = st.sidebar.slider("MA window",      5, 60, st.session_state["ma_window"],   key="ma_window_v20")
rsi_period  = st.sidebar.slider("RSI lookback",   5, 30, st.session_state["rsi_period"],  key="rsi_period_v20")
macd_fast   = st.sidebar.slider("MACD fast span", 5, 20, st.session_state["macd_fast"],   key="macd_fast_v20")
macd_slow   = st.sidebar.slider("MACD slow span", 20, 50, st.session_state["macd_slow"],  key="macd_slow_v20")
macd_signal = st.sidebar.slider("MACD sig span",  5, 20, st.session_state["macd_signal"], key="macd_signal_v20")

st.sidebar.subheader("Composite v2 (advanced)")
use_weighted = st.sidebar.toggle("Use weighted composite", value=True, key="use_weighted_v20")
include_bb   = st.sidebar.toggle("Include Bollinger Bands", value=True, key="include_bb_v20")
w_ma   = st.sidebar.slider("Weight • MA",   0.0, 2.0, 1.0, 0.1, key="w_ma_v20")
w_rsi  = st.sidebar.slider("Weight • RSI",  0.0, 2.0, 1.0, 0.1, key="w_rsi_v20")
w_macd = st.sidebar.slider("Weight • MACD", 0.0, 2.0, 1.0, 0.1, key="w_macd_v20")
w_bb   = st.sidebar.slider("Weight • BB",   0.0, 2.0, 0.5, 0.1, key="w_bb_v20") if include_bb else 0.0
comp_thr = st.sidebar.slider("Composite trigger (enter/exit)", 0.0, 3.0, 1.0, 0.1, key="comp_thr_v20")

st.sidebar.subheader("Risk & Costs")
allow_short = st.sidebar.toggle("Allow shorts", value=False, key="allow_short_v20")
cost_bps    = st.sidebar.slider("Trading cost (bps/side)", 0.0, 25.0, 5.0, 0.5, key="cost_bps_v20")
sl_atr_mult = st.sidebar.slider("Stop • ATR ×", 0.0, 5.0, 2.0, 0.1, key="sl_atr_mult_v20")
tp_atr_mult = st.sidebar.slider("Target • ATR ×", 0.0, 8.0, 3.0, 0.1, key="tp_atr_mult_v20")
vol_target  = st.sidebar.slider("Vol targeting (annual)", 0.0, 0.5, 0.0, 0.05, key="vol_target_v20")

st.sidebar.subheader("Data")
period_sel   = st.sidebar.selectbox("History", ["6mo","1y","2y","5y"], index=1, key="period_sel_v20")
interval_sel = st.sidebar.selectbox("Interval", ["1d","1h"], index=0, key="interval_sel_v20")

st.sidebar.subheader("Portfolio Guardrails")
profit_target = st.sidebar.slider("Profit target (%)", 1, 100, 10, key="profit_target_v20")
loss_limit    = st.sidebar.slider("Loss limit (%)",  1, 100, 5,  key="loss_limit_v20")

# ───────────────────────────── Helpers ─────────────────────────────
def utcnow():
    return pd.Timestamp.now(tz="UTC")

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
    """Robust loader with retry; auto_adjust=True."""
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

def idx_to_utc(ts: pd.Timestamp):
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")

def data_health(df: pd.DataFrame, interval: str) -> dict:
    """Freshness and latency badge."""
    if df.empty:
        return {"fresh": False, "age_hours": np.nan}
    last_ts = df.index[-1]
    last_ts = idx_to_utc(pd.Timestamp(last_ts))
    now = utcnow()
    age_hours = max(0.0, (now - last_ts).total_seconds() / 3600.0)
    thr = 36.0 if interval == "1d" else 3.0
    return {"fresh": age_hours <= thr, "age_hours": age_hours}

def safe_get_news(symbol: str) -> list:
    """Try yfinance news; gracefully fallback to empty list."""
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
    """Returns DataFrame with normalized 'earn_date' (datetime64)."""
    try:
        cal = yf.Ticker(_map_symbol(symbol)).get_earnings_dates(limit=12)
        if isinstance(cal, pd.DataFrame) and not cal.empty:
            df = cal.copy()
            if isinstance(df.index, pd.DatetimeIndex) or (df.index.name and "date" in str(df.index.name).lower()):
                df = df.reset_index()
            # pick any datetime-like column
            date_col = None
            for c in df.columns:
                if "earn" in c.lower() and "date" in c.lower():
                    date_col = c; break
            if date_col is None:
                for c in df.columns:
                    if pd.api.types.is_datetime64_any_dtype(df[c]):
                        date_col = c; break
            if date_col is None:
                date_col = df.columns[0]
            df = df.rename(columns={date_col: "earn_date"})
            df["earn_date"] = pd.to_datetime(df["earn_date"], errors="coerce")
            return df.dropna(subset=["earn_date"]).sort_values("earn_date")
    except Exception:
        pass
    return pd.DataFrame()

def next_earnings_line(symbol: str) -> str:
    er = safe_earnings(symbol)
    if er.empty:
        return "📅 Earnings: unavailable"
    today = utcnow().date()
    future = er[er["earn_date"].dt.date >= today]
    if not future.empty:
        nxt = future.iloc[0]["earn_date"].date()
        return f"📅 Next Earnings: **{nxt}**"
    # else show last past date
    prev = er.iloc[-1]["earn_date"].date()
    return f"📅 Last Earnings: {prev} (no upcoming date found)"

# ─────────── Indicators / Composite ───────────
def compute_indicators(df: pd.DataFrame, ma_w: int, rsi_p: int, mf: int, ms: int, sig: int,
                       use_bb: bool = True) -> pd.DataFrame:
    d = df.copy()
    if d.empty or not set(["Open", "High", "Low", "Close"]).issubset(d.columns):
        return pd.DataFrame()

    d[f"MA{ma_w}"] = d["Close"].rolling(ma_w, min_periods=ma_w).mean()

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
    d["ATR"] = tr.ewm(alpha=1/14, adjust=False).mean()

    if use_bb:
        w = 20; k = 2.0
        mid = d["Close"].rolling(w, min_periods=w).mean()
        sd  = d["Close"].rolling(w, min_periods=w).std(ddof=0)
        d["BB_M"], d["BB_U"], d["BB_L"] = mid, mid + k*sd, mid - k*sd

    klen = 14
    ll = d["Low"].rolling(klen, min_periods=klen).min(); hh = d["High"].rolling(klen, min_periods=klen).max()
    rng = (hh - ll).replace(0, np.nan)
    d["STO_K"] = 100 * (d["Close"] - ll) / rng
    d["STO_D"] = d["STO_K"].rolling(3, min_periods=3).mean()

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
    d["DC_U"] = d["High"].rolling(dc_n, min_periods=dc_n).max()
    d["DC_L"] = d["Low"].rolling(dc_n, min_periods=dc_n).min()

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

    # Base position
    if allow_short:
        d["Position"] = d.get("Trade", 0).shift(1).fillna(0).clip(-1, 1)
        base_ret = np.where(d["Position"] >= 0, d["Return"], -d["Return"])
    else:
        d["Position"] = d.get("Trade", 0).shift(1).fillna(0).clip(0, 1)
        base_ret = d["Position"] * d["Return"]

    # Vol targeting
    if vol_target and vol_target > 0:
        look = 20
        daily_vol = d["Return"].rolling(look).std(ddof=0)
        ann = 252 if interval == "1d" else 252*6
        realized = daily_vol * math.sqrt(ann)
        scale = (vol_target / realized).clip(0, 3.0).fillna(0.0)
        base_ret = base_ret * scale

    # Trading costs
    cost = cost_bps/10000.0
    pos_change = d["Position"].diff().fillna(0).abs()
    tcost = -2.0*cost*(pos_change > 0).astype(float)  # open+close
    d["StratRet"] = pd.Series(base_ret, index=d.index).fillna(0.0) + tcost

    # ATR exits
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

# ─────────── Advanced analytics: Earnings Event Study & Factor Box ───────────
def earnings_event_study(px: pd.DataFrame, earn_df: pd.DataFrame,
                         window_pre=5, window_post=5) -> dict:
    if px.empty or earn_df.empty or "earn_date" not in earn_df:
        return {}
    px = px.copy().sort_index()
    px["ret"] = px["Close"].pct_change()
    closes = px["Close"]
    dates = pd.to_datetime(earn_df["earn_date"]).dropna().unique()
    paths, gaps, d1 = [], [], []
    for d in dates:
        if d not in px.index:
            idx = px.index.searchsorted(pd.Timestamp(d))
            if idx >= len(px.index):
                continue
            d = px.index[idx]
        loc = px.index.get_loc(d)
        if isinstance(loc, slice):
            loc = loc.start
        i0, i1 = max(0, loc-window_pre), min(len(px)-1, loc+window_post)
        seg = closes.iloc[i0:i1+1]
        if len(seg) < window_pre + window_post + 1:
            continue
        base = seg.iloc[window_pre-1] if window_pre>0 else seg.iloc[0]
        norm = (seg/float(base) - 1.0) * 100.0
        paths.append(norm.reset_index(drop=True))
        if {"Open","Close"}.issubset(px.columns):
            prev_close = px["Close"].iloc[loc-1] if loc>0 else np.nan
            open_px    = px["Open"].iloc[loc]
            if prev_close and not np.isnan(prev_close):
                gaps.append((open_px/prev_close-1)*100.0)
        if loc+1 < len(px):
            d1.append((px["Close"].iloc[loc+1]/px["Close"].iloc[loc]-1)*100.0)
    if not paths:
        return {}
    M = pd.concat(paths, axis=1).mean(axis=1)
    return {
        "mean_path_pct": M,
        "gap_mean_pct": np.nanmean(gaps) if gaps else np.nan,
        "gap_std_pct":  np.nanstd(gaps)  if gaps else np.nan,
        "next_day_mean_pct": np.nanmean(d1) if d1 else np.nan,
        "next_day_hit_rate": float(np.mean(np.array(d1)>0))*100 if d1 else np.nan,
        "n_events": len(paths)
    }

def factor_exposures(px: pd.DataFrame, lookback="3y") -> pd.DataFrame:
    proxies = {"SPY":"MKT","QQQ":"TECH","IWM":"SMB","TLT":"RATES","HYG":"CREDIT","UUP":"USD","GLD":"GOLD"}
    if px.empty or "Close" not in px: return pd.DataFrame()
    end = px.index.max()
    beg = end - pd.Timedelta(lookback)
    px = px.loc[px.index>=beg].copy()
    r_y = px["Close"].pct_change().dropna()
    if r_y.empty: return pd.DataFrame()
    regs = []
    X_cols = []
    for t, name in proxies.items():
        ref = load_prices(t, "5y", "1d")
        if ref.empty or "Close" not in ref: 
            continue
        r = ref["Close"].pct_change()
        df = pd.concat([r_y, r], axis=1, join="inner").dropna()
        if df.empty: 
            continue
        Y = df.iloc[:,0].values.reshape(-1,1)
        X = df.iloc[:,1].values.reshape(-1,1)
        Xc = np.column_stack([np.ones(len(X)), X])
        beta, *_ = np.linalg.lstsq(Xc, Y, rcond=None)
        resid = Y - Xc @ beta
        R2 = 1 - (resid**2).sum()/((Y - Y.mean())**2).sum()
        regs.append((name, float(beta[1]), float(beta[0]), float(R2)))
        X_cols.append(name)
    if not regs:
        return pd.DataFrame()
    return pd.DataFrame(regs, columns=["Factor","Beta","Alpha","R2"]).set_index("Factor")

# ─────────── Risk helpers & Confidence ───────────
def var_cvar(returns: pd.Series, alpha=0.05) -> Tuple[float,float]:
    r = returns.dropna().values
    if len(r) < 100: return np.nan, np.nan
    q = np.quantile(r, alpha)
    cvar = r[r <= q].mean() if (r <= q).any() else np.nan
    return float(q), float(cvar)

def position_size(capital: float, entry: float, atr: float, stop_atr=2.0, risk_frac=0.01) -> float:
    if any(map(lambda x: x is None or np.isnan(x) or x<=0, [capital, entry, atr])):
        return 0.0
    risk_per_share = stop_atr * atr
    max_loss = capital * risk_frac
    qty = max_loss / risk_per_share
    return float(max(0.0, np.floor(qty)))

def confidence_score(comp: float, mtf_agree: bool, senti: float) -> Tuple[int,str]:
    comp_s = np.tanh(comp/3.0)           # [-1,1] saturating
    mtf_s  = 0.2 if mtf_agree else -0.1
    sen_s  = np.clip(senti, -1, 1) * 0.3
    raw = comp_s*0.6 + mtf_s + sen_s
    score = int(np.clip((raw+1)/2*100, 0, 100))
    label = "Strongly Bullish" if score>=75 else ("Bullish" if score>=60 else ("Neutral" if score>=40 else ("Bearish" if score>=25 else "Strongly Bearish")))
    return score, label

# ───────────────────────────── ENGINE ─────────────────────────────
with tab_engine:
    st.title("QuantaraX — Decision Engine (v20)")

    c_head1, c_head2 = st.columns([3,1])
    with c_head1:
        ticker = st.text_input("Symbol (e.g., AAPL or BTC/USDT)", "AAPL", key="inp_engine_ticker_v20").upper()
    with c_head2:
        mode = st.selectbox("Mode", ["Beginner","Pro"], index=0, key="mode_v20")

    # Live pricing & freshness
    px_live = load_prices(ticker, "5d", "1d")
    last_px = _to_float(px_live["Close"].iloc[-1]) if not px_live.empty else np.nan
    meta = data_health(px_live, "1d")
    c1, c2, c3 = st.columns([1,1,1])
    c1.metric("💲 Last Close", f"${last_px:.2f}" if not np.isnan(last_px) else "N/A")
    c2.metric("✅ Fresh" if meta["fresh"] else "⚠️ Stale",
              "Yes" if meta["fresh"] else "No")
    c3.metric("⏱ Age", f"{meta['age_hours']:.1f}h" if not np.isnan(meta["age_hours"]) else "N/A")

    # Earnings line (robust future/past handling)
    st.info(next_earnings_line(ticker))

    # News pipeline
    news = safe_get_news(ticker)
    if news:
        st.markdown("#### 📰 Recent News & Sentiment")
        shown_scores=[]
        for art in news[:5]:
            t_ = art.get("title",""); l_ = art.get("link",""); txt = art.get("summary", t_)
            if not (t_ and l_): continue
            score = analyzer.polarity_scores(txt)["compound"]
            shown_scores.append(score)
            emoji = "🔺" if score>0.1 else ("🔻" if score<-0.1 else "➖")
            st.markdown(f"- [{t_}]({l_}) {emoji}")
        avg_sent = float(np.mean(shown_scores)) if shown_scores else 0.0
    else:
        rss = rss_news(ticker, limit=5)
        avg_sent = 0.0
        if rss:
            st.markdown("#### 📰 Recent News (RSS fallback)")
            for r in rss:
                st.markdown(f"- [{r['title']}]({r['link']})")
        else:
            st.info("No recent news found.")

    # Backtest controls
    if st.button("▶️ Run Composite Backtest", key="btn_engine_backtest_v20"):
        px = load_prices(ticker, period_sel, interval_sel)
        if px.empty:
            st.error(f"No data for '{ticker}'"); st.stop()
        df_raw = compute_indicators(px, ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=include_bb)
        if df_raw.empty:
            st.error("Not enough data after indicators (adjust windows or history)."); st.stop()
        df_sig = build_composite(df_raw, ma_window, rsi_period,
                                 use_weighted=use_weighted, w_ma=w_ma, w_rsi=w_rsi, w_macd=w_macd, w_bb=w_bb,
                                 include_bb=include_bb, threshold=comp_thr, allow_short=allow_short)
        df_c, max_dd, sharpe, win_rt, trades, tim, cagr = backtest(
            df_sig, allow_short=allow_short, cost_bps=cost_bps,
            sl_atr_mult=sl_atr_mult, tp_atr_mult=tp_atr_mult, vol_target=vol_target, interval=interval_sel
        )

        last_trade = int(df_sig["Trade"].tail(1).iloc[0]) if "Trade" in df_sig.columns and not df_sig.empty else 0
        rec = rec_map.get(1 if last_trade>0 else (-1 if last_trade<0 else 0), "🟡 HOLD")
        st.success(f"**{ticker}**: {rec}")

        # Explain signals
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

        # Charts
        idx = df_c.index
        fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(11,12), sharex=True)
        ax1.plot(idx, df_c["Close"], label="Close")
        if f"MA{ma_window}" in df_c: ax1.plot(idx, df_c[f"MA{ma_window}"], label=f"MA{ma_window}")
        if include_bb and {"BB_U","BB_L"}.issubset(df_c.columns):
            ax1.plot(idx, df_c["BB_U"], label="BB Upper"); ax1.plot(idx, df_c["BB_L"], label="BB Lower")
        ax1.legend(); ax1.set_title("Price & Indicators")
        ax2.bar(idx, df_c["Composite"]); ax2.set_title("Composite (weighted)")
        ax3.plot(idx, df_c["CumBH"], ":", label="BH"); ax3.plot(idx, df_c["CumStrat"], "-", label="Strat"); ax3.legend(); ax3.set_title("Equity")
        plt.xticks(rotation=45); plt.tight_layout()
        st.pyplot(fig)

        # MTF Confirmation (no DeltaGenerator repr)
        with st.expander("⏱️ Multi-Timeframe Confirmation", expanded=False):
            try:
                d1 = compute_indicators(load_prices(ticker, "1y", "1d"), ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=True)
                dH = compute_indicators(load_prices(ticker, "30d", "1h"), ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=True)
                if d1.empty or dH.empty:
                    st.warning("Insufficient data for MTF.")
                else:
                    c1 = build_composite(d1, ma_window, rsi_period, use_weighted=True, w_ma=1.0, w_rsi=1.0, w_macd=1.0, w_bb=0.5, include_bb=True, threshold=1.0)
                    cH = build_composite(dH, ma_window, rsi_period, use_weighted=True, w_ma=1.0, w_rsi=1.0, w_macd=1.0, w_bb=0.5, include_bb=True, threshold=1.0)
                    daily  = float(c1["Composite"].iloc[-1]); hourly = float(cH["Composite"].iloc[-1])
                    st.write(f"**Daily composite:** {daily:.2f}  |  **Hourly composite:** {hourly:.2f}")
                    if np.sign(daily) == np.sign(hourly):
                        st.success("✅ Signals agree")
                        mtf_ok = True
                    else:
                        st.warning("⚠️ Signals disagree")
                        mtf_ok = False
            except Exception as e:
                st.error(f"MTF error: {e}")
                mtf_ok = False

        # Earnings Radar
        with st.expander("📈 Earnings Radar — historical behavior", expanded=False):
            try:
                px_full = load_prices(ticker, "5y", "1d")
                er_df   = safe_earnings(ticker)
                stats   = earnings_event_study(px_full, er_df, 5, 5)
                if stats:
                    st.write(f"Events analyzed: **{stats['n_events']}**")
                    st.write(f"Avg gap: **{stats['gap_mean_pct']:.2f}%**  | Next-day mean: **{stats['next_day_mean_pct']:.2f}%**  | Hit-rate: **{stats['next_day_hit_rate']:.1f}%**")
                    path = stats["mean_path_pct"]
                    fig2, ax = plt.subplots(figsize=(6,3))
                    ax.plot(range(-5, len(path)-5), path.values)
                    ax.axvline(0, ls="--", alpha=0.6); ax.set_title("Avg path around earnings (T=earnings)")
                    ax.set_xlabel("Days from event"); ax.set_ylabel("% from T-1 close")
                    st.pyplot(fig2)
                else:
                    st.info("Not enough history to compute event-study.")
            except Exception as e:
                st.info(f"Earnings study unavailable: {e}")

        # Factor Box
        with st.expander("🧭 Factor Box — exposures via ETF proxies", expanded=False):
            try:
                px_full = load_prices(ticker, "5y", "1d")
                fx = factor_exposures(px_full, "3y")
                if not fx.empty:
                    st.dataframe(fx, use_container_width=True)
                else:
                    st.info("Not enough data for factor regression.")
            except Exception as e:
                st.info(f"Factor box unavailable: {e}")

        # Risk Suite & Confidence
        with st.expander("🛡️ Risk — VaR/CVaR & Position Sizing", expanded=False):
            try:
                r_ = df_c["StratRet"].dropna()
                v, cv = var_cvar(r_, 0.05)
                st.write(f"5% Historical VaR (1-bar): **{v*100:.2f}%**  | CVaR: **{cv*100:.2f}%**")
                atr_latest = float(df_c["ATR"].iloc[-1]) if "ATR" in df_c and not df_c["ATR"].empty else np.nan
                capital    = st.number_input("Account size ($)", 1000, 10_000_000, 50_000, step=1000, key="risk_cap_v20")
                stop_x     = st.slider("Stop distance (ATR×)", 0.5, 5.0, float(sl_atr_mult or 2.0), 0.5, key="risk_stopx_v20")
                px_latest  = float(df_c["Close"].iloc[-1])
                qty        = position_size(capital, px_latest, atr_latest, stop_x, 0.01)
                st.write(f"Suggested size (1% risk, ATR stop): **{qty:,.0f}** shares")
            except Exception:
                st.info("Risk metrics unavailable for this symbol/interval.")

        with st.expander("🧭 Confidence Gauge", expanded=True):
            try:
                comp_last = float(df_sig["Composite"].iloc[-1]) if "Composite" in df_sig else 0.0
                # Reuse MTF result if set; else compute quick
                mtf_agree = 'mtf_ok' in locals() and isinstance(mtf_ok, bool) and mtf_ok
                senti = float(avg_sent) if isinstance(avg_sent, float) else 0.0
                sc, label = confidence_score(comp_last, mtf_agree, senti)
                st.metric("Confidence", f"{sc}/100", label)
            except Exception:
                st.info("Confidence gauge unavailable (needs backtest run).")

    # Batch Backtest
    st.markdown("---")
    st.markdown("### Batch Backtest")
    batch = st.text_area("Tickers (comma-separated)", "AAPL, MSFT, TSLA, SPY, QQQ", key="ta_batch_v20").upper()
    if st.button("▶️ Run Batch Backtest", key="btn_batch_v20"):
        perf=[]
        for t in [x.strip() for x in batch.split(",") if x.strip()]:
            px = load_prices(t, period_sel, interval_sel)
            if px.empty: continue
            df_t = compute_indicators(px, ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=include_bb)
            if df_t.empty: continue
            df_tc = build_composite(df_t, ma_window, rsi_period,
                                    use_weighted=use_weighted, w_ma=w_ma, w_rsi=w_rsi, w_macd=w_macd, w_bb=w_bb,
                                    include_bb=include_bb, threshold=comp_thr, allow_short=allow_short)
            bt, md, sh, wr, trd, tim, cagr = backtest(df_tc, allow_short=allow_short, cost_bps=cost_bps,
                                                      sl_atr_mult=sl_atr_mult, tp_atr_mult=tp_atr_mult,
                                                      vol_target=vol_target, interval=interval_sel)
            comp_last = float(df_tc["Composite"].tail(1).iloc[0]) if "Composite" in df_tc and not df_tc["Composite"].empty else 0.0
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
            st.download_button("Download CSV", df_perf.to_csv(), "batch.csv", key="dl_batch_v20")
        else:
            st.error("No valid data for batch tickers.")

# ───────────────────────────── ML LAB ─────────────────────────────
with tab_ml:
    st.title("🧠 ML Lab — Probabilistic Signals")
    if not SKLEARN_OK:
        st.warning("scikit-learn not installed. Run: pip install scikit-learn")
    symbol = st.text_input("Symbol (ML)", value="AAPL", key="inp_ml_symbol_v20").upper()
    horizon = st.slider("Prediction horizon (bars)", 1, 5, 1, key="ml_horizon_v20")
    train_frac = st.slider("Train fraction", 0.5, 0.95, 0.8, key="ml_train_frac_v20")
    proba_enter = st.slider("Enter if P(long) ≥", 0.50, 0.80, 0.55, 0.01, key="ml_p_enter_v20")
    proba_exit  = st.slider("Enter short if P(long) ≤", 0.20, 0.50, 0.45, 0.01, key="ml_p_exit_v20")
    run_ml = st.button("🤖 Train & Backtest", key="btn_ml_run_v20")

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
            if not SKLEARN_OK: st.stop()
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

# ───────────────────────────── SCANNER ─────────────────────────────
with tab_scan:
    st.title("📡 Universe Scanner — Composite + (optional) ML")
    universe = st.text_area("Tickers (comma-separated)",
                            "AAPL, MSFT, NVDA, TSLA, AMZN, GOOGL, META, NFLX, SPY, QQQ",
                            key="ta_scan_universe_v20").upper()
    use_ml_scan = st.toggle("Include ML probability (needs scikit-learn)", value=False, key="tg_ml_scan_v20")
    run_scan = st.button("🔎 Scan", key="btn_scan_v20")

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
    sym = st.text_input("Symbol (Regime)", value="SPY", key="inp_regime_symbol_v20").upper()
    run_rg = st.button("Cluster Regimes", key="btn_regimes_v20")

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
            labels = None
            if SKLEARN_OK:
                from sklearn.cluster import KMeans
                km = KMeans(n_clusters=3, n_init=10, random_state=42)
                labels = km.fit_predict(feat)
            else:
                q1 = feat.rank(pct=True)
                labels = (q1.mean(axis=1) > 0.66).astype(int) + (q1.mean(axis=1) < 0.33).astype(int)*2
            reg = pd.Series(labels, index=feat.index, name="Regime")
            joined = ind.join(reg, how="right")
            ret = joined["Close"].pct_change().groupby(joined["Regime"]).mean().sort_values()
            ord_map = {old:i for i, old in enumerate(ret.index)}
            joined["Regime"] = joined["Regime"].map(ord_map)
            st.dataframe(joined[["Close","Regime"]].tail(10))
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(joined.index, joined["Close"], label="Close")
            for r in sorted([x for x in joined["Regime"].dropna().unique()]):
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
    opt_tickers = st.text_input("Tickers (comma-sep)", "AAPL, MSFT, TSLA, SPY, QQQ", key="inp_opt_tickers_v20").upper()
    if st.button("🧮 Optimize (Risk Parity)", key="btn_opt_rp_v20"):
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
    mc_symbol = st.text_input("Symbol (MC)", value="AAPL", key="inp_mc_symbol_v20").upper()
    n_paths = st.slider("Paths", 200, 3000, 800, 100, key="mc_paths_v20")
    run_mc = st.button("Run Monte Carlo", key="btn_mc_v20")

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
    st.header("How to use QuantaraX (v20)")
    st.markdown("""
**QuantaraX** is a decision engine for traders. It blends classic technical signals, event/catalyst analytics,
risk controls, and optional ML into a single workflow. You can use it in two ways:

### 1) Beginner quick-start
- **Type a symbol** and press **Run Composite Backtest**.
- Read the **Signal card** (BUY/HOLD/SELL) and the **Confidence gauge** (0–100).
- Open **Why This Signal?** to see the logic in plain English (MA/RSI/MACD/BB).
- Use **Risk — VaR/CVaR & Position Sizing** to pick a share size that caps loss to ~1% of account.
- Check **Earnings Radar** to see how the stock *usually behaves* around earnings.
- If Daily vs Hourly **agree** (MTF), confidence is stronger.
- If you're investing, open the **Factor Box** to understand what risks (market, tech, rates, etc.) you’re exposed to.

### 2) Pro workflow
- Tune **Composite weights & thresholds**, enable **shorts**, set **trading costs**, **vol targeting**, and **ATR exits**.
- Use **Walk-forward thinking** via the Composite + batch backtests, then scan a universe with the **Scanner**.
- In **ML Lab**, convert probabilities to trades and evaluate OOS performance (accuracy, AUC, perm. importance).
- In **Regimes**, map market states (vol/momentum/slope) and visually align strategy performance by regime.
- In **Portfolio**, allocate with **Risk Parity** and validate distribution of outcomes with **Monte Carlo**.

### What the main metrics mean
- **Composite**: weighted combo of MA crossovers, RSI (oversold/overbought), MACD crossovers, and (optional) Bollinger mean-reversion.  
- **Sharpe**: risk-adjusted return (annualized).  
- **Max Drawdown**: worst peak-to-trough loss (percent).  
- **VaR/CVaR**: historical tail loss estimate for a single bar; CVaR = average of the worst tail.  
- **Confidence**: combines Composite strength, MTF agreement, and news sentiment into a 0–100 score.

### Earnings: how to use it
- **Next Earnings** shows the upcoming date (or last, if none future).
- **Earnings Radar** studies historical behavior around earnings: average gap, next-day drift, and mean path (−5..+5 days).

### Factor Box (exposures)
We regress your asset’s daily returns vs ETF proxies (SPY/QQQ/IWM/TLT/HYG/UUP/GLD) to show **beta exposures** and **R²**.
Use this to answer: *“Am I secretly just long tech?”* or *“How rate-sensitive is this?”*.

### Risk discipline
Pick a fixed **risk per trade** (e.g., 1% of equity). The **Position Sizing** tool estimates shares using ATR-based stops.
Combine with **volatility targeting** if you want steadier risk over time.

### Scanner & Portfolio
- **Scanner**: ranks a basket by Composite and optional ML probability for fast idea generation.
- **Portfolio**: risk-parity allocator for diversified weights; Monte Carlo to visualize possible outcomes.

> **Disclaimers:** This is educational software. Markets are risky. Backtests are not guarantees of future results.
Adjust parameters, diversify, and practice solid risk management.
""")
