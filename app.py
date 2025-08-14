# app.py — QuantaraX Pro v27 (revolution upgrade)
# -----------------------------------------------------------------------------
# pip install:
#   streamlit yfinance pandas numpy matplotlib feedparser vaderSentiment scikit-learn
#
# Notes:
# - scikit-learn is optional; ML features will gracefully disable if missing
# - No extra libs beyond your existing stack
# - All network calls are guarded & retried; app degrades gracefully offline

import io
import math
import time
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Optional ML
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.inspection import permutation_importance
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

warnings.simplefilter("ignore", FutureWarning)

# ───────────────────────────── Page Setup ─────────────────────────────
st.set_page_config(page_title="QuantaraX Pro v27", layout="wide")
analyzer = SentimentIntensityAnalyzer()
REC_MAP = {1: "🟢 BUY", 0: "🟡 HOLD", -1: "🔴 SELL"}

# ───────────────────────────── Sidebar (Modes + Global) ─────────────────────────────
st.sidebar.header("Mode & Global Controls")

user_mode = st.sidebar.radio(
    "Experience",
    ["Beginner", "Pro"],
    index=0,
    key="mode_exp",
    help="Beginner shows a guided story; Pro exposes full controls."
)

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
    if "/" in s:
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

def _utcnow_ts() -> pd.Timestamp:
    # Avoid tz_localize on already tz-aware timestamps
    return pd.Timestamp.utcnow().tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT")

@st.cache_data(show_spinner=False, ttl=900)
def load_prices(symbol: str, period: str, interval: str) -> pd.DataFrame:
    sym = _map_symbol(symbol)
    for attempt in range(3):
        try:
            df = yf.download(sym, period=period, interval=interval, auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            if not df.empty:
                return df.dropna()
        except Exception:
            time.sleep(0.6 * (attempt + 1))
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

def next_earnings_date(symbol: str) -> Optional[pd.Timestamp]:
    """Return next future earnings date (UTC) or None."""
    try:
        cal = yf.Ticker(_map_symbol(symbol)).get_earnings_dates(limit=16)
        if isinstance(cal, pd.DataFrame) and not cal.empty:
            df = cal.copy()
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
            # find the datetime column
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
            now = _utcnow_ts()
            fut = df.dropna(subset=["earn_date"])
            fut = fut[fut["earn_date"] > now].sort_values("earn_date")
            if not fut.empty:
                return fut["earn_date"].iloc[0]
    except Exception:
        pass
    return None

# ─────────── Indicators / Composite ───────────
def compute_indicators(df: pd.DataFrame, ma_w: int, rsi_p: int, mf: int, ms: int, sig: int,
                       use_bb: bool = True) -> pd.DataFrame:
    d = df.copy()
    need = {"Open","High","Low","Close"}
    if d.empty or not need.issubset(d.columns):
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

    # Stochastic
    klen = 14
    ll = d["Low"].rolling(klen, min_periods=klen).min(); hh = d["High"].rolling(klen, min_periods=klen).max()
    rng = (hh - ll).replace(0, np.nan)
    d["STO_K"] = 100 * (d["Close"] - ll) / rng
    d["STO_D"] = d["STO_K"].rolling(3, min_periods=3).mean()

    # ADX (simplified)
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

    # Donchian / Keltner
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

# ─────────── Confidence 2.0 & Edge Decomposition ───────────
def edge_decomposition(row: pd.Series, weights: Dict[str, float]) -> Dict[str, float]:
    """Return contribution by component (signed, weighted)."""
    out = {}
    out["MA"]   = weights["MA"]   * float(row.get("MA_Signal", 0))
    out["RSI"]  = weights["RSI"]  * float(row.get("RSI_Signal", 0))
    out["MACD"] = weights["MACD"] * float(row.get("MACD_Signal2", 0))
    if "BB_Signal" in row.index:
        out["BB"] = weights.get("BB", 0.0) * float(row.get("BB_Signal", 0))
    return out

def calibrated_confidence(sig_df: pd.DataFrame, horizon: int = 1) -> Tuple[float, float, float]:
    """
    Estimate P(win) for next 'horizon' bars and a calibration coverage (%).
    Uses historical windows with similar Composite (binning).
    Returns (confidence_0to100, p_win, coverage_80_band).
    """
    try:
        if sig_df.empty or "Composite" not in sig_df or "Close" not in sig_df:
            return 50.0, 0.5, np.nan
        d = sig_df.copy()
        fwd = d["Close"].pct_change(horizon).shift(-horizon)
        d["fwd"] = fwd
        last_c = float(d["Composite"].iloc[-1])
        # bucket ±0.5 around last composite
        bucket = d[(d["Composite"] >= last_c - 0.5) & (d["Composite"] <= last_c + 0.5)]
        bucket = bucket.dropna(subset=["fwd"])
        if len(bucket) < 30:
            bucket = d.dropna(subset=["fwd"]).tail(400)  # fallback recent
        if bucket.empty:
            return 50.0, 0.5, np.nan
        wins = (bucket["fwd"] > 0).mean()
        # conformal-ish: empirical 10th/90th to form ~80% band
        q_lo, q_hi = np.percentile(bucket["fwd"], [10, 90])
        covered = ((bucket["fwd"] >= q_lo) & (bucket["fwd"] <= q_hi)).mean()
        conf = float(np.clip(wins * 100.0, 0, 100))
        return conf, float(wins), float(covered)
    except Exception:
        return 50.0, 0.5, np.nan

def beginner_story(sig_row: pd.Series, comp_val: float, rsi_p: int, ma_w: int) -> List[str]:
    """Plain-English narration of what's happening now."""
    out = []
    ma_s  = int(sig_row.get("MA_Signal", 0))
    rsi_s = int(sig_row.get("RSI_Signal", 0))
    macd_s= int(sig_row.get("MACD_Signal2", 0))
    rsi_v = float(sig_row.get(f"RSI{rsi_p}", np.nan))

    if ma_s == 1: out.append(f"Price just crossed **above** its {ma_w}-day average (a bullish momentum sign).")
    elif ma_s == -1: out.append(f"Price just crossed **below** its {ma_w}-day average (a bearish momentum sign).")

    if not np.isnan(rsi_v):
        if rsi_v < 30: out.append(f"RSI is **{rsi_v:.1f}** (oversold). Bounces are more common from here.")
        elif rsi_v > 70: out.append(f"RSI is **{rsi_v:.1f}** (overbought). Pullbacks are more common.")
        else: out.append(f"RSI is **{rsi_v:.1f}** (neutral).")

    if macd_s == 1: out.append("MACD crossed **above** its signal (bullish).")
    elif macd_s == -1: out.append("MACD crossed **below** its signal (bearish).")

    if "BB_Signal" in sig_row.index:
        bb_s = int(sig_row.get("BB_Signal", 0))
        if bb_s == 1: out.append("Price touched **below** the lower Bollinger band (mean-revert **long**).")
        elif bb_s == -1: out.append("Price touched **above** the upper Bollinger band (mean-revert **short**).")

    tilt = "bullish" if comp_val > 0 else ("bearish" if comp_val < 0 else "neutral")
    out.append(f"Overall tilt is **{tilt}** based on our weighted composite.")
    return out

# ─────────── Options Lens (lite) ───────────
def options_lens(symbol: str) -> Dict[str, Optional[float]]:
    """
    Try to compute: put/call ratio (volume-based), simple skew (IV proxy from option midpoints),
    fallback to realized vol if options unavailable.
    """
    out = {"put_call_ratio": None, "skew_proxy": None, "realized_vol_20": None}
    try:
        px = load_prices(symbol, "6mo", "1d")
        if not px.empty:
            out["realized_vol_20"] = float(px["Close"].pct_change().rolling(20).std(ddof=0) * np.sqrt(252)).__abs__()
        tk = yf.Ticker(_map_symbol(symbol))
        exps = tk.options
        if exps:
            # take nearest
            chain = tk.option_chain(exps[0])
            calls = chain.calls if hasattr(chain, "calls") else pd.DataFrame()
            puts  = chain.puts  if hasattr(chain, "puts")  else pd.DataFrame()
            if not calls.empty and not puts.empty:
                pc = puts.get("volume", pd.Series(dtype=float)).sum() / max(1.0, calls.get("volume", pd.Series(dtype=float)).sum())
                out["put_call_ratio"] = float(pc)
                # skew proxy: avg OTM put IV - avg OTM call IV (~ delta wing)
                def otm(df, kind="call"):
                    if df.empty: return np.nan
                    # Avoid requiring IV column; approximate from last_price vs intrinsic → noisy; skip if iv absent
                    if "impliedVolatility" in df.columns:
                        if kind == "call":
                            return df.sort_values("strike").head(10)["impliedVolatility"].mean()
                        else:
                            return df.sort_values("strike", ascending=False).head(10)["impliedVolatility"].mean()
                    return np.nan
                sk = otm(puts, "put") - otm(calls, "call")
                out["skew_proxy"] = float(sk) if not np.isnan(sk) else None
    except Exception:
        pass
    return out

# ─────────── Portfolio Doctor & Stress Replay ───────────
def portfolio_doctor(df_port: pd.DataFrame,
                     comp_thr: float,
                     profit_target: float,
                     loss_limit: float) -> pd.DataFrame:
    """
    Accepts a DataFrame with columns:
      ['Shares','Cost Basis','Price','Composite','P/L %']
    Returns suggestions: 'Top-up', 'Trim', 'Hold', 'Exit' + rationale.
    """
    adv = []
    for tkr, row in df_port.iterrows():
        pnl_pct = float(row.get("P/L %", np.nan))
        comp = float(row.get("Composite", 0.0))
        sugg = "Hold"; why = []
        if not np.isnan(pnl_pct):
            if pnl_pct >= profit_target:
                sugg = "Trim / Take Profit"; why.append(f"P/L {pnl_pct:.1f}% ≥ target {profit_target}%")
            elif pnl_pct <= -loss_limit:
                sugg = "Review / Consider Exit"; why.append(f"Drawdown {pnl_pct:.1f}% ≤ -{loss_limit}%")
        # Composite overlay
        if comp >= comp_thr:
            why.append(f"Composite {comp:.2f} ≥ thr {comp_thr} (bullish)")
            if sugg == "Hold": sugg = "Top-up (small)"
        elif comp <= -comp_thr:
            why.append(f"Composite {comp:.2f} ≤ -thr {comp_thr} (bearish)")
            if "Trim" not in sugg and "Exit" not in sugg:
                sugg = "Trim / Hedge"
        if not why: why.append("No strong signal. Stay sized.")
        adv.append({"Ticker": tkr, "Suggestion": sugg, "Why": " | ".join(why)})
    return pd.DataFrame(adv).set_index("Ticker")

STRESS_PRESETS = {
    "2020 COVID Crash": ("2020-02-19", "2020-03-23"),
    "2008 GFC Leg": ("2008-09-01", "2008-11-20"),
    "2013 Taper Tantrum": ("2013-05-01", "2013-06-25"),
}

def stress_replay(positions: Dict[str, float], period: Tuple[str, str]) -> pd.DataFrame:
    """Compute portfolio % change under historical window (simple replay)."""
    start, end = period
    rows = []
    for tkr, shares in positions.items():
        try:
            px = yf.download(_map_symbol(tkr), start=start, end=end, auto_adjust=True, progress=False)
            if px.empty: continue
            r = (px["Close"].iloc[-1] / px["Close"].iloc[0]) - 1.0
            rows.append({"Ticker": tkr, "Shock %": r*100})
        except Exception:
            continue
    df = pd.DataFrame(rows).set_index("Ticker")
    df["Weighted %"] = df["Shock %"]  # If shares vary, scale by MV / total; simplified here
    return df

# ─────────── Alerts & Journal ───────────
def check_alerts(signals: Dict[str, float], threshold: float) -> List[str]:
    """Return list of alert strings for symbols whose |composite| crossed threshold."""
    hits = []
    for tkr, comp in signals.items():
        if abs(comp) >= threshold:
            dir_ = "BUY" if comp > 0 else "SELL"
            hits.append(f"{tkr}: {dir_} (Composite={comp:.2f})")
    return hits

def add_journal_entry(tkr: str, action: str, reason: str, conf: float):
    if "journal" not in st.session_state:
        st.session_state["journal"] = []
    st.session_state["journal"].append({
        "time": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC"),
        "ticker": tkr,
        "action": action,
        "confidence": round(conf, 1),
        "reason": reason
    })

def generate_pdf_report(title: str, paragraphs: List[str], figures: List[plt.Figure]) -> bytes:
    """Create a simple multi-page PDF from text + matplotlib figures."""
    from matplotlib.backends.backend_pdf import PdfPages  # within std stack
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # Text page
        fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
        txt = "\n\n".join(paragraphs)
        fig.text(0.08, 0.95, title, fontsize=16, va="top", ha="left")
        fig.text(0.08, 0.90, txt, fontsize=10, va="top", ha="left", wrap=True)
        pdf.savefig(fig); plt.close(fig)
        # Figures
        for f in figures:
            pdf.savefig(f); plt.close(f)
    return buf.getvalue()

# ───────────────────────────── TABS ─────────────────────────────
tabs = st.tabs(["🚀 Engine", "🧠 ML Lab", "📡 Scanner", "📉 Regimes", "💼 Portfolio", "🔔 Alerts & Journal", "❓ Help"])
tab_engine, tab_ml, tab_scan, tab_regime, tab_port, tab_alerts, tab_help = tabs

# ───────────────────────────── ENGINE ─────────────────────────────
with tab_engine:
    st.title("🚀 QuantaraX — Composite Engine")

    # Inputs
    col0, col1 = st.columns([2,1])
    with col0:
        ticker = st.text_input("Symbol", "AAPL", key="inp_engine_ticker").upper().strip()
    with col1:
        horizon_view = st.selectbox("Horizon", ["1 bar", "3 bars", "5 bars"], index=0, key="engine_hz")
        hz = {"1 bar":1, "3 bars":3, "5 bars":5}[horizon_view]

    # Live Price
    px_live = load_prices(ticker, "5d", "1d")
    if not px_live.empty and "Close" in px_live:
        last_px = _to_float(px_live["Close"].iloc[-1])
        st.subheader(f"💲 Last close: ${last_px:.2f}")

    # News (guarded)
    news = safe_get_news(ticker)
    if news:
        with st.expander("📰 Recent News & Sentiment", expanded=(user_mode=="Beginner")):
            shown = 0
            for art in news:
                t_ = art.get("title",""); l_ = art.get("link","")
                if not (t_ and l_): continue
                txt = art.get("summary", t_)
                score = analyzer.polarity_scores(txt)["compound"]
                emoji = "🔺" if score>0.10 else ("🔻" if score<-0.10 else "➖")
                st.markdown(f"- [{t_}]({l_}) {emoji}")
                shown += 1
                if shown >= 5: break
    else:
        rss = rss_news(ticker, limit=5)
        if rss:
            with st.expander("📰 Recent News (RSS fallback)", expanded=(user_mode=="Beginner")):
                for r in rss:
                    st.markdown(f"- [{r['title']}]({r['link']})")

    # Earnings (future-only)
    nxt_ed = next_earnings_date(ticker)
    if nxt_ed is not None:
        st.info(f"📅 Next Earnings: **{nxt_ed.date()}**")

    # Run Engine
    if st.button("▶️ Run Composite Backtest", key="btn_engine_backtest"):
        px = load_prices(ticker, period_sel, interval_sel)
        if px.empty:
            st.error(f"No data for '{ticker}'."); st.stop()

        ind = compute_indicators(px, ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=include_bb)
        if ind.empty:
            st.error("Not enough data after indicators."); st.stop()

        sig = build_composite(ind, ma_window, rsi_period,
                              use_weighted=use_weighted, w_ma=w_ma, w_rsi=w_rsi, w_macd=w_macd, w_bb=w_bb,
                              include_bb=include_bb, threshold=comp_thr, allow_short=allow_short)
        bt, md, sh, wr, trd, tim, cagr = backtest(sig, allow_short=allow_short, cost_bps=cost_bps,
                                                  sl_atr_mult=sl_atr_mult, tp_atr_mult=tp_atr_mult,
                                                  vol_target=vol_target, interval=interval_sel)

        # Signal + Confidence
        last_trade = int(sig["Trade"].iloc[-1]) if "Trade" in sig.columns else 0
        rec = REC_MAP.get(1 if last_trade>0 else (-1 if last_trade<0 else 0), "🟡 HOLD")
        conf100, pwin, cover = calibrated_confidence(sig, horizon=hz)

        # Edge decomposition
        last = sig.iloc[-1]
        weights = {"MA": w_ma, "RSI": w_rsi, "MACD": w_macd, "BB": w_bb}
        ed = edge_decomposition(last, weights)
        bull_bear = "Bullish" if last_trade>0 else ("Bearish" if last_trade<0 else "Neutral")
        st.success(f"{ticker}: {rec} • **Confidence {conf100:.0f}/100** • Coverage(≈80% band)={np.nan if cover!=cover else cover*100:.0f}% • {bull_bear}")

        # Beginner Story Mode
        if user_mode == "Beginner":
            story = beginner_story(last, float(last.get("Composite",0)), rsi_period, ma_window)
            with st.expander("📖 Story (What’s happening & why)", expanded=True):
                for s in story:
                    st.write(f"- {s}")
                st.caption("Confidence is data-driven: we look at similar past setups and estimate the chance of a winning move over your chosen horizon.")

        # Metrics
        bh_last    = float(bt["CumBH"].iloc[-1]) if "CumBH" in bt else 1.0
        strat_last = float(bt["CumStrat"].iloc[-1]) if "CumStrat" in bt else 1.0
        colA, colB, colC, colD, colE, colF = st.columns(6)
        colA.metric("CAGR", f"{(0 if np.isnan(cagr) else cagr):.2f}%")
        colB.metric("Sharpe", f"{(0 if np.isnan(sh) else sh):.2f}")
        colC.metric("Max DD", f"{md:.2f}%")
        colD.metric("Win Rate", f"{wr:.1f}%")
        colE.metric("Trades", f"{trd}")
        colF.metric("Time in Mkt", f"{tim:.1f}%")
        st.markdown(f"- **Buy & Hold:** {(bh_last-1)*100:.2f}%  \n- **Strategy:** {(strat_last-1)*100:.2f}%")

        # Edge bar
        with st.expander("🧩 Edge Decomposition", expanded=(user_mode=="Pro")):
            ed_df = pd.DataFrame.from_dict(ed, orient="index", columns=["Contribution"])
            st.bar_chart(ed_df)

        # Options lens (lite)
        with st.expander("🧯 Options Lens (lite)", expanded=False):
            ol = options_lens(ticker)
            if any(v is not None for v in ol.values()):
                st.write({k: (None if v is None else float(v)) for k, v in ol.items()})
                hints = []
                if ol.get("put_call_ratio") is not None:
                    if ol["put_call_ratio"] > 1.2:
                        hints.append("Elevated put/call → defensive positioning.")
                    elif ol["put_call_ratio"] < 0.8:
                        hints.append("Low put/call → complacency risk.")
                if ol.get("realized_vol_20") is not None:
                    hints.append(f"Realized vol(20d) ≈ {ol['realized_vol_20']*100:.1f}% ann.")
                st.caption(" • ".join(hints) if hints else "No option signal available.")
            else:
                st.info("Option data unavailable; showing realized vol only if possible.")

        # Plots
        figs_for_pdf = []

        idx = bt.index
        fig1, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(11,12), sharex=True)
        ax1.plot(idx, bt["Close"], label="Close")
        if f"MA{ma_window}" in bt: ax1.plot(idx, bt[f"MA{ma_window}"], label=f"MA{ma_window}")
        if include_bb and {"BB_U","BB_L"}.issubset(bt.columns):
            ax1.plot(idx, bt["BB_U"], label="BB Upper"); ax1.plot(idx, bt["BB_L"], label="BB Lower")
        ax1.legend(); ax1.set_title("Price & Indicators")
        if "Composite" in bt:
            ax2.bar(idx, bt["Composite"]); ax2.set_title("Composite (weighted)")
        else:
            ax2.set_title("Composite (no data)")
        ax3.plot(idx, bt["CumBH"], ":", label="BH")
        ax3.plot(idx, bt["CumStrat"], "-", label="Strat"); ax3.legend(); ax3.set_title("Equity")
        plt.xticks(rotation=45); plt.tight_layout()
        st.pyplot(fig1)
        figs_for_pdf.append(fig1)

        # PDF Report
        with st.expander("📄 Export Decision Report (PDF)"):
            bullets = [
                f"Symbol: {ticker}",
                f"Recommendation: {rec}",
                f"Confidence: {conf100:.0f}/100 (p(win)≈{pwin:.2f})",
                f"Composite: {float(last.get('Composite',0)):.2f}",
                f"Risk: MaxDD={md:.2f}% | Sharpe={0 if np.isnan(sh) else sh:.2f} | Trades={trd}",
            ]
            pdf_bytes = generate_pdf_report(
                title=f"QuantaraX Decision Report — {ticker}",
                paragraphs=bullets,
                figures=figs_for_pdf
            )
            st.download_button("Download PDF", pdf_bytes, file_name=f"{ticker}_report.pdf", mime="application/pdf", key="dl_pdf_engine")

        # Quick journal
        with st.expander("📝 Add to Journal"):
            reason = st.text_area("Why are you acting (or not)?", key="jr_reason_engine")
            act = st.selectbox("Action", ["No action", "Buy", "Sell", "Hedge", "Reduce"], index=0, key="jr_action_engine")
            if st.button("Save Journal Entry", key="jr_save_engine"):
                add_journal_entry(ticker, act, reason, conf100)
                st.success("Saved to journal.")

    # Multi-Timeframe Confirmation (clean message, no Delta object)
    with st.expander("⏱️ Multi-Timeframe Confirmation", expanded=False):
        mtf_symbol = st.text_input("Symbol (MTF)", value=ticker or "AAPL", key="inp_mtf_symbol")
        if st.button("🔍 Check MTF", key="btn_mtf_check"):
            try:
                d1 = compute_indicators(load_prices(mtf_symbol, "1y", "1d"), ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=True)
                dH = compute_indicators(load_prices(mtf_symbol, "30d", "1h"), ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=True)
                if d1.empty or dH.empty:
                    st.warning("Insufficient data for MTF.")
                else:
                    c1 = build_composite(d1, ma_window, rsi_period, use_weighted=True, w_ma=1.0, w_rsi=1.0, w_macd=1.0, w_bb=0.5, include_bb=True, threshold=1.0)
                    cH = build_composite(dH, ma_window, rsi_period, use_weighted=True, w_ma=1.0, w_rsi=1.0, w_macd=1.0, w_bb=0.5, include_bb=True, threshold=1.0)
                    daily  = float(c1["Composite"].iloc[-1]); hourly = float(cH["Composite"].iloc[-1])
                    st.write(f"**Daily composite:** {daily:.2f} | **Hourly composite:** {hourly:.2f}")
                    if np.sign(daily) == np.sign(hourly):
                        st.success("✅ Signals agree")
                    else:
                        st.warning("⚠️ Signals disagree")
            except Exception as e:
                st.error(f"MTF error: {e}")

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
                comp = float(sig["Composite"].iloc[-1]) if "Composite" in sig else 0.0
                conf, pwin, _ = calibrated_confidence(sig, horizon=1)
                rec = REC_MAP.get(int(np.sign(comp)), "🟡 HOLD")
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
                rows.append({"Ticker":t, "Composite":comp, "Signal":rec, "Confidence":conf, "ML P(long)":mlp})
            except Exception:
                continue
        if rows:
            df = pd.DataFrame(rows).set_index("Ticker").sort_values(["Signal","Confidence","Composite"], ascending=[True,False,False])
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
    st.title("💼 Portfolio — Doctor, Optimizers & Stress Replay")
    st.markdown("Enter your positions in CSV: `ticker,shares,cost_basis`")

    holdings = st.text_area("Positions CSV", "AAPL,10,150\nMSFT,5,300\nSPY,2,420", height=120, key="ta_portfolio")
    if st.button("▶️ Analyze Portfolio", key="btn_port_analyze"):
        # Parse
        rows = [r.strip().split(",") for r in holdings.splitlines() if r.strip()]
        positions = []
        for idx, row in enumerate(rows, 1):
            if len(row) != 3:
                st.warning(f"Skipping invalid row {idx}: {row}"); continue
            tkr, shares, cost = row
            try:
                positions.append((tkr.strip().upper(), float(shares), float(cost)))
            except Exception:
                st.warning(f"Invalid numbers on row {idx}: {row}"); continue
        if not positions:
            st.error("No valid positions provided."); st.stop()

        # Price fetch + composites
        port_rows = []
        for tkr, sh, cb in positions:
            hist = load_prices(tkr, "5d", "1d")
            if hist.empty:
                st.warning(f"No price for {tkr}"); continue
            price = _to_float(hist["Close"].iloc[-1])
            invested = sh*cb; value=sh*price; pnl=value-invested
            pnl_pct=(pnl/invested*100) if invested else np.nan

            px = load_prices(tkr, period_sel, interval_sel)
            comp_val = np.nan
            if not px.empty:
                ind = compute_indicators(px, ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=include_bb)
                if not ind.empty:
                    sig = build_composite(ind, ma_window, rsi_period,
                                          use_weighted=use_weighted, w_ma=w_ma, w_rsi=w_rsi, w_macd=w_macd, w_bb=w_bb,
                                          include_bb=include_bb, threshold=comp_thr, allow_short=allow_short)
                    comp_val = float(sig["Composite"].iloc[-1]) if "Composite" in sig else np.nan

            port_rows.append({
                "Ticker":tkr, "Shares":sh, "Cost Basis":cb, "Price":price,
                "Market Value":value, "Invested":invested, "P/L":pnl, "P/L %":pnl_pct,
                "Composite": comp_val
            })

        if not port_rows:
            st.error("Could not load any positions.")
        else:
            df_port = pd.DataFrame(port_rows).set_index("Ticker")
            st.dataframe(df_port, use_container_width=True)
            c1,c2,c3 = st.columns(3)
            c1.metric("Total MV", f"${df_port['Market Value'].sum():,.2f}")
            c2.metric("Total Invested", f"${df_port['Invested'].sum():,.2f}")
            c3.metric("Total P/L", f"${(df_port['Market Value'].sum()-df_port['Invested'].sum()):,.2f}")

            st.subheader("👨‍⚕️ Portfolio Doctor — Target Plan")
            advice = portfolio_doctor(df_port, comp_thr, profit_target, loss_limit)
            st.dataframe(advice, use_container_width=True)

            # Stress Replay
            st.subheader("⛈️ Stress Replay")
            stress_name = st.selectbox("Scenario", list(STRESS_PRESETS.keys()), index=0, key="sel_stress")
            df_str = stress_replay({k: float(df_port.loc[k, "Shares"]) for k in df_port.index},
                                   STRESS_PRESETS[stress_name])
            if not df_str.empty:
                st.dataframe(df_str, use_container_width=True)
                st.caption("Approximate % move if the same historical shock repeated. (No path dependency; simple replay).")
            else:
                st.info("Stress data unavailable for given tickers.")

# ───────────────────────────── ALERTS & JOURNAL ─────────────────────────────
with tab_alerts:
    st.title("🔔 Alerts & Journal")

    # Alerts check for a small universe
    universe_alert = st.text_input("Tickers to watch (comma-sep)", "AAPL, MSFT, NVDA, SPY", key="alert_universe").upper()
    thr = st.slider("Alert threshold |Composite| ≥", 0.5, 3.0, 1.0, 0.1, key="alert_thr")
    if st.button("Check Alerts Now", key="btn_alert_check"):
        tickers = [t.strip() for t in universe_alert.split(",") if t.strip()]
        signals = {}
        for t in tickers:
            px = load_prices(t, "1y", "1d")
            ind = compute_indicators(px, ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=True)
            if ind.empty: continue
            sig = build_composite(ind, ma_window, rsi_period, use_weighted=True, w_ma=1.0, w_rsi=1.0, w_macd=1.0, w_bb=0.5,
                                  include_bb=True, threshold=1.0, allow_short=allow_short)
            signals[t] = float(sig["Composite"].iloc[-1]) if "Composite" in sig else 0.0
        hits = check_alerts(signals, thr)
        if hits:
            st.success("Alerts triggered:\n- " + "\n- ".join(hits))
        else:
            st.info("No alerts at the moment.")

    st.subheader("📔 Journal")
    if "journal" in st.session_state and st.session_state["journal"]:
        dfj = pd.DataFrame(st.session_state["journal"])
        st.dataframe(dfj, use_container_width=True)
        st.download_button("Export Journal CSV", dfj.to_csv(index=False), "journal.csv", key="dl_journal")
    else:
        st.info("No journal entries yet. Add them from the Engine tab after running a backtest.")

# ───────────────────────────── HELP ─────────────────────────────
with tab_help:
    st.title("❓ Help — How to use QuantaraX Pro v27")

    st.markdown("""
### What is this?
QuantaraX Pro turns well-known technical signals (MA crossovers, RSI zones, MACD flips, Bollinger touches) into a single **Composite Score** and then **backtests** those signals with realistic frictions (costs, stops/targets, volatility targeting). On top, it adds **Confidence 2.0** (calibrated win probability), **Edge Decomposition**, and a **Portfolio Doctor** that translates signal language into plain, portfolio-level actions.

---

### Quick Start (Beginner mode)
1. **Engine tab → enter a symbol → Run Composite Backtest.**  
   You’ll see:
   - **Recommendation** (Buy/Hold/Sell), and **Confidence 0–100** = estimated chance the next few bars are up.
   - A **story** explaining *why* (MA, RSI, MACD, BB) and what that usually meant historically.
   - Charts: price with indicators, composite bar, and cumulative returns vs Buy & Hold.

2. **Portfolio tab → paste your positions** as `ticker,shares,cost_basis`.  
   Get:
   - **Suggestions** per position (Top-up / Trim / Review) with reasons.
   - **Stress Replay**: how your book likely would have behaved in past shocks.

3. **Scanner tab** to sort a watchlist by Composite and Confidence.

4. **Alerts & Journal** to watch a small universe and record your decisions. Export as CSV or save a **PDF** report from the Engine tab.

---

### What does Confidence 0–100 mean?
We look back over **historical setups similar to now** (based on Composite). The confidence is **the historical win rate** (fraction of positive returns over your chosen horizon). We also show an approximate **coverage** number for an 80% band, indicating calibration quality.

**Example:** Confidence 67/100 on a 3-bar horizon ≈ historically, in similar setups, the next 3 bars were up ~67% of the time.

---

### Composite & Edge Decomposition (Pro)
- **Composite** combines directional cues:
  - **MA** crossover signals trend continuation/reversal
  - **RSI** detects overbought/oversold zones
  - **MACD** captures momentum shifts
  - **BB** flags mean-reversion extremes
- **Weights** are adjustable in the sidebar.  
- **Edge Decomposition** shows how much each component contributes **right now** (signed). If your edge is all MACD and nothing else, your thesis is momentum-heavy.

---

### Backtest realism
- **Transaction costs** (bps per side)
- **Volatility targeting** (keeps risk stable; scales positions)
- **ATR-based stop & target** (exit on next bar)
- **Optionally allow shorts**

**Key metrics:** Sharpe, Max Drawdown, Win Rate, Trades, Time in Market, CAGR.

---

### Portfolio Doctor
- Reads your **P/L %**, **cost basis**, **current price**, and **Composite**.
- Combines **guardrails** (profit target / loss limit) + **signal tilt** to suggest **Trim / Top-up / Hold / Hedge** with reasons.

**Note:** This is **not** investment advice—just structured, data-driven guidance.

---

### Multi-Timeframe (MTF)
Checks whether **daily** and **hourly** composites agree. Agreement → more robust timing; disagreement → proceed cautiously.

---

### Options Lens (lite)
If options data is available, we show a **put/call ratio** and a simple **skew** proxy. Otherwise we display **realized volatility**. Use this to gauge **market stress vs complacency** around a symbol.

---

### Regimes
Clustering daily features (vol, momentum, MA slope) into 3 regimes and shading the chart. Expect different strategies to work in different regimes (trend vs mean-revert).

---

### Alerts & Journal
- Alert when **|Composite| ≥ threshold** on your chosen tickers.
- Journal captures **time, action, confidence, and your reason**. Export for review.

---

### Tips
- **Beginners:** Stick to daily bars, keep thresholds near 1.0, and focus on large-cap tickers (cleaner signals).
- **Pros:** Use WFO on your end (external), tighten costs, explore **vol targeting**, test **shorts**, and watch **Edge Decomposition** for regime drift.

---

### Caveats
- This app uses public data sources that may be delayed or unavailable.
- All analytics are **educational** and **not investment advice**.
    """)

# End of file
