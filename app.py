# app.py — QuantaraX Pro (v9, investor-ready)
# ---------------------------------------------------------------------------------
# pip install:
#   streamlit yfinance pandas numpy matplotlib feedparser vaderSentiment scikit-learn

import math
import time
import warnings
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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
st.set_page_config(page_title="QuantaraX Pro v9", layout="wide")
analyzer = SentimentIntensityAnalyzer()
rec_map = {1: "🟢 BUY", 0: "🟡 HOLD", -1: "🔴 SELL"}


# ───────────────────────────── Tabs ─────────────────────────────
TAB_TITLES = [
    "🚀 Engine",
    "📋 Playbook",
    "🧠 ML Lab",
    "📡 Scanner",
    "📉 Regimes",
    "💼 Portfolio",
    "❓ Help",
]
(tab_engine, tab_play, tab_ml, tab_scan, tab_regime, tab_port, tab_help) = st.tabs(TAB_TITLES)


# ───────────────────────────── Sidebar (unique keys) ─────────────────────────────
st.sidebar.header("Global Controls")
DEFAULTS = dict(ma_window=10, rsi_period=14, macd_fast=12, macd_slow=26, macd_signal=9)
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.sidebar.button("🔄 Reset to defaults", key="btn_reset_defaults"):
    for k, v in DEFAULTS.items():
        st.session_state[k] = v

st.sidebar.subheader("Indicator Parameters")
ma_window = st.sidebar.slider("MA window", 5, 60, st.session_state["ma_window"], key="ma_window")
rsi_period = st.sidebar.slider("RSI lookback", 5, 30, st.session_state["rsi_period"], key="rsi_period")
macd_fast = st.sidebar.slider("MACD fast span", 5, 20, st.session_state["macd_fast"], key="macd_fast")
macd_slow = st.sidebar.slider("MACD slow span", 20, 50, st.session_state["macd_slow"], key="macd_slow")
macd_signal = st.sidebar.slider("MACD sig span", 5, 20, st.session_state["macd_signal"], key="macd_signal")

st.sidebar.subheader("Composite v2 (advanced)")
use_weighted = st.sidebar.toggle("Use weighted composite", value=True, key="use_weighted")
include_bb = st.sidebar.toggle("Include Bollinger Bands", value=True, key="include_bb")
w_ma = st.sidebar.slider("Weight • MA", 0.0, 2.0, 1.0, 0.1, key="w_ma")
w_rsi = st.sidebar.slider("Weight • RSI", 0.0, 2.0, 1.0, 0.1, key="w_rsi")
w_macd = st.sidebar.slider("Weight • MACD", 0.0, 2.0, 1.0, 0.1, key="w_macd")
w_bb = st.sidebar.slider("Weight • BB", 0.0, 2.0, 0.5, 0.1, key="w_bb") if include_bb else 0.0
comp_thr = st.sidebar.slider("Composite trigger (enter/exit)", 0.0, 3.0, 1.0, 0.1, key="comp_thr")

st.sidebar.subheader("Risk & Costs")
allow_short = st.sidebar.toggle("Allow shorts", value=False, key="allow_short")
cost_bps = st.sidebar.slider("Trading cost (bps/side)", 0.0, 25.0, 5.0, 0.5, key="cost_bps")
sl_atr_mult = st.sidebar.slider("Stop • ATR ×", 0.0, 5.0, 2.0, 0.1, key="sl_atr_mult")
tp_atr_mult = st.sidebar.slider("Target • ATR ×", 0.0, 8.0, 3.0, 0.1, key="tp_atr_mult")
vol_target = st.sidebar.slider("Vol targeting (annual)", 0.0, 0.5, 0.0, 0.05, key="vol_target")

st.sidebar.subheader("Data")
period_sel = st.sidebar.selectbox("History", ["6mo", "1y", "2y", "5y"], index=1, key="period_sel")
interval_sel = st.sidebar.selectbox("Interval", ["1d", "1h"], index=0, key="interval_sel")

st.sidebar.subheader("Account / Guardrails")
acct_equity = st.sidebar.number_input("Account equity ($)", min_value=1000.0, value=10000.0, step=500.0, key="acct_equity")
risk_per_trade_pct = st.sidebar.slider("Risk per trade (%)", 0.1, 2.0, 0.5, 0.1, key="risk_trade_pct")
profit_target = st.sidebar.slider("Profit target (%)", 1, 100, 10, key="profit_target")
loss_limit = st.sidebar.slider("Loss limit (%)", 1, 100, 5, key="loss_limit")


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


def safe_earnings(symbol: str) -> pd.DataFrame:
    """
    Returns DataFrame with normalized 'earn_date' (datetime64).
    Handles index/column schema variations from yfinance.
    """
    try:
        cal = yf.Ticker(_map_symbol(symbol)).get_earnings_dates(limit=12)
        if isinstance(cal, pd.DataFrame) and not cal.empty:
            df = cal.copy()

            # If dates are in the index, lift to column
            if isinstance(df.index, pd.DatetimeIndex) or (
                df.index.name and "earn" in str(df.index.name).lower() and "date" in str(df.index.name).lower()
            ):
                df = df.reset_index()

            # Detect the date column
            date_col = None
            for c in df.columns:
                cl = c.lower().replace(" ", "")
                if "earn" in cl and "date" in cl:
                    date_col = c
                    break
            if date_col is None:
                for c in df.columns:
                    if pd.api.types.is_datetime64_any_dtype(df[c]):
                        date_col = c
                        break
            if date_col is None:
                date_col = df.columns[0]

            df = df.rename(columns={date_col: "earn_date"})
            df["earn_date"] = pd.to_datetime(df["earn_date"], errors="coerce")
            cols = ["earn_date"] + [c for c in df.columns if c != "earn_date"]
            return df[cols].dropna(subset=["earn_date"])
    except Exception:
        pass
    return pd.DataFrame()


def next_earnings_date(symbol: str) -> Optional[pd.Timestamp]:
    df = safe_earnings(symbol)
    if df.empty:
        return None
    now = pd.Timestamp.utcnow().tz_localize("UTC")
    df["earn_date"] = pd.to_datetime(df["earn_date"], utc=True, errors="coerce")
    fut = df[df["earn_date"] >= now]
    if not fut.empty:
        return fut.sort_values("earn_date")["earn_date"].iloc[0]
    # If none in future, return the most recent past (for info)
    return df.sort_values("earn_date")["earn_date"].iloc[-1]


# ─────────── Indicators / Composite ───────────
def compute_indicators(
    df: pd.DataFrame, ma_w: int, rsi_p: int, mf: int, ms: int, sig: int, use_bb: bool = True
) -> pd.DataFrame:
    d = df.copy()
    if d.empty or not set(["Open", "High", "Low", "Close"]).issubset(d.columns):
        return pd.DataFrame()

    # MA
    d[f"MA{ma_w}"] = d["Close"].rolling(ma_w, min_periods=ma_w).mean()

    # RSI (EMA-based)
    chg = d["Close"].diff()
    up, dn = chg.clip(lower=0), -chg.clip(upper=0)
    ema_up = up.ewm(com=rsi_p - 1, adjust=False).mean()
    ema_down = dn.ewm(com=rsi_p - 1, adjust=False).mean()
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
    tr = pd.concat(
        [(d["High"] - d["Low"]).abs(), (d["High"] - pc).abs(), (d["Low"] - pc).abs()], axis=1
    ).max(axis=1)
    d["ATR"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    # Bollinger
    if use_bb:
        w = 20
        k = 2.0
        mid = d["Close"].rolling(w, min_periods=w).mean()
        sd = d["Close"].rolling(w, min_periods=w).std(ddof=0)
        d["BB_M"], d["BB_U"], d["BB_L"] = mid, mid + k * sd, mid - k * sd

    # Stochastic
    klen = 14
    ll = d["Low"].rolling(klen, min_periods=klen).min()
    hh = d["High"].rolling(klen, min_periods=klen).max()
    rng = (hh - ll).replace(0, np.nan)
    d["STO_K"] = 100 * (d["Close"] - ll) / rng
    d["STO_D"] = d["STO_K"].rolling(3, min_periods=3).mean()

    # ADX (simplified Wilder's)
    adx_n = 14
    up_move = d["High"].diff()
    dn_move = -d["Low"].diff()
    plus_dm = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
    tr_sm = tr.ewm(alpha=1 / adx_n, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=d.index).ewm(alpha=1 / adx_n, adjust=False).mean() / tr_sm
    minus_di = (
        100 * pd.Series(minus_dm, index=d.index).ewm(alpha=1 / adx_n, adjust=False).mean() / tr_sm
    )
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan) * 100
    d["ADX"] = dx.ewm(alpha=1 / adx_n, adjust=False).mean()

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


def build_composite(
    df: pd.DataFrame,
    ma_w: int,
    rsi_p: int,
    *,
    use_weighted=True,
    w_ma=1.0,
    w_rsi=1.0,
    w_macd=1.0,
    w_bb=0.5,
    include_bb=True,
    threshold=0.0,
    allow_short=False,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    d = df.copy()
    n = len(d)
    close = d["Close"].to_numpy()
    ma = d[f"MA{ma_w}"].to_numpy()
    rsi = d[f"RSI{rsi_p}"].to_numpy()
    macd = d["MACD"].to_numpy()
    sigl = d["MACD_Signal"].to_numpy()

    ma_sig = np.zeros(n, int)
    rsi_sig = np.zeros(n, int)
    macd_sig2 = np.zeros(n, int)
    bb_sig = np.zeros(n, int)

    for i in range(1, n):
        if close[i - 1] < ma[i - 1] and close[i] > ma[i]:
            ma_sig[i] = 1
        elif close[i - 1] > ma[i - 1] and close[i] < ma[i]:
            ma_sig[i] = -1
        if rsi[i] < 30:
            rsi_sig[i] = 1
        elif rsi[i] > 70:
            rsi_sig[i] = -1
        if macd[i - 1] < sigl[i - 1] and macd[i] > sigl[i]:
            macd_sig2[i] = 1
        elif macd[i - 1] > sigl[i - 1] and macd[i] < sigl[i]:
            macd_sig2[i] = -1
        if include_bb and {"BB_U", "BB_L"}.issubset(d.columns):
            if close[i] < d["BB_L"].iloc[i]:
                bb_sig[i] = 1
            elif close[i] > d["BB_U"].iloc[i]:
                bb_sig[i] = -1

    comp = (
        w_ma * ma_sig + w_rsi * rsi_sig + w_macd * macd_sig2 + (w_bb * bb_sig if include_bb else 0)
    ) if use_weighted else (ma_sig + rsi_sig + macd_sig2)

    if allow_short:
        trade = np.where(comp >= threshold, 1, np.where(comp <= -threshold, -1, 0))
    else:
        trade = np.where(comp >= threshold, 1, 0)

    d["MA_Signal"], d["RSI_Signal"], d["MACD_Signal2"] = ma_sig, rsi_sig, macd_sig2
    if include_bb:
        d["BB_Signal"] = bb_sig
    d["Composite"] = comp.astype(float)
    d["Trade"] = trade.astype(int)
    return d


# ─────────── Backtest ───────────
def _stats_from_equity(d: pd.DataFrame, interval: str) -> Tuple[float, float, float, float, int, float, float]:
    ann = 252 if interval == "1d" else 252 * 6
    if d["CumStrat"].notna().any():
        dd = d["CumStrat"] / d["CumStrat"].cummax() - 1
        max_dd = float(dd.min() * 100)
        last_cum = float(d["CumStrat"].dropna().iloc[-1])
    else:
        max_dd = 0.0
        last_cum = 1.0
    mean_ann = float(d["StratRet"].mean() * ann)
    vol_ann = float(d["StratRet"].std(ddof=0) * math.sqrt(ann))
    sharpe = (mean_ann / vol_ann) if vol_ann > 0 else np.nan
    win_rt = float((d["StratRet"] > 0).mean() * 100)
    pos_change = d["Position"].diff().fillna(0).abs()
    trades = int((pos_change > 0).sum())
    tim = float((d["Position"] != 0).mean() * 100)
    n_eff = int(d["StratRet"].notna().sum())
    cagr = ((last_cum ** (ann / max(n_eff, 1))) - 1) * 100 if n_eff > 0 else np.nan
    return max_dd, sharpe, win_rt, trades, tim, cagr, last_cum


def backtest(
    df: pd.DataFrame,
    *,
    allow_short=False,
    cost_bps=0.0,
    sl_atr_mult=0.0,
    tp_atr_mult=0.0,
    vol_target=0.0,
    interval="1d",
):
    d = df.copy()
    if d.empty or "Close" not in d:
        sk = d.copy()
        for col in ["Return", "Position", "StratRet", "CumBH", "CumStrat"]:
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
        ann = 252 if interval == "1d" else 252 * 6
        realized = daily_vol * math.sqrt(ann)
        scale = (vol_target / realized).clip(0, 3.0).fillna(0.0)  # cap leverage
        base_ret = base_ret * scale

    # Costs on trades
    cost = cost_bps / 10000.0
    pos_change = d["Position"].diff().fillna(0).abs()
    tcost = -2.0 * cost * (pos_change > 0).astype(float)  # open+close
    d["StratRet"] = pd.Series(base_ret, index=d.index).fillna(0.0) + tcost

    # ATR exits → flatten next bar
    if (sl_atr_mult > 0 or tp_atr_mult > 0) and "ATR" in d.columns:
        flat = np.zeros(len(d), dtype=int)
        entry = np.nan
        for i in range(len(d)):
            p, c = d["Position"].iat[i], d["Close"].iat[i]
            a = d["ATR"].iat[i] if "ATR" in d.columns else np.nan
            if p != 0 and np.isnan(entry):
                entry = c
            if p == 0:
                entry = np.nan
            if p != 0 and not np.isnan(a):
                if p == 1 and (c <= entry - sl_atr_mult * a or c >= entry + tp_atr_mult * a):
                    flat[i] = 1
                    entry = np.nan
                if p == -1 and (c >= entry + sl_atr_mult * a or c <= entry - tp_atr_mult * a):
                    flat[i] = 1
                    entry = np.nan
        if flat.any():
            d.loc[flat == 1, "Position"] = 0

    ret_bh = d["Return"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ret_st = d["StratRet"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    d["CumBH"] = (1 + ret_bh).cumprod()
    d["CumStrat"] = (1 + ret_st).cumprod()

    max_dd, sharpe, win_rt, trades, tim, cagr, _ = _stats_from_equity(d, interval)
    return d, max_dd, sharpe, win_rt, trades, tim, cagr


# ─────────── Extras for Investor-Ready UX ───────────
def mtf_signs(symbol: str):
    d1 = compute_indicators(load_prices(symbol, "1y", "1d"), ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=True)
    dH = compute_indicators(load_prices(symbol, "30d", "1h"), ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=True)
    if d1.empty or dH.empty:
        return np.nan, np.nan, False
    c1 = build_composite(d1, ma_window, rsi_period, use_weighted=True, w_ma=1.0, w_rsi=1.0, w_macd=1.0, w_bb=0.5, include_bb=True, threshold=1.0)
    cH = build_composite(dH, ma_window, rsi_period, use_weighted=True, w_ma=1.0, w_rsi=1.0, w_macd=1.0, w_bb=0.5, include_bb=True, threshold=1.0)
    daily = float(c1["Composite"].iloc[-1])
    hourly = float(cH["Composite"].iloc[-1])
    agree = (np.sign(daily) == np.sign(hourly)) and (np.sign(daily) != 0)
    return daily, hourly, agree


def regime_edge_simple(px: pd.DataFrame) -> float:
    """
    Simple regime edge score in [-1, 1]:
    - uses 20d momentum and 20d realized vol percentile to gauge tailwind/headwind.
    """
    if px.empty or "Close" not in px:
        return 0.0
    ret = px["Close"].pct_change()
    mom20 = px["Close"].pct_change(20)
    vol20 = ret.rolling(20).std()
    if mom20.dropna().empty or vol20.dropna().empty:
        return 0.0
    m = float(mom20.iloc[-1])
    v = float(vol20.iloc[-1])
    vp = float((vol20.rank(pct=True).iloc[-1]))  # high vol → headwind
    mp = float((mom20.rank(pct=True).iloc[-1]))  # high momentum → tailwind
    edge = (mp - 0.5) - (vp - 0.5)  # tailwind minus headwind
    return float(np.clip(edge * 2.0, -1.0, 1.0))  # scale a bit


def composite_percentile(df_sig: pd.DataFrame) -> float:
    """Return percentile of latest Composite in [0,1]."""
    if df_sig.empty or "Composite" not in df_sig:
        return 0.5
    s = df_sig["Composite"].dropna()
    if len(s) < 20:
        return 0.5
    latest = s.iloc[-1]
    pct = float((s < latest).mean())
    return float(np.clip(pct, 0.0, 1.0))


def freshness_penalty(px: pd.DataFrame) -> float:
    """1.0 if last bar < 24h old, decays thereafter."""
    if px.empty:
        return 0.0
    last = px.index[-1]
    if not isinstance(last, pd.Timestamp):
        return 0.5
    now = pd.Timestamp.utcnow().tz_localize("UTC")
    if last.tzinfo is None:
        last = last.tz_localize("UTC")
    hours = (now - last).total_seconds() / 3600.0
    score = np.exp(-max(0.0, hours - 24) / 48.0)
    return float(np.clip(score, 0.0, 1.0))


def earnings_penalty(days_to_e: Optional[int]) -> float:
    """
    0.0 → no penalty; 1.0 → full penalty.
    Heavier penalty as earnings approach.
    """
    if days_to_e is None:
        return 0.0
    if days_to_e <= 0:
        return 1.0
    if days_to_e <= 1:
        return 0.9
    if days_to_e <= 3:
        return 0.6
    if days_to_e <= 7:
        return 0.4
    if days_to_e <= 14:
        return 0.2
    return 0.0


def confidence_score(
    df_sig: pd.DataFrame, px_all: pd.DataFrame, mtf_agree: bool, comp_pct: float, regime_edge: float, days_to_e: Optional[int]
) -> float:
    """
    Blend:
      +35% composite strength percentile
      +25% MTF agreement
      +20% regime edge
      +10% data freshness
      -10% earnings proximity penalty
    Return 0..100
    """
    fresh = freshness_penalty(px_all)
    earn_pen = earnings_penalty(days_to_e)
    mtf = 1.0 if mtf_agree else 0.0

    raw = (0.35 * comp_pct) + (0.25 * mtf) + (0.20 * (regime_edge * 0.5 + 0.5)) + (0.10 * fresh) - (0.10 * earn_pen)
    return float(np.clip(raw * 100.0, 0.0, 100.0))


def suggest_position_size(acct_equity: float, atr: float, price: float, stop_mult: float, risk_pct: float) -> int:
    """
    Very simple ATR-based sizing:
      - Risk per trade = equity * risk_pct
      - Stop distance = ATR * max(stop_mult, 1.5)
      - Size = floor(risk_dollars / stop_distance)
    """
    if any([acct_equity <= 0, atr <= 0, price <= 0]):
        return 0
    risk_dollars = acct_equity * (risk_pct / 100.0)
    stop_dist = max(stop_mult, 1.5) * atr
    if stop_dist <= 0:
        return 0
    size = int(max(0, math.floor(risk_dollars / stop_dist)))
    return size


def render_next_earnings(symbol: str):
    dt = next_earnings_date(symbol)
    if dt is None:
        st.info("📅 Earnings: unavailable")
        return None
    # If future → “Next”; if past → “Last reported”
    now = pd.Timestamp.utcnow().tz_localize("UTC")
    if dt >= now:
        days = int((dt - now).days)
        st.info(f"📅 Next Earnings: **{dt.date()}**  ({days} day(s))")
        return days
    else:
        st.info(f"📅 Last reported earnings: **{dt.date()}**")
        return None


# ───────────────────────────── ENGINE ─────────────────────────────
with tab_engine:
    st.title("🚀 QuantaraX — Composite Signal Engine (v9)")

    st.markdown("### Single‐Ticker Backtest")
    ticker = st.text_input("Ticker (e.g. AAPL or BTC/USDT)", "AAPL", key="inp_engine_ticker").upper()

    # Live Price (cached loader)
    px_live = load_prices(ticker, "5d", "1d")
    if not px_live.empty and "Close" in px_live:
        last_px = _to_float(px_live["Close"].iloc[-1])
        st.subheader(f"💲 Last close: ${last_px:.2f}")

    # News (safe → RSS fallback)
    news = safe_get_news(ticker)
    if news:
        st.markdown("#### 📰 Recent News & Sentiment (YFinance)")
        shown = 0
        for art in news:
            t_ = art.get("title", "")
            l_ = art.get("link", "")
            if not (t_ and l_):
                continue
            txt = art.get("summary", t_)
            score = analyzer.polarity_scores(txt)["compound"]
            emoji = "🔺" if score > 0.1 else ("🔻" if score < -0.1 else "➖")
            st.markdown(f"- [{t_}]({l_}) {emoji}")
            shown += 1
            if shown >= 5:
                break
    else:
        rss = rss_news(ticker, limit=5)
        if rss:
            st.markdown("#### 📰 Recent News (RSS Fallback)")
            for r in rss:
                st.markdown(f"- [{r['title']}]({r['link']})")
        else:
            st.info("No recent news found.")

    # Earnings (robust)
    days_to_e = render_next_earnings(ticker)

    if st.button("▶️ Run Composite Backtest", key="btn_engine_backtest"):
        px = load_prices(ticker, period_sel, interval_sel)
        if px.empty:
            st.error(f"No data for '{ticker}'")
            st.stop()

        df_raw = compute_indicators(px, ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=include_bb)
        if df_raw.empty:
            st.error("Not enough data after indicators (try longer period or smaller windows).")
            st.stop()

        df_sig = build_composite(
            df_raw,
            ma_window,
            rsi_period,
            use_weighted=use_weighted,
            w_ma=w_ma,
            w_rsi=w_rsi,
            w_macd=w_macd,
            w_bb=w_bb,
            include_bb=include_bb,
            threshold=comp_thr,
            allow_short=allow_short,
        )
        if df_sig.empty:
            st.error("Composite could not be built (insufficient rows).")
            st.stop()

        # MTF + Confidence
        daily_c, hourly_c, agree = mtf_signs(ticker)
        comp_pct = composite_percentile(df_sig)
        regime = regime_edge_simple(px)
        conf = confidence_score(df_sig, px, agree, comp_pct, regime, days_to_e)

        # Backtest
        df_c, max_dd, sharpe, win_rt, trades, tim, cagr = backtest(
            df_sig,
            allow_short=allow_short,
            cost_bps=cost_bps,
            sl_atr_mult=sl_atr_mult,
            tp_atr_mult=tp_atr_mult,
            vol_target=vol_target,
            interval=interval_sel,
        )

        last_trade = int(df_sig["Trade"].tail(1).iloc[0]) if "Trade" in df_sig.columns and not df_sig.empty else 0
        rec = rec_map.get(1 if last_trade > 0 else (-1 if last_trade < 0 else 0), "🟡 HOLD")

        # ── Signal Cards ──
        cA, cB, cC, cD = st.columns(4)
        cA.metric("Signal", rec)
        cB.metric("Confidence", f"{conf:.0f}/100")
        cC.metric("Sharpe (BT)", f"{(0 if np.isnan(sharpe) else sharpe):.2f}")
        cD.metric("Max DD (BT)", f"{max_dd:.2f}%")

        # Position sizing helper
        last_row = df_raw.tail(1)
        atr_v = float(last_row["ATR"].iloc[0]) if "ATR" in df_raw and not last_row.empty else float("nan")
        px_v = float(df_raw["Close"].iloc[-1])
        size = suggest_position_size(acct_equity, atr_v, px_v, sl_atr_mult if sl_atr_mult > 0 else 2.0, risk_per_trade_pct)
        with st.expander("📏 Sizing helper"):
            st.write(
                f"- ATR ≈ **{atr_v:.2f}** | Price ≈ **{px_v:.2f}** | Risk/trade ≈ **${acct_equity * (risk_per_trade_pct/100):.0f}**"
            )
            st.write(f"- Suggested position size: **{size}** shares (ATR stop)")

        # Reasoning
        last = df_sig.tail(1).iloc[0]
        ma_s = int(last.get("MA_Signal", 0))
        rsi_s = int(last.get("RSI_Signal", 0))
        macd_s = int(last.get("MACD_Signal2", 0))
        rsi_v = float(last.get(f"RSI{rsi_period}", np.nan))
        ma_txt = {
            1: f"Price ↑ crossed above MA{ma_window}.",
            0: "No MA crossover.",
            -1: f"Price ↓ crossed below MA{ma_window}.",
        }.get(ma_s, "No MA crossover.")
        rsi_txt = (
            "RSI data unavailable."
            if np.isnan(rsi_v)
            else {
                1: f"RSI ({rsi_v:.1f}) < 30 → oversold.",
                0: f"RSI ({rsi_v:.1f}) neutral.",
                -1: f"RSI ({rsi_v:.1f}) > 70 → overbought.",
            }.get(rsi_s, f"RSI ({rsi_v:.1f}) neutral.")
        )
        macd_txt = {1: "MACD ↑ crossed above signal.", 0: "No MACD crossover.", -1: "MACD ↓ crossed below signal."}.get(
            macd_s, "No MACD crossover."
        )

        with st.expander("🔎 Why This Signal?"):
            st.write(f"- **MA:**  {ma_txt}")
            st.write(f"- **RSI:** {rsi_txt}")
            st.write(f"- **MACD:** {macd_txt}")
            if include_bb and "BB_Signal" in df_sig.columns:
                bb_s = int(last.get("BB_Signal", 0))
                bb_txt = {1: "Close under lower band (mean-revert long).", 0: "Inside bands.", -1: "Close over upper band (mean-revert short)."}.get(
                    bb_s, "Inside bands."
                )
                st.write(f"- **BB:** {bb_txt}")
            st.write(f"- **Composite:** {float(last.get('Composite', 0)):.2f} (threshold={comp_thr:.1f})")
            st.write(
                f"- **MTF:** Daily {daily_c:+.2f} vs Hourly {hourly_c:+.2f} → {'✅ agree' if agree else '⚠️ disagree'}"
            )
            st.write(f"- **Regime edge:** {regime:+.2f}  |  **Composite pct:** {comp_pct:.2f}")

        # Metrics
        bh_last = float(df_c["CumBH"].tail(1).iloc[0]) if "CumBH" in df_c and not df_c["CumBH"].empty else 1.0
        strat_last = float(df_c["CumStrat"].tail(1).iloc[0]) if "CumStrat" in df_c and not df_c["CumStrat"].empty else 1.0
        colA, colB, colC, colD, colE, colF = st.columns(6)
        colA.metric("CAGR", f"{(0 if np.isnan(cagr) else cagr):.2f}%")
        colB.metric("Sharpe", f"{(0 if np.isnan(sharpe) else sharpe):.2f}")
        colC.metric("Max DD", f"{max_dd:.2f}%")
        colD.metric("Win Rate", f"{win_rt:.1f}%")
        colE.metric("Trades", f"{trades}")
        colF.metric("Time in Mkt", f"{tim:.1f}%")

        st.markdown(f"- **Buy & Hold:** {(bh_last - 1) * 100:.2f}%  \n- **Strategy:** {(strat_last - 1) * 100:.2f}%")

        # Plots
        idx = df_c.index
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
        ax1.plot(idx, df_c["Close"], label="Close")
        if f"MA{ma_window}" in df_c:
            ax1.plot(idx, df_c[f"MA{ma_window}"], label=f"MA{ma_window}")
        if include_bb and {"BB_U", "BB_L"}.issubset(df_c.columns):
            ax1.plot(idx, df_c["BB_U"], label="BB Upper")
            ax1.plot(idx, df_c["BB_L"], label="BB Lower")
        ax1.legend()
        ax1.set_title("Price & Indicators")
        if "Composite" in df_c:
            ax2.bar(idx, df_c["Composite"])
            ax2.set_title("Composite (weighted)")
        else:
            ax2.set_title("Composite (no data)")
        ax3.plot(idx, df_c["CumBH"], ":", label="BH")
        ax3.plot(idx, df_c["CumStrat"], "-", label="Strat")
        ax3.legend()
        ax3.set_title("Equity")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

    # Multi-Timeframe Confirmation (explicit if/else to avoid docstring dump)
    st.markdown("---")
    with st.expander("⏱️ Multi-Timeframe Confirmation", expanded=False):
        mtf_symbol = st.text_input("Symbol (MTF)", value=ticker or "AAPL", key="inp_mtf_symbol")
        if st.button("🔍 Check MTF", key="btn_mtf"):
            try:
                d_d, d_h, ok = mtf_signs(mtf_symbol)
                st.write(f"**Daily composite:** {d_d:+.2f}")
                st.write(f"**Hourly composite:** {d_h:+.2f}")
                if ok:
                    st.success("✅ Signals agree")
                else:
                    st.warning("⚠️ Signals disagree")
            except Exception as e:
                st.error(f"MTF error: {e}")


# ───────────────────────────── PLAYBOOK (Daily) ─────────────────────────────
with tab_play:
    st.title("📋 Daily Playbook — What to Trade Today")

    universe = st.text_area(
        "Tickers (comma-separated)",
        "AAPL, MSFT, NVDA, TSLA, AMZN, GOOGL, META, NFLX, SPY, QQQ",
        key="ta_play_universe",
    ).upper()
    min_conf = st.slider("Min confidence to include", 0, 100, 55, key="play_min_conf")
    if st.button("⚡ Build Playbook", key="btn_build_playbook"):
        rows = []
        tickers = [t.strip() for t in universe.split(",") if t.strip()]
        for t in tickers:
            try:
                px = load_prices(t, period_sel, interval_sel)
                if px.empty:
                    continue
                ind = compute_indicators(px, ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=True)
                sig = build_composite(
                    ind,
                    ma_window,
                    rsi_period,
                    use_weighted=use_weighted,
                    w_ma=w_ma,
                    w_rsi=w_rsi,
                    w_macd=w_macd,
                    w_bb=w_bb,
                    include_bb=include_bb,
                    threshold=comp_thr,
                    allow_short=allow_short,
                )
                if sig.empty:
                    continue
                comp = float(sig["Composite"].tail(1).iloc[0])
                side = 1 if comp >= comp_thr else (-1 if comp <= -comp_thr else 0)

                # Confidence & earnings penalty
                d_d, d_h, ok = mtf_signs(t)
                cp = composite_percentile(sig)
                reg = regime_edge_simple(px)
                dte_ts = next_earnings_date(t)
                dte = None
                if dte_ts is not None:
                    now = pd.Timestamp.utcnow().tz_localize("UTC")
                    if dte_ts >= now:
                        dte = int((dte_ts - now).days)
                conf = confidence_score(sig, px, ok, cp, reg, dte)

                if conf < min_conf or side == 0:
                    continue

                # Rough edge score for sort
                edge = (abs(comp) * 0.5) + (reg * 0.5)
                rows.append(
                    {
                        "Ticker": t,
                        "Side": "LONG" if side > 0 else "SHORT",
                        "Composite": comp,
                        "MTF agree": ok,
                        "Regime edge": reg,
                        "Confidence": conf,
                        "Days→Earnings": dte,
                        "EdgeScore": edge,
                    }
                )
            except Exception:
                continue

        if rows:
            dfp = pd.DataFrame(rows).sort_values(["Confidence", "EdgeScore"], ascending=[False, False])
            st.dataframe(dfp.drop(columns=["EdgeScore"]).set_index("Ticker"), use_container_width=True)
            st.caption("Playbook excludes low-confidence or neutral names and applies an earnings proximity penalty.")
        else:
            st.info("No names met the confidence threshold. Loosen filters or extend history.")


# ───────────────────────────── ML LAB ─────────────────────────────
with tab_ml:
    st.title("🧠 ML Lab — Probabilistic Signals")
    if not SKLEARN_OK:
        st.warning("scikit-learn not installed. Run: pip install scikit-learn")

    symbol = st.text_input("Symbol (ML)", value="AAPL", key="inp_ml_symbol").upper()
    horizon = st.slider("Prediction horizon (bars)", 1, 5, 1, key="ml_horizon")
    train_frac = st.slider("Train fraction", 0.5, 0.95, 0.8, key="ml_train_frac")
    proba_enter = st.slider("Enter if P(long) ≥", 0.50, 0.80, 0.55, 0.01, key="ml_p_enter")
    proba_exit = st.slider("Enter short if P(long) ≤", 0.20, 0.50, 0.45, 0.01, key="ml_p_exit")
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
        if {"BB_U", "BB_L"}.issubset(d.columns):
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
            if ind.empty:
                st.error("Not enough data for indicators.")
                st.stop()
            X = _ml_features(ind)
            y = (ind["Close"].pct_change(horizon).shift(-horizon) > 0).reindex(X.index).astype(int)
            data = pd.concat([X, y.rename("y")], axis=1).dropna()
            if len(data) < 200:
                st.warning("Not enough rows for ML. Try longer history or daily interval.")
                st.stop()
            split = int(len(data) * float(train_frac))
            train, test = data.iloc[:split], data.iloc[split:]
            clf = RandomForestClassifier(n_estimators=400, max_depth=6, random_state=42, n_jobs=-1)
            clf.fit(train.drop(columns=["y"]), train["y"])
            proba = clf.predict_proba(test.drop(columns=["y"]))[:, 1]
            y_true = test["y"].values
            acc = accuracy_score(y_true, (proba > 0.5).astype(int))
            try:
                auc = roc_auc_score(y_true, proba)
            except Exception:
                auc = np.nan

            st.subheader("Out-of-sample performance")
            c1, c2 = st.columns(2)
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
            bt, md, sh, wr, trd, tim, cagr = backtest(
                ml_df,
                allow_short=allow_short,
                cost_bps=cost_bps,
                sl_atr_mult=sl_atr_mult,
                tp_atr_mult=tp_atr_mult,
                vol_target=vol_target,
                interval=interval_sel,
            )
            st.markdown(
                f"**ML Strategy OOS:** Return={(bt['CumStrat'].iloc[-1]-1)*100:.2f}% | Sharpe={sh:.2f} | MaxDD={md:.2f}% | Trades={trd}"
            )
            fig, ax = plt.subplots(figsize=(9, 3))
            ax.plot(bt.index, bt["CumBH"], ":", label="BH")
            ax.plot(bt.index, bt["CumStrat"], label="ML Strat")
            ax.legend()
            ax.set_title("ML OOS Equity")
            st.pyplot(fig)

            latest_p = clf.predict_proba(data.drop(columns=["y"]).tail(1))[:, 1][0]
            st.info(f"Latest P(long) = {latest_p:.3f}")
        except Exception as e:
            st.error(f"ML error: {e}")


# ───────────────────────────── SCANNER ─────────────────────────────
with tab_scan:
    st.title("📡 Universe Scanner — Composite + (optional) ML")
    universe = st.text_area(
        "Tickers (comma-separated)",
        "AAPL, MSFT, NVDA, TSLA, AMZN, GOOGL, META, NFLX, SPY, QQQ",
        key="ta_scan_universe",
    ).upper()
    use_ml_scan = st.toggle("Include ML probability (needs scikit-learn)", value=False, key="tg_ml_scan")
    run_scan = st.button("🔎 Scan", key="btn_scan")

    if run_scan:
        rows = []
        tickers = [t.strip() for t in universe.split(",") if t.strip()]
        for t in tickers:
            try:
                px = load_prices(t, period_sel, interval_sel)
                if px.empty:
                    continue
                ind = compute_indicators(px, ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=True)
                sig = build_composite(
                    ind,
                    ma_window,
                    rsi_period,
                    use_weighted=use_weighted,
                    w_ma=w_ma,
                    w_rsi=w_rsi,
                    w_macd=w_macd,
                    w_bb=w_bb,
                    include_bb=include_bb,
                    threshold=comp_thr,
                    allow_short=allow_short,
                )
                if sig.empty:
                    continue
                comp = float(sig["Composite"].tail(1).iloc[0]) if "Composite" in sig else 0.0
                rec = rec_map.get(int(np.sign(comp)), "🟡 HOLD")
                mlp = np.nan
                if use_ml_scan and SKLEARN_OK:
                    X = pd.DataFrame(index=ind.index)
                    X["ret1"] = ind["Close"].pct_change()
                    X["rsi"] = ind.get(f"RSI{rsi_period}", np.nan)
                    X["macd"] = ind.get("MACD", np.nan)
                    X = X.dropna()
                    y = (ind["Close"].pct_change().shift(-1) > 0).reindex(X.index).astype(int)
                    if len(X) > 200 and y.notna().sum() > 100:
                        split = int(len(X) * 0.8)
                        clf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=0)
                        clf.fit(X.iloc[:split], y.iloc[:split])
                        mlp = float(clf.predict_proba(X.iloc[split:])[-1, 1])
                rows.append({"Ticker": t, "Composite": comp, "Signal": rec, "ML P(long)": mlp})
            except Exception:
                continue
        if rows:
            df = pd.DataFrame(rows).set_index("Ticker").sort_values(["Signal", "Composite"], ascending=[True, False])
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
            if ind.empty:
                st.error("Not enough data.")
                st.stop()
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
                lab = (q1.mean(axis=1) > 0.66).astype(int) + (q1.mean(axis=1) < 0.33).astype(int) * 2
            reg = pd.Series(lab, index=feat.index, name="Regime")
            joined = ind.join(reg, how="right")
            ret = joined["Close"].pct_change().groupby(joined["Regime"]).mean().sort_values()
            ord_map = {old: i for i, old in enumerate(ret.index)}
            joined["Regime"] = joined["Regime"].map(ord_map)
            st.dataframe(joined[["Close", "Regime"]].tail(10))
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(joined.index, joined["Close"], label="Close")
            for r in sorted(joined["Regime"].dropna().unique()):
                seg = joined[joined["Regime"] == r]
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
            rets = []
            valid = []
            for t in tickers:
                px = load_prices(t, "1y", "1d")
                if px.empty:
                    continue
                valid.append(t)
                rets.append(px["Close"].pct_change().dropna())
            if not rets:
                st.error("No valid tickers/data.")
                st.stop()
            R = pd.concat(rets, axis=1)
            R.columns = valid
            cov = R.cov()
            n = len(valid)
            w = np.ones(n) / n
            for _ in range(500):
                mrc = cov @ w
                rc = w * mrc
                target = rc.mean()
                grad = rc - target
                w = np.clip(w - 0.05 * grad, 0, None)
                s = w.sum()
                w = w / s if s > 1e-12 else np.ones(n) / n
                if np.linalg.norm(grad) < 1e-6:
                    break
            weights = pd.Series(w, index=valid, name="Weight")
            st.dataframe(weights.to_frame().T, use_container_width=True)
            fig, ax = plt.subplots(figsize=(5, 5))
            weights.plot.pie(autopct="%.1f%%", ax=ax)
            ax.set_ylabel("")
            ax.set_title("Risk-Parity Weights")
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
            sig = build_composite(
                ind,
                ma_window,
                rsi_period,
                use_weighted=use_weighted,
                w_ma=w_ma,
                w_rsi=w_rsi,
                w_macd=w_macd,
                w_bb=w_bb,
                include_bb=include_bb,
                threshold=comp_thr,
                allow_short=allow_short,
            )
            bt, *_ = backtest(
                sig,
                allow_short=allow_short,
                cost_bps=cost_bps,
                sl_atr_mult=sl_atr_mult,
                tp_atr_mult=tp_atr_mult,
                vol_target=vol_target,
                interval="1d",
            )
            r = bt["StratRet"].dropna().values
            if len(r) < 50:
                st.warning("Not enough strategy bars to bootstrap.")
                st.stop()
            N = len(r)
            endings = []
            for _ in range(int(n_paths)):
                samp = np.random.choice(r, size=N, replace=True)
                eq = (1 + pd.Series(samp)).cumprod().iloc[-1]
                endings.append(eq)
            endings = np.array(endings)
            pct = (np.percentile(endings, [5, 25, 50, 75, 95]) - 1) * 100
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("P5%", f"{pct[0]:.1f}%")
            c2.metric("P25%", f"{pct[1]:.1f}%")
            c3.metric("Median", f"{pct[2]:.1f}%")
            c4.metric("P75%", f"{pct[3]:.1f}%")
            c5.metric("P95%", f"{pct[4]:.1f}%")
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.hist((endings - 1) * 100, bins=30, alpha=0.8)
            ax.set_title("Monte Carlo: Distribution of End Returns (%)")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Monte Carlo error: {e}")


# ───────────────────────────── HELP ─────────────────────────────
with tab_help:
    st.header("How QuantaraX Pro Works")
    st.markdown(
        r"""
**QuantaraX Pro v9** adds investor-ready UX and safety rails:

- **Confidence Score (0-100)** blending composite strength, MTF agreement, regime edge, freshness, and earnings proximity penalty.
- **Signal Cards** with a simple **ATR sizing assistant** based on your account equity & risk per trade.
- **Daily Playbook** that filters names by confidence and excludes those too close to earnings.
- **Hardened data loading** (auto-adjust, retries), robust **earnings parsing**, safe news w/ RSS fallback.
- Everything continues to be guarded for empty data / short histories to avoid runtime errors.

> Educational use only. Backtests are not guarantees. Always manage risk.
"""
    )
