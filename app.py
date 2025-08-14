# app.py — QuantaraX Pro v27 (all-in-one, hardened)
# ---------------------------------------------------------------------------------
# pip install:
#   streamlit yfinance pandas numpy matplotlib feedparser vaderSentiment scikit-learn
# Optional for PDF export:
#   reportlab
#
# Notes:
# - All network calls are guarded. If a provider times out, UI degrades gracefully.
# - Beginner/Pro mode switch is in the sidebar only (not shown in main UI).

import io
import math
import time
import warnings
from typing import List, Tuple, Optional

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import feedparser
from urllib.parse import quote_plus
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.simplefilter("ignore", FutureWarning)

# Optional ML imports
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.inspection import permutation_importance
    from sklearn.cluster import KMeans
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

# Optional PDF export
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

# ───────────────────────────── Page Setup ─────────────────────────────
st.set_page_config(page_title="QuantaraX Pro v27", layout="wide")
analyzer = SentimentIntensityAnalyzer()

rec_map = {1: "🟢 BUY", 0: "🟡 HOLD", -1: "🔴 SELL"}

TAB_TITLES = ["🚀 Engine", "🧠 ML Lab", "📡 Scanner", "📉 Regimes", "💼 Portfolio", "❓ Help"]
(tab_engine, tab_ml, tab_scan, tab_regime, tab_port, tab_help) = st.tabs(TAB_TITLES)

# ───────────────────────────── Sidebar (unique keys) ─────────────────────────────
st.sidebar.header("Mode & Global Controls")

# Beginner/Pro switch (not displayed on main page)
user_mode = st.sidebar.radio("Experience mode", ["Beginner", "Pro"], index=0, key="mode_select")

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

# ───────────────────────────── Utilities ─────────────────────────────
def _utcnow_ts() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")

def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    idx = df.index
    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is None:
            df = df.tz_localize("UTC")
        else:
            df = df.tz_convert("UTC")
    return df

def _map_symbol(sym: str) -> str:
    s = sym.strip().upper()
    if "/" in s:  # e.g., BTC/USDT → BTC-USD
        base, quote = s.split("/")
        quote = "USD" if quote in ("USDT","USD") else quote
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
                df = df.dropna()
                return _ensure_utc_index(df)
        except Exception:
            time.sleep(0.8 * (attempt + 1))
    return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=600)
def fetch_news_bundle(symbol: str, max_items: int = 12) -> List[dict]:
    """
    Robust multi-source news:
      1) yfinance ticker.news
      2) Yahoo Finance RSS
      3) Google News RSS (query: '<SYM> stock')
    Returns: list of dicts {title, link, summary, source, ts, sentiment}
    """
    out = []
    sym = _map_symbol(symbol)

    # 1) yfinance news
    try:
        ny = getattr(yf.Ticker(sym), "news", []) or []
        for a in ny:
            title = a.get("title", "")
            link  = a.get("link", "")
            if not title or not link:
                continue
            summary = a.get("summary", title)
            ts = a.get("providerPublishTime", None)
            if isinstance(ts, (int, float)):
                ts = pd.to_datetime(ts, unit="s", utc=True)
            else:
                ts = None
            out.append({"title": title, "link": link, "summary": summary, "source": a.get("publisher") or "Yahoo", "ts": ts})
    except Exception:
        pass

    # 2) Yahoo RSS
    try:
        rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={sym}&region=US&lang=en-US"
        feed = feedparser.parse(rss_url)
        for e in feed.entries:
            title = getattr(e, "title", "")
            link  = getattr(e, "link", "")
            if not title or not link:
                continue
            summary = getattr(e, "summary", title)
            ts = None
            if getattr(e, "published_parsed", None):
                ts = pd.to_datetime(time.mktime(e.published_parsed), unit="s", utc=True)
            out.append({"title": title, "link": link, "summary": summary, "source": "Yahoo RSS", "ts": ts})
    except Exception:
        pass

    # 3) Google News RSS
    try:
        q = quote_plus(f"{symbol} stock")
        g_url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(g_url)
        for e in feed.entries:
            title = getattr(e, "title", "")
            link  = getattr(e, "link", "")
            if not title or not link:
                continue
            summary = getattr(e, "summary", title)
            ts = None
            if getattr(e, "published_parsed", None):
                ts = pd.to_datetime(time.mktime(e.published_parsed), unit="s", utc=True)
            out.append({"title": title, "link": link, "summary": summary, "source": "Google News", "ts": ts})
    except Exception:
        pass

    if not out:
        return []

    # De-dup by link (fallback: title)
    seen = set(); dedup = []
    for a in out:
        key = a["link"] or a["title"]
        if key in seen:
            continue
        seen.add(key)
        dedup.append(a)

    # Sentiment + sort newest first
    for a in dedup:
        txt = a.get("summary") or a.get("title") or ""
        a["sentiment"] = analyzer.polarity_scores(txt)["compound"]
    dedup.sort(key=lambda x: (x["ts"] is not None, x["ts"]), reverse=True)
    return dedup[:max_items]

def render_news(symbol: str, expand: bool = True):
    items = fetch_news_bundle(symbol)
    if not items:
        st.info("📰 No recent news from providers.")
        return
    with st.expander("📰 News & Sentiment (multi-source)", expanded=expand):
        for a in items:
            title = a["title"]; link = a["link"]; src = a.get("source", "News"); ts = a.get("ts", None)
            age = ""
            if isinstance(ts, pd.Timestamp):
                delta = _utcnow_ts() - ts
                hrs = int(delta.total_seconds() // 3600)
                age = ("· just now" if hrs < 1 else (f"· {hrs}h ago" if hrs < 24 else f"· {hrs//24}d ago"))
            s = a.get("sentiment", 0.0)
            emoji = "🔺" if s>0.10 else ("🔻" if s<-0.10 else "➖")
            st.markdown(f"- [{title}]({link}) {emoji} — *{src}* {age}")

def safe_earnings(symbol: str) -> pd.DataFrame:
    """
    Returns DataFrame with normalized 'earn_date' (UTC) + 'is_estimate' column if present.
    Only returns rows with valid dates.
    """
    try:
        cal = yf.Ticker(_map_symbol(symbol)).get_earnings_dates(limit=16)
        if isinstance(cal, pd.DataFrame) and not cal.empty:
            df = cal.copy()
            # Promote index date if applicable
            if isinstance(df.index, pd.DatetimeIndex) or (
                df.index.name and "date" in str(df.index.name).lower()
            ):
                df = df.reset_index()
            # Find date column
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
            if "is_estimate" not in df.columns:
                df["is_estimate"] = False
            return df.dropna(subset=["earn_date"])
    except Exception:
        pass
    return pd.DataFrame()

def next_earnings_date(symbol: str) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp], bool]:
    """Return (next_upcoming, last_reported, next_is_estimate)."""
    df = safe_earnings(symbol)
    if df.empty:
        return None, None, False
    now = _utcnow_ts()
    upcoming = df[df["earn_date"] >= now].sort_values("earn_date")
    if not upcoming.empty:
        row = upcoming.iloc[0]
        return row["earn_date"], df["earn_date"].max(), bool(row.get("is_estimate", False))
    # No future → show last reported
    last_row = df.sort_values("earn_date").iloc[-1]
    return None, last_row["earn_date"], bool(last_row.get("is_estimate", False))

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
            a = d["ATR"].iat[i]
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

# ─────────── Confidence scoring ───────────
def confidence_score(
    comp_value: float,
    comp_max: float,
    mtf_agree: Optional[bool],
    news_items: List[dict],
    in_good_regime: Optional[bool],
) -> int:
    """Blend composite magnitude, MTF agreement, average news sentiment, regime."""
    s_comp = min(1.0, abs(comp_value) / max(comp_max, 1e-9))
    s_mtf = 1.0 if mtf_agree else 0.0 if mtf_agree is not None else 0.5
    if news_items:
        avg_sent = float(np.mean([a.get("sentiment", 0.0) for a in news_items]))
        s_news = (avg_sent + 1) / 2  # [-1,1] -> [0,1]
    else:
        s_news = 0.5
    s_reg = 1.0 if in_good_regime else 0.0 if in_good_regime is not None else 0.5
    score = 100 * (0.45*s_comp + 0.25*s_mtf + 0.20*s_news + 0.10*s_reg)
    return int(round(max(0, min(100, score))))

# ─────────── Factor / ETF exposures ───────────
@st.cache_data(show_spinner=False, ttl=1200)
def factor_lens(symbol: str, lookback="1y") -> Optional[pd.Series]:
    """
    Quick-n-dirty factor loads vs ETFs: SPY (MKT), IWM (SMB), IWD (Value), MTUM (Momentum).
    Returns beta series indexed by ['SPY','IWM','IWD','MTUM'] or None.
    """
    tickers = ["SPY","IWM","IWD","MTUM"]
    data = {}
    for t in [symbol] + tickers:
        px = load_prices(t, period=lookback, interval="1d")
        if px.empty: return None
        data[t] = px["Close"].pct_change().dropna()
    df = pd.concat([data[symbol]] + [data[t] for t in tickers], axis=1).dropna()
    df.columns = ["asset"] + tickers
    # OLS: asset ~ factors
    X = df[tickers].values
    Y = df["asset"].values
    X = np.c_[np.ones(len(X)), X]  # intercept
    try:
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        b = pd.Series(beta[1:], index=tickers, name="Beta")
        return b
    except Exception:
        return None

# ─────────── PDF/HTML Report ───────────
def build_report(symbol: str, advice_rows: List[dict], context_lines: List[str]) -> Tuple[str, bytes, str]:
    """
    Build a PDF if reportlab available; otherwise HTML.
    Returns (mime, content_bytes, filename)
    """
    ts = pd.Timestamp.now().strftime("%Y-%m-%d_%H%M")
    title = f"QuantaraX Report — {symbol} — {ts}"
    if REPORTLAB_OK:
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=letter)
        width, height = letter
        y = height - 50
        c.setFont("Helvetica-Bold", 14); c.drawString(40, y, title); y -= 24
        c.setFont("Helvetica", 10)
        for line in context_lines:
            for chunk in [line[i:i+95] for i in range(0, len(line), 95)]:
                c.drawString(40, y, chunk); y -= 14
                if y < 60: c.showPage(); y = height - 50
        y -= 10
        c.setFont("Helvetica-Bold", 12); c.drawString(40, y, "Positions & Suggestions"); y -= 18
        c.setFont("Helvetica", 10)
        for r in advice_rows:
            s = f"{r['Ticker']}: {r['Suggestion']} | P/L%={r.get('P/L %', np.nan):.2f} | Composite={r.get('Composite', np.nan):.2f} | Reason: {r.get('Reason','')}"
            for chunk in [s[i:i+95] for i in range(0, len(s), 95)]:
                c.drawString(40, y, chunk); y -= 14
                if y < 60: c.showPage(); y = height - 50
        c.showPage(); c.save()
        buf.seek(0)
        return "application/pdf", buf.read(), f"QuantaraX_{symbol}_{ts}.pdf"
    else:
        html = [f"<h2>{title}</h2>"]
        for line in context_lines:
            html.append(f"<p>{line}</p>")
        html.append("<h3>Positions & Suggestions</h3><ul>")
        for r in advice_rows:
            html.append(
                f"<li><b>{r['Ticker']}</b>: {r['Suggestion']} | P/L%={r.get('P/L %', np.nan):.2f} "
                f"| Composite={r.get('Composite', np.nan):.2f} | Reason: {r.get('Reason','')}</li>"
            )
        html.append("</ul>")
        content = "\n".join(html).encode("utf-8")
        return "text/html", content, f"QuantaraX_{symbol}_{ts}.html"

# ───────────────────────────── ENGINE ─────────────────────────────
with tab_engine:
    st.title("🚀 QuantaraX — Composite Signal Engine")

    st.markdown("### Single‐Ticker Backtest")
    ticker = st.text_input("Ticker (e.g. AAPL or BTC/USDT)", "AAPL", key="inp_engine_ticker").upper()

    # Live (last close)
    px_live = load_prices(ticker, "5d", "1d")
    if not px_live.empty and "Close" in px_live:
        last_px = _to_float(px_live["Close"].iloc[-1])
        st.subheader(f"💲 Last close: ${last_px:.2f}")

    # News (robust)
    render_news(ticker, expand=(user_mode == "Beginner"))

    # Earnings (upcoming only; else last reported)
    next_e, last_e, is_est = next_earnings_date(ticker)
    if next_e is not None:
        tag = " (est.)" if is_est else ""
        st.info(f"📅 Next Earnings: **{next_e.date()}**{tag}")
    elif last_e is not None:
        st.info(f"📅 Last Earnings: **{last_e.date()}** · Next not available")
    else:
        st.info("📅 Earnings: unavailable")

    # Run backtest
    if st.button("▶️ Run Composite Backtest", key="btn_engine_backtest"):
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

        df_c, max_dd, sharpe, win_rt, trades, tim, cagr = backtest(
            df_sig, allow_short=allow_short, cost_bps=cost_bps,
            sl_atr_mult=sl_atr_mult, tp_atr_mult=tp_atr_mult, vol_target=vol_target, interval=interval_sel
        )

        # Last signal + confidence
        last_trade = int(df_sig["Trade"].tail(1).iloc[0]) if "Trade" in df_sig.columns and not df_sig.empty else 0
        rec = rec_map.get(1 if last_trade>0 else (-1 if last_trade<0 else 0), "🟡 HOLD")

        # MTF agree (daily vs hourly)
        d1 = compute_indicators(load_prices(ticker, "1y", "1d"), ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=True)
        dH = compute_indicators(load_prices(ticker, "30d", "1h"), ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=True)
        mtf_agree = None
        if not d1.empty and not dH.empty:
            c1 = build_composite(d1, ma_window, rsi_period, use_weighted=True, w_ma=1.0, w_rsi=1.0, w_macd=1.0, w_bb=0.5, include_bb=True, threshold=1.0)
            cH = build_composite(dH, ma_window, rsi_period, use_weighted=True, w_ma=1.0, w_rsi=1.0, w_macd=1.0, w_bb=0.5, include_bb=True, threshold=1.0)
            mtf_agree = int(np.sign(c1["Composite"].iloc[-1])) == int(np.sign(cH["Composite"].iloc[-1]))

        # Regime check (quick 3-cluster by vol/mom/MA slope; "green" if best)
        in_good_regime = None
        try:
            ind_rg = compute_indicators(load_prices(ticker, "2y", "1d"), ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=False)
            if not ind_rg.empty:
                feat = pd.DataFrame(index=ind_rg.index)
                feat["vol20"] = ind_rg["Close"].pct_change().rolling(20).std()
                feat["mom20"] = ind_rg["Close"].pct_change(20)
                feat["ma_slope"] = ind_rg[f"MA{ma_window}"].diff()
                feat = feat.dropna()
                if SKLEARN_OK and len(feat) >= 60:
                    km = KMeans(n_clusters=3, n_init=10, random_state=42)
                    lab = pd.Series(km.fit_predict(feat), index=feat.index)
                else:
                    q1 = feat.rank(pct=True)
                    lab = (q1.mean(axis=1) > 0.66).astype(int) + (q1.mean(axis=1) < 0.33).astype(int)*2
                joined = ind_rg.join(lab.rename("Regime"), how="right")
                ret = joined["Close"].pct_change().groupby(joined["Regime"]).mean().sort_values()
                ord_map = {old:i for i, old in enumerate(ret.index)}  # 0=worst → 2=best
                cur_r = ord_map.get(lab.iloc[-1], None)
                in_good_regime = (cur_r == 2) if cur_r is not None else None
        except Exception:
            pass

        comp_val = float(df_sig["Composite"].iloc[-1])
        comp_max = (w_ma + w_rsi + w_macd + (w_bb if include_bb else 0.0)) if use_weighted else 3.0
        news_items = fetch_news_bundle(ticker)[:5]
        conf = confidence_score(comp_val, comp_max, mtf_agree, news_items, in_good_regime)

        st.success(f"**{ticker}**: {rec}  ·  Confidence: **{conf}/100**")

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
            st.write(f"- **Composite (weighted):** {comp_val:.2f}  (threshold={comp_thr:.1f})")
            if mtf_agree is not None:
                st.write("- **MTF:** " + ("Daily and Hourly agree ✅" if mtf_agree else "Daily and Hourly disagree ⚠️"))
            if in_good_regime is not None:
                st.write("- **Regime:** " + ("Favorable (green) ✅" if in_good_regime else "Unfavorable (red) ⚠️"))

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
        ax2.bar(idx, df_c["Composite"]); ax2.set_title("Composite (weighted)")
        ax3.plot(idx, df_c["CumBH"], ":", label="BH")
        ax3.plot(idx, df_c["CumStrat"], "-", label="Strat"); ax3.legend(); ax3.set_title("Equity")
        plt.xticks(rotation=45); plt.tight_layout()
        st.pyplot(fig)

    # Extra tools
    st.markdown("---")
    with st.expander("⏱️ Multi-Timeframe Confirmation", expanded=False):
        mtf_symbol = st.text_input("Symbol (MTF)", value=ticker or "AAPL", key="inp_mtf_symbol")
        if st.button("🔍 Check MTF", key="btn_mtf"):
            try:
                d1 = compute_indicators(load_prices(mtf_symbol, "1y", "1d"), ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=True)
                dH = compute_indicators(load_prices(mtf_symbol, "30d", "1h"), ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=True)
                if d1.empty or dH.empty:
                    st.warning("Insufficient data for MTF."); st.stop()
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

    with st.expander("🧭 ETF / Factor Lens", expanded=False):
        fac_symbol = st.text_input("Symbol (Factor Lens)", value=ticker or "AAPL", key="inp_factor_symbol")
        if st.button("Run Factor Lens", key="btn_factor"):
            b = factor_lens(fac_symbol, lookback="1y")
            if b is None or b.empty:
                st.warning("Could not compute factor loadings.")
            else:
                st.dataframe(b.to_frame().T, use_container_width=True)
                fig, ax = plt.subplots(figsize=(6,3))
                b.plot(kind="bar", ax=ax); ax.set_title("Factor Loadings vs ETFs"); plt.tight_layout()
                st.pyplot(fig)

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
                px_full = load_prices(wf_symbol, "2y", "1d")
                if px_full.empty: st.warning("No data for WFO."); st.stop()

                def run_eq(ma_list, rsi_list, mf_list, ms_list, sig_list):
                    oos_curves = []; summary = []
                    start=200; i=start
                    while i + ins_bars + oos_bars <= len(px_full):
                        ins = px_full.iloc[i : i+ins_bars]
                        oos = px_full.iloc[i+ins_bars : i+ins_bars+oos_bars]
                        best=None; best_score=-1e9
                        for mw in ma_list:
                            for rp in rsi_list:
                                for mf in mf_list:
                                    for ms in ms_list:
                                        for s in sig_list:
                                            ins_ind = compute_indicators(ins, mw, rp, mf, ms, s, use_bb=True)
                                            if ins_ind.empty: continue
                                            ins_sig = build_composite(ins_ind, mw, rp, use_weighted=True, w_ma=1.0, w_rsi=1.0, w_macd=1.0, w_bb=0.5, include_bb=True, threshold=w_thr, allow_short=wf_allow_short)
                                            ins_bt, md, sh, *_ = backtest(ins_sig, allow_short=wf_allow_short, cost_bps=5.0)
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
                        oos_sig = build_composite(oos_ind, mw, rp, use_weighted=True, w_ma=1.0, w_rsi=1.0, w_macd=1.0, w_bb=0.5, include_bb=True, threshold=w_thr, allow_short=wf_allow_short)
                        oos_bt, mo_dd, mo_sh, *_ = backtest(oos_sig, allow_short=wf_allow_short, cost_bps=5.0)
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
            df_tc = build_composite(
                df_t, ma_window, rsi_period,
                use_weighted=use_weighted, w_ma=w_ma, w_rsi=w_rsi, w_macd=w_macd, w_bb=w_bb,
                include_bb=include_bb, threshold=comp_thr, allow_short=allow_short
            )
            if df_tc.empty: continue
            bt, md, sh, wr, trd, tim, cagr = backtest(
                df_tc, allow_short=allow_short, cost_bps=cost_bps,
                sl_atr_mult=sl_atr_mult, tp_atr_mult=tp_atr_mult,
                vol_target=vol_target, interval=interval_sel
            )
            comp_last = float(df_tc["Composite"].tail(1).iloc[0]) if "Composite" in df_tc else 0.0
            bh_last = float(bt["CumBH"].tail(1).iloc[0]) if "CumBH" in bt else 1.0
            strat_last = float(bt["CumStrat"].tail(1).iloc[0]) if "CumStrat" in bt else 1.0
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
            st.download_button("Download CSV", df_perf.to_csv(), "batch.csv", key="dl_batch")
        else:
            st.error("No valid data for batch tickers.")

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
            if SKLEARN_OK and len(feat) >= 60:
                km = KMeans(n_clusters=3, n_init=10, random_state=42)
                lab = km.fit_predict(feat)
            else:
                q1 = feat.rank(pct=True)
                lab = (q1.mean(axis=1) > 0.66).astype(int) + (q1.mean(axis=1) < 0.33).astype(int)*2
            reg = pd.Series(lab, index=feat.index, name="Regime")
            joined = ind.join(reg, how="right")
            ret = joined["Close"].pct_change().groupby(joined["Regime"]).mean().sort_values()
            ord_map = {old:i for i, old in enumerate(ret.index)}  # 0=worst → 2=best
            joined["Regime"] = joined["Regime"].map(ord_map)
            st.dataframe(joined[["Close","Regime"]].tail(10))
            # Plot
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(joined.index, joined["Close"], label="Close")
            # Shade regimes
            cur = None
            for i in range(len(joined)):
                r = joined["Regime"].iloc[i]
                if pd.isna(r): continue
                if cur is None:
                    cur = (r, i)
                elif r != cur[0]:
                    seg = joined.iloc[cur[1]:i]
                    ax.axvspan(seg.index[0], seg.index[-1], alpha=0.08)
                    cur = (r, i)
            if cur is not None:
                seg = joined.iloc[cur[1]:]
                ax.axvspan(seg.index[0], seg.index[-1], alpha=0.08)
            ax.set_title("Price with Regime Shading")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Regime error: {e}")

# ───────────────────────────── PORTFOLIO ─────────────────────────────
with tab_port:
    st.title("💼 Portfolio — Optimizers, Advisor & Monte Carlo")

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

    st.subheader("📊 Portfolio Advisor (paste CSV: ticker,shares,cost_basis)")
    holdings = st.text_area("Positions CSV", "AAPL,10,150\nMSFT,5,300", height=100, key="ta_portfolio")
    if st.button("▶️ Analyze & Advise", key="btn_sim_port"):
        rows = [r.strip().split(",") for r in holdings.splitlines() if r.strip()]
        data=[]; report_rows=[]
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

            # Composite suggestion + reason
            px = load_prices(tkr, period_sel, interval_sel)
            comp_sugg="N/A"; score=np.nan; reason=""
            if not px.empty:
                df_i = compute_indicators(px, ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=include_bb)
                if not df_i.empty:
                    df_csig = build_composite(df_i, ma_window, rsi_period,
                                              use_weighted=use_weighted, w_ma=w_ma, w_rsi=w_rsi, w_macd=w_macd, w_bb=w_bb,
                                              include_bb=include_bb, threshold=comp_thr, allow_short=allow_short)
                    if not df_csig.empty:
                        score = float(df_csig["Composite"].tail(1).iloc[0]) if "Composite" in df_csig else 0.0
                        comp_sugg = "🟢 BUY" if score>=comp_thr else ("🔴 SELL" if score<=-comp_thr else "🟡 HOLD")
                        last = df_csig.tail(1).iloc[0]
                        reason = "MA:{} | RSI:{} | MACD:{}".format(int(last.get("MA_Signal",0)), int(last.get("RSI_Signal",0)), int(last.get("MACD_Signal2",0)))

            # Guardrails override
            if pnl_pct > profit_target:     suggestion="🔴 SELL"
            elif pnl_pct < -loss_limit:     suggestion="🟢 BUY"
            else:                           suggestion=comp_sugg

            rec_row = {
                "Ticker":tkr,"Shares":s,"Cost Basis":c,"Price":price,
                "Market Value":value,"Invested":invested,"P/L":pnl,
                "P/L %":pnl_pct,"Composite Sig":comp_sugg,"Suggestion":suggestion,"Composite":score,"Reason":reason
            }
            data.append(rec_row); report_rows.append(rec_row)

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

            # Downloadable report with rationale
            context = [
                f"Mode: {user_mode}",
                f"Profit target override: {profit_target}%, Loss limit override: {loss_limit}%",
                "Composite: integrates MA/RSI/MACD (+BB optional) with weights & thresholds."
            ]
            mime, content, fname = build_report("Portfolio", report_rows, context)
            st.download_button("⬇️ Download Advice Report", content, file_name=fname, mime=mime, key="dl_report")
        else:
            st.error("No valid holdings provided.")

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
    st.header("How QuantaraX Pro Works (Deep Guide)")
    st.markdown(r"""
### What you get
QuantaraX is a research & decision assistant. It **scores entries**, **backtests**, **explains** signals in plain English, and **advises portfolios** with risk controls and exportable reports.

---

### The core signal (Composite v2)
We blend four classic signals, each normalized into **+1 (bullish), 0 (neutral), -1 (bearish)**:
- **MA crossover** (Close vs. MA\*N\*)
- **RSI** (oversold <30 → +1; overbought >70 → -1)
- **MACD** (line crosses signal: up → +1, down → -1)
- **Bollinger Bands** *(optional)* (below lower → +1, above upper → -1)

Weights (MA/RSI/MACD/BB) are user-tunable. The **Composite** is their (weighted) sum; when it exceeds your **Trigger**, we enter.

---

### Confidence (0–100)
A quick-read score that blends:
1. **Composite strength** (how far from 0 relative to max possible)
2. **Multi-Timeframe agreement** (daily vs hourly sign match)
3. **News sentiment** (VADER on headlines; multi-source, deduped)
4. **Market regime** (clustered by vol/momentum/MA-slope; “green”=favorable)

> Confidence is a **heuristic**, not a guarantee.

---

### Backtesting & risk
We simulate bar-by-bar:
- **Shorts** (optional), **trading costs**, **ATR stop/target**, and **volatility targeting**.
- Key metrics: **CAGR, Sharpe, MaxDD, Win Rate, Trades, Time-in-Market**.

---

### Walk-Forward Optimization (WFO)
We sweep nearby parameters on rolling windows:
1. Find best parameters **in-sample** via a simple score (return – |drawdown|).
2. Apply them **out-of-sample**, stitch equity.
3. Review OOS performance and parameter stability.

---

### Scanner
Paste a universe; we rank by Composite and (optionally) **ML probability of an up-move** (RandomForest on simple features). This is a **quick screen**, not your final answer.

---

### ETF / Factor Lens (explain a symbol’s behavior)
We regress the symbol’s returns on ETF proxies:
- **SPY** (market), **IWM** (size), **IWD** (value), **MTUM** (momentum)
and report loadings (betas). Use this to sanity-check exposures.

---

### Portfolio Advisor
Paste **ticker, shares, cost_basis** rows. For each holding:
- We compute **P/L**, infer a latest **Composite suggestion** (Buy/Sell/Hold), and
- Apply **guardrails**: if P/L% > profit target → *trim*; if below loss limit → *buy/add or reconsider*.
- Export a **PDF/HTML report** with rationale for each position.

---

### Earnings
We pull the calendar and show only **upcoming** dates. If none available, we show the **last reported** date and say next is unavailable.

---

### Modes (Beginner vs Pro)
- **Beginner:** more explanations expanded by default; sensible defaults.
- **Pro:** same engine; you just get more room for parameter play.

---

### Tips
- For very short lookbacks or illiquid names, many indicators will drop rows. **Lengthen period** or shorten windows.
- Confidence going from 40 → 70 usually means **broader alignment** (MTF + news + regime).
- Don’t overfit WFO; look for **stable** solutions (similar params keep working).

---

### Disclaimers
This is **research software**, not investment advice. Markets are noisy; use multiple tools and judgment.
""")
