# app.py — QuantaraX Pro v29 (upload & downloads + dual confidence)
# ---------------------------------------------------------------------------------
# pip install:
#   streamlit yfinance pandas numpy matplotlib feedparser vaderSentiment scikit-learn
# Optional for PDF export:
#   reportlab

import io
import json
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
st.set_page_config(page_title="QuantaraX Pro v29", layout="wide")
analyzer = SentimentIntensityAnalyzer()

rec_map = {1: "🟢 BUY", 0: "🟡 HOLD", -1: "🔴 SELL"}

TAB_TITLES = ["🚀 Engine", "🧠 ML Lab", "📡 Scanner", "📉 Regimes", "💼 Portfolio", "❓ Help"]
(tab_engine, tab_ml, tab_scan, tab_regime, tab_port, tab_help) = st.tabs(TAB_TITLES)

# ───────────────────────────── Sidebar (unique keys) ─────────────────────────────
st.sidebar.header("Mode & Global Controls")

# Beginner/Pro switch (sidebar only, not printed in main)
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
        scale = (vol_target / realized).clip(0, 3.0).fillna(0.0)
        base_ret = base_ret * scale

    # Costs on trades
    cost = cost_bps/10000.0
    pos_change = d["Position"].diff().fillna(0).abs()
    tcost = -2.0*cost*(pos_change > 0).astype(float)
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

# ─────────── Confidence scoring (dual) ───────────
def confidence_score_side(
    comp_value: float,
    comp_max: float,
    mtf_side_agree: Optional[bool],
    news_items: List[dict],
    regime_is_good_for_side: Optional[bool],
    side: str,  # "long" or "short"
) -> int:
    """
    Side-aware confidence:
      - For LONG: use positive component of composite.
      - For SHORT: use negative component.
    """
    if side == "long":
        s_comp = min(1.0, max(0.0, comp_value) / max(comp_max, 1e-9))
    else:  # short
        s_comp = min(1.0, max(0.0, -comp_value) / max(comp_max, 1e-9))

    s_mtf = 1.0 if mtf_side_agree else 0.0 if mtf_side_agree is not None else 0.5

    if news_items:
        avg_sent = float(np.mean([a.get("sentiment", 0.0) for a in news_items]))
        # Assume good news favors LONG; bad news favors SHORT
        if side == "long":
            s_news = (avg_sent + 1) / 2  # [-1,1] -> [0,1]
        else:
            s_news = (1 - ((avg_sent + 1) / 2))
    else:
        s_news = 0.5

    s_reg = 1.0 if regime_is_good_for_side else 0.0 if regime_is_good_for_side is not None else 0.5
    score = 100 * (0.45*s_comp + 0.25*s_mtf + 0.20*s_news + 0.10*s_reg)
    return int(round(max(0, min(100, score))))

# ─────────── Factor / ETF exposures ───────────
@st.cache_data(show_spinner=False, ttl=1200)
def factor_lens(symbol: str, lookback="1y") -> Optional[pd.Series]:
    tickers = ["SPY","IWM","IWD","MTUM"]
    data = {}
    for t in [symbol] + tickers:
        px = load_prices(t, period=lookback, interval="1d")
        if px.empty: return None
        data[t] = px["Close"].pct_change().dropna()
    df = pd.concat([data[symbol]] + [data[t] for t in tickers], axis=1).dropna()
    df.columns = ["asset"] + tickers
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

        # Last signal
        last_trade = int(df_sig["Trade"].tail(1).iloc[0]) if "Trade" in df_sig.columns and not df_sig.empty else 0
        rec = rec_map.get(1 if last_trade>0 else (-1 if last_trade<0 else 0), "🟡 HOLD")

        # MTF agree → side-specific
        d1 = compute_indicators(load_prices(ticker, "1y", "1d"), ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=True)
        dH = compute_indicators(load_prices(ticker, "30d", "1h"), ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=True)
        mtf_long = None; mtf_short = None
        if not d1.empty and not dH.empty:
            c1 = build_composite(d1, ma_window, rsi_period, use_weighted=True, w_ma=1.0, w_rsi=1.0, w_macd=1.0, w_bb=0.5, include_bb=True, threshold=1.0)
            cH = build_composite(dH, ma_window, rsi_period, use_weighted=True, w_ma=1.0, w_rsi=1.0, w_macd=1.0, w_bb=0.5, include_bb=True, threshold=1.0)
            s1 = int(np.sign(c1["Composite"].iloc[-1])); sH = int(np.sign(cH["Composite"].iloc[-1]))
            mtf_long  = (s1 > 0 and sH > 0)
            mtf_short = (s1 < 0 and sH < 0)

        # Regime favorability → side-specific
        long_good = None; short_good = None
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
                if cur_r is not None:
                    long_good  = (cur_r == 2)
                    short_good = (cur_r == 0)
        except Exception:
            pass

        # Dual confidence
        comp_val = float(df_sig["Composite"].iloc[-1])
        comp_max = (w_ma + w_rsi + w_macd + (w_bb if include_bb else 0.0)) if use_weighted else 3.0
        news_items = fetch_news_bundle(ticker)[:5]
        conf_long  = confidence_score_side(comp_val, comp_max, mtf_long,  news_items, long_good,  "long")
        conf_short = confidence_score_side(comp_val, comp_max, mtf_short, news_items, short_good, "short")

        st.success(f"**{ticker}**: {rec}  ·  Confidence — Long **{conf_long}/100** | Short **{conf_short}/100**")

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
            if mtf_long is not None and mtf_short is not None:
                st.write(f"- **MTF:** Long-agree: {'✅' if mtf_long else '⚠️'} | Short-agree: {'✅' if mtf_short else '⚠️'}")
            if long_good is not None and short_good is not None:
                st.write(f"- **Regime:** Long favorable: {'✅' if long_good else '⚠️'} | Short favorable: {'✅' if short_good else '⚠️'}")

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

    # Tools
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
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(joined.index, joined["Close"], label="Close")
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

    st.subheader("📥 Upload or Paste Portfolio (ticker,shares,cost_basis)")
    col_up1, col_up2 = st.columns([2,1])
    with col_up1:
        upload = st.file_uploader("Upload CSV", type=["csv"], key="uploader_port")
    with col_up2:
        template_csv = "ticker,shares,cost_basis\nAAPL,10,150\nMSFT,5,300\n"
        st.download_button("⬇️ Template CSV", data=template_csv, file_name="portfolio_template.csv", key="dl_template_csv")

    st.caption("Or paste CSV below:")
    holdings_text = st.text_area("Positions CSV", "AAPL,10,150\nMSFT,5,300", height=120, key="ta_portfolio")

    def _parse_portfolio(upload_file, pasted_text) -> List[Tuple[str,float,float]]:
        rows = []
        if upload_file is not None:
            try:
                df = pd.read_csv(upload_file)
                cols = {c.lower().strip(): c for c in df.columns}
                # Map flexible headers
                sym_col   = next((cols[c] for c in cols if c in ["ticker","symbol"]), None)
                qty_col   = next((cols[c] for c in cols if c in ["shares","qty","quantity"]), None)
                cost_col  = next((cols[c] for c in cols if c in ["cost_basis","avg_price","price","cost"]), None)
                if not (sym_col and qty_col and cost_col):
                    st.error("CSV must include columns: ticker/symbol, shares/qty/quantity, cost_basis/avg_price.")
                    return []
                for _, r in df[[sym_col, qty_col, cost_col]].dropna().iterrows():
                    rows.append((str(r[sym_col]).strip(), float(r[qty_col]), float(r[cost_col])))
            except Exception as e:
                st.error(f"Upload parse error: {e}")
                return []
        else:
            # Parse pasted
            for line in pasted_text.splitlines():
                if not line.strip(): continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) != 3:
                    st.warning(f"Skipping invalid row: {line}")
                    continue
                try:
                    rows.append((parts[0], float(parts[1]), float(parts[2])))
                except Exception:
                    st.warning(f"Invalid numbers on row: {line}")
                    continue
        return rows

    if st.button("▶️ Analyze & Advise", key="btn_sim_port"):
        parsed = _parse_portfolio(upload, holdings_text)
        data=[]; report_rows=[]
        for idx, (ticker_, shares, cost) in enumerate(parsed, 1):
            tkr = _map_symbol(ticker_.upper().strip())
            hist = load_prices(tkr, "5d", "1d")
            if hist.empty:
                st.warning(f"No price for {tkr}"); continue
            price=_to_float(hist["Close"].iloc[-1])
            invested=shares*cost; value=shares*price; pnl=value-invested
            pnl_pct=(pnl/invested*100) if invested else np.nan

            # Composite suggestion, reason, and dual confidence
            px = load_prices(tkr, period_sel, interval_sel)
            comp_sugg="N/A"; score=np.nan; reason=""; confL=np.nan; confS=np.nan
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
                        # Dual confidence for each holding (uses cached news & regime quick check)
                        comp_max_local = (w_ma + w_rsi + w_macd + (w_bb if include_bb else 0.0)) if use_weighted else 3.0
                        # quick MTF signs
                        try:
                            d1 = compute_indicators(load_prices(tkr, "1y", "1d"), ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=True)
                            dH = compute_indicators(load_prices(tkr, "30d", "1h"), ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=True)
                            if not d1.empty and not dH.empty:
                                c1 = build_composite(d1, ma_window, rsi_period, use_weighted=True, w_ma=1.0, w_rsi=1.0, w_macd=1.0, w_bb=0.5, include_bb=True, threshold=1.0)
                                cH = build_composite(dH, ma_window, rsi_period, use_weighted=True, w_ma=1.0, w_rsi=1.0, w_macd=1.0, w_bb=0.5, include_bb=True, threshold=1.0)
                                s1 = int(np.sign(c1["Composite"].iloc[-1])); sH = int(np.sign(cH["Composite"].iloc[-1]))
                                mtf_long = (s1 > 0 and sH > 0); mtf_short = (s1 < 0 and sH < 0)
                            else:
                                mtf_long = mtf_short = None
                        except Exception:
                            mtf_long = mtf_short = None
                        # regime quick
                        long_good=short_good=None
                        try:
                            ind_rg = compute_indicators(load_prices(tkr, "2y", "1d"), ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=False)
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
                                ord_map = {old:i for i, old in enumerate(ret.index)}
                                cur_r = ord_map.get(lab.iloc[-1], None)
                                if cur_r is not None:
                                    long_good  = (cur_r == 2)
                                    short_good = (cur_r == 0)
                        except Exception:
                            pass
                        news_items = fetch_news_bundle(tkr)[:5]
                        confL = confidence_score_side(score, comp_max_local, mtf_long, news_items, long_good, "long")
                        confS = confidence_score_side(score, comp_max_local, mtf_short, news_items, short_good, "short")

            # Guardrails override final suggestion
            if pnl_pct > profit_target:     suggestion="🔴 SELL"
            elif pnl_pct < -loss_limit:     suggestion="🟢 BUY"
            else:                           suggestion=comp_sugg

            rec_row = {
                "Ticker":tkr,"Shares":shares,"Cost Basis":cost,"Price":price,
                "Market Value":value,"Invested":invested,"P/L":pnl,
                "P/L %":pnl_pct,"Composite Sig":comp_sugg,"Suggestion":suggestion,
                "Composite":score,"Conf Long":confL,"Conf Short":confS,"Reason":reason
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

            # Downloadable datasets
            st.download_button("⬇️ Download Simulated Portfolio (CSV)",
                               data=df_port.to_csv(),
                               file_name="simulated_portfolio.csv",
                               key="dl_port_csv")
            st.download_button("⬇️ Download Simulated Portfolio (JSON)",
                               data=df_port.to_json(orient="table"),
                               file_name="simulated_portfolio.json",
                               key="dl_port_json")

            # Advice report with rationale
            context = [
                f"Mode: {user_mode}",
                f"Profit target override: {profit_target}%, Loss limit override: {loss_limit}%",
                "Composite integrates MA/RSI/MACD (+BB optional) with weights & thresholds.",
                "Confidence is side-aware (Long/Short) using MTF, news, and regime."
            ]
            mime, content, fname = build_report("Portfolio", report_rows, context)
            st.download_button("⬇️ Download Advice Report (PDF/HTML)", content, file_name=fname, mime=mime, key="dl_report")
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

### Dual Confidence (0–100 for Long & Short)
Two separate confidence scores:
1. **Composite strength** toward that side (e.g., positive magnitude for Long).
2. **Multi-Timeframe agreement** (Daily and Hourly both bullish → Long; both bearish → Short).
3. **News sentiment** (VADER on headlines; positive favors Long, negative favors Short).
4. **Market regime** (clustered by vol/momentum/MA-slope; best regime favors Long; worst regime favors Short).

> Confidence is a **heuristic**, not a guarantee.

---

### Backtesting & risk
We simulate bar-by-bar:
- **Shorts** (optional), **trading costs**, **ATR stop/target**, and **volatility targeting**.
- Key metrics: **CAGR, Sharpe, MaxDD, Win Rate, Trades, Time-in-Market**.

---

### Scanner
Paste a universe; we rank by Composite and (optionally) **ML probability of an up-move** (RandomForest). Use as a **screen**, not a final decision.

---

### ETF / Factor Lens
We regress a symbol on ETF proxies (**SPY/IWM/IWD/MTUM**) to show rough factor exposures.

---

### Portfolio Advisor
Upload or paste **ticker,shares,cost_basis**. We compute **P/L**, **Composite suggestion**, **dual confidence**, and apply **guardrails**:
- If P/L% > profit target → consider **trim**.
- If P/L% < -loss limit → consider **buy/add** or reassess.
Export **CSV/JSON** datasets and a **PDF/HTML report** with reasoning.

---

### Earnings
We show **upcoming** earnings if available, else **last reported**.

---

### Modes (Beginner vs Pro)
- **Beginner:** more explanations expanded by default; sensible defaults.
- **Pro:** same engine; more freedom to tune.

---

### Tips
- Lengthen history or reduce indicator windows if you see "not enough rows".
- Confidence rising toward one side indicates more **alignment** (MTF + news + regime).
- WFO is illustrative; avoid overfitting—seek **stable** params.

---

### Disclaimers
This is **research software**, not investment advice. Markets are noisy; use multiple tools and judgment.
""")
