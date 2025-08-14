# app.py — QuantaraX Pro (v23, bearish + ETF/factor)
# ---------------------------------------------------------------------------------
# pip install:
#   streamlit yfinance pandas numpy matplotlib feedparser vaderSentiment scikit-learn

import math
import time
import warnings
from typing import List, Tuple, Union
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.simplefilter("ignore", FutureWarning)

# Optional ML imports (app runs without sklearn)
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.inspection import permutation_importance
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


# ───────────────────────────── Page Setup / Theme ─────────────────────────────
st.set_page_config(page_title="QuantaraX Pro v23", layout="wide")
analyzer = SentimentIntensityAnalyzer()
rec_map = {1: "🟢 BUY", 0: "🟡 HOLD", -1: "🔴 SELL"}


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
allow_short = st.sidebar.toggle("Allow shorts", value=True, key="allow_short")
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
    s = (sym or "").strip().upper()
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

def ensure_utc_index(px: pd.DataFrame) -> pd.DataFrame:
    if px is None or px.empty:
        return px
    out = px.copy()
    out.index = pd.to_datetime(out.index, utc=True)
    return out

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
            time.sleep(0.8*(attempt+1))
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

def fetch_earnings_dates(symbol: str, limit: int = 16) -> pd.DataFrame:
    try:
        cal = yf.Ticker(_map_symbol(symbol)).get_earnings_dates(limit=limit)
        if isinstance(cal, pd.DataFrame) and not cal.empty:
            df = cal.copy()
            if isinstance(df.index, pd.DatetimeIndex):
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
            df = df.rename(columns={date_col: "earn_ts"})
            df["earn_ts"] = pd.to_datetime(df["earn_ts"], utc=True, errors="coerce")
            return df.dropna(subset=["earn_ts"]).sort_values("earn_ts")
    except Exception:
        pass
    return pd.DataFrame()

def human_when(ts: pd.Timestamp, now=None) -> str:
    now = now or pd.Timestamp.now(tz="UTC")
    if ts.tz is None: ts = ts.tz_localize("UTC")
    delta = ts - now
    s = int(delta.total_seconds()); sign = "in " if s >= 0 else ""; s = abs(s)
    d, r = divmod(s, 86400); h, r = divmod(r, 3600); m, _ = divmod(r, 60)
    if d>0: return f"{sign}{d}d {h}h"
    if h>0: return f"{sign}{h}h {m}m"
    return f"{sign}{m}m"

def render_next_earnings(symbol: str) -> pd.DataFrame:
    er = fetch_earnings_dates(symbol, limit=16)
    if er.empty:
        st.info("📅 Earnings: unavailable"); return pd.DataFrame()
    now = pd.Timestamp.now(tz="UTC")
    upcoming = er[er["earn_ts"] >= now]
    if not upcoming.empty:
        nxt = upcoming.iloc[0]["earn_ts"]
        st.info(f"📅 Next earnings: **{nxt.date()}** ({human_when(nxt, now)})")
    else:
        last = er[er["earn_ts"] < now].tail(1)
        if not last.empty:
            st.info(f"📅 Last earnings: {last['earn_ts'].iloc[0].date()} (no upcoming date found)")
        else:
            st.info("📅 Earnings: unavailable")
    return er

def data_health(px: pd.DataFrame, interval: str) -> dict:
    if px is None or px.empty: return {"fresh": False, "ago_h": float("nan")}
    px = ensure_utc_index(px)
    last_ts = px.index.max(); now = pd.Timestamp.now(tz="UTC")
    fresh_hours = max(0.0, (now - last_ts).total_seconds()/3600.0)
    if interval == "1d": return {"fresh": fresh_hours < 30, "ago_h": fresh_hours}
    if interval == "1h": return {"fresh": fresh_hours < 3, "ago_h": fresh_hours}
    return {"fresh": fresh_hours < 24, "ago_h": fresh_hours}


# ─────────── Indicators / Composite ───────────
def compute_indicators(df: pd.DataFrame, ma_w: int, rsi_p: int, mf: int, ms: int, sig: int,
                       use_bb: bool = True) -> pd.DataFrame:
    d = df.copy()
    if d.empty or not {"Open","High","Low","Close"}.issubset(d.columns): return pd.DataFrame()

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
    up_move = d["High"].diff(); dn_move = -d["Low"].diff()
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
    d["KC_U"] = ema_mid + 2 * d["ATR"]; d["KC_L"] = ema_mid - 2 * d["ATR"]

    return d.dropna()

def build_composite(df: pd.DataFrame, ma_w: int, rsi_p: int,
                    *, use_weighted=True, w_ma=1.0, w_rsi=1.0, w_macd=1.0, w_bb=0.5,
                    include_bb=True, threshold=0.0, allow_short=False) -> pd.DataFrame:
    if df.empty: return df.copy()
    d = df.copy(); n = len(d)
    close = d["Close"].to_numpy(); ma = d[f"MA{ma_w}"].to_numpy()
    rsi   = d[f"RSI{rsi_p}"].to_numpy(); macd = d["MACD"].to_numpy(); sigl = d["MACD_Signal"].to_numpy()

    ma_sig = np.zeros(n, int); rsi_sig = np.zeros(n, int); macd_sig2 = np.zeros(n, int); bb_sig = np.zeros(n, int)
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

    if allow_short: trade = np.where(comp >= threshold, 1, np.where(comp <= -threshold, -1, 0))
    else:           trade = np.where(comp >= threshold, 1, 0)

    d["MA_Signal"], d["RSI_Signal"], d["MACD_Signal2"] = ma_sig, rsi_sig, macd_sig2
    if include_bb: d["BB_Signal"] = bb_sig
    d["Composite"] = comp.astype(float); d["Trade"] = trade.astype(int)
    return d


# ─────────── Backtest ───────────
def _stats_from_equity(d: pd.DataFrame, interval: str) -> Tuple[float,float,float,float,int,float,float]:
    ann = 252 if interval == "1d" else 252*6
    if d["CumStrat"].notna().any():
        dd = d["CumStrat"]/d["CumStrat"].cummax() - 1
        max_dd = float(dd.min()*100); last_cum = float(d["CumStrat"].dropna().iloc[-1])
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
        sk["CumBH"] = 1.0; sk["CumStrat"] = 1.0
        return sk, 0.0, np.nan, np.nan, 0, 0.0, np.nan

    d["Return"] = d["Close"].pct_change().fillna(0.0)

    if allow_short:
        d["Position"] = d.get("Trade", 0).shift(1).fillna(0).clip(-1,1)
        base_ret = np.where(d["Position"]>=0, d["Return"], -d["Return"])
    else:
        d["Position"] = d.get("Trade", 0).shift(1).fillna(0).clip(0,1)
        base_ret = d["Position"] * d["Return"]

    if vol_target and vol_target > 0:
        look = 20; daily_vol = d["Return"].rolling(look).std(ddof=0)
        ann = 252 if interval == "1d" else 252*6
        realized = daily_vol * math.sqrt(ann)
        scale = (vol_target / realized).clip(0, 3.0).fillna(0.0)
        base_ret = base_ret * scale

    cost = cost_bps/10000.0
    pos_change = d["Position"].diff().fillna(0).abs()
    tcost = -2.0*cost*(pos_change > 0).astype(float)
    d["StratRet"] = pd.Series(base_ret, index=d.index).fillna(0.0) + tcost

    if (sl_atr_mult>0 or tp_atr_mult>0) and "ATR" in d.columns:
        flat = np.zeros(len(d), dtype=int); entry = np.nan
        for i in range(len(d)):
            p, c = d["Position"].iat[i], d["Close"].iat[i]
            a = d["ATR"].iat[i] if "ATR" in d.columns else np.nan
            if p != 0 and np.isnan(entry): entry = c
            if p == 0: entry = np.nan
            if p != 0 and not np.isnan(a):
                if p == 1 and (c <= entry - sl_atr_mult*a or c >= entry + tp_atr_mult*a): flat[i] = 1; entry = np.nan
                if p == -1 and (c >= entry + sl_atr_mult*a or c <= entry - tp_atr_mult*a): flat[i] = 1; entry = np.nan
        if flat.any(): d.loc[flat==1, "Position"] = 0

    ret_bh = d["Return"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ret_st = d["StratRet"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    d["CumBH"] = (1 + ret_bh).cumprod()
    d["CumStrat"] = (1 + ret_st).cumprod()

    max_dd, sharpe, win_rt, trades, tim, cagr, _ = _stats_from_equity(d, interval)
    return d, max_dd, sharpe, win_rt, trades, tim, cagr


# ─────────── Earnings event study (UTC-safe) ───────────
def earnings_event_study(px: pd.DataFrame, earn_df: pd.DataFrame, window_pre=5, window_post=5) -> dict:
    if px.empty or earn_df.empty or "earn_ts" not in earn_df.columns: return {}
    px = ensure_utc_index(px); closes = px["Close"].copy()
    opens = px["Open"].copy() if "Open" in px else None

    dates = pd.to_datetime(earn_df["earn_ts"], utc=True).dropna().sort_values().unique()
    if len(dates) == 0: return {}

    paths, gaps, d1 = [], [], []
    for d in dates:
        loc = px.index.searchsorted(pd.Timestamp(d))
        if loc >= len(px.index): continue
        i0 = max(0, loc-5); i1 = min(len(px)-1, loc+5)
        seg_close = closes.iloc[i0:i1+1]
        if len(seg_close) < 11: continue
        base = seg_close.iloc[4] if 5>0 else seg_close.iloc[0]
        norm = (seg_close / float(base) - 1.0) * 100.0
        paths.append(norm.reset_index(drop=True))

        if opens is not None and loc>0:
            prev_close = closes.iloc[loc-1]; open_px = opens.iloc[loc]
            if pd.notna(prev_close) and pd.notna(open_px) and prev_close != 0:
                gaps.append((open_px/prev_close - 1.0)*100.0)

        if loc + 1 < len(px):
            nxt = (closes.iloc[loc+1] / closes.iloc[loc] - 1.0) * 100.0
            if pd.notna(nxt): d1.append(float(nxt))

    if not paths: return {}
    M = pd.concat(paths, axis=1).mean(axis=1)
    return {
        "mean_path_pct": M,
        "gap_mean_pct": float(np.nanmean(gaps)) if gaps else np.nan,
        "gap_std_pct":  float(np.nanstd(gaps))  if gaps else np.nan,
        "next_day_mean_pct": float(np.nanmean(d1)) if d1 else np.nan,
        "next_day_hit_rate": float(np.mean(np.array(d1)>0)*100.0) if d1 else np.nan,
        "n_events": len(paths),
    }


# ─────────── Factor exposures (Y/M/W/D lookback) ───────────
def _parse_lookback_offset(lookback) -> Union[pd.DateOffset, pd.Timedelta]:
    if isinstance(lookback, str):
        m = re.fullmatch(r"\s*(\d+)\s*([yYmMwWdD])\s*", lookback)
        if m:
            n = int(m.group(1)); u = m.group(2).lower()
            if u == "y": return pd.DateOffset(years=n)
            if u == "m": return pd.DateOffset(months=n)
            if u == "w": return pd.DateOffset(weeks=n)
            return pd.Timedelta(days=n)
        if lookback.isdigit(): return pd.Timedelta(days=int(lookback))
    elif isinstance(lookback, (int, float)):
        return pd.Timedelta(days=int(lookback))
    return pd.DateOffset(years=3)

def factor_exposures(px: pd.DataFrame, lookback="3y") -> pd.DataFrame:
    proxies = {"SPY":"MKT","QQQ":"TECH","IWM":"SMB","TLT":"RATES","HYG":"CREDIT","UUP":"USD","GLD":"GOLD"}
    if px.empty or "Close" not in px: return pd.DataFrame()
    px = ensure_utc_index(px); end = px.index.max(); beg = end - _parse_lookback_offset(lookback)
    px = px.loc[(px.index >= beg) & (px.index <= end)].copy()
    r_y = px["Close"].pct_change().dropna()
    if r_y.empty: return pd.DataFrame()

    rows=[]
    for tkr, name in proxies.items():
        ref = load_prices(tkr, "5y", "1d")
        if ref.empty or "Close" not in ref: continue
        ref = ensure_utc_index(ref)
        r = ref["Close"].pct_change()
        df = pd.concat([r_y, r], axis=1, join="inner").dropna()
        if df.empty: continue
        Y = df.iloc[:,0].values.reshape(-1,1); X = df.iloc[:,1].values.reshape(-1,1)
        Xc = np.column_stack([np.ones(len(X)), X])
        beta, *_ = np.linalg.lstsq(Xc, Y, rcond=None)
        resid = Y - Xc @ beta
        denom = (Y - Y.mean()).var() * len(Y)
        R2 = float(1 - (resid**2).sum()/denom) if denom != 0 else np.nan
        rows.append((name, float(beta[1]), float(beta[0]), R2))
    if not rows: return pd.DataFrame()
    return pd.DataFrame(rows, columns=["Factor","Beta","Alpha","R2"]).set_index("Factor")


# ─────────── Breadth monitor ───────────
def breadth_from_universe(tickers: List[str], period: str, interval: str,
                          ma_w: int, rsi_p: int, comp_thr: float) -> dict:
    up = 0; down = 0; total = 0
    for t in tickers:
        px = load_prices(t, period, interval)
        if px.empty: continue
        ind = compute_indicators(px, ma_w, rsi_p, macd_fast, macd_slow, macd_signal, use_bb=True)
        sig = build_composite(ind, ma_w, rsi_p, use_weighted=True, w_ma=1.0, w_rsi=1.0, w_macd=1.0, w_bb=0.5,
                              include_bb=True, threshold=comp_thr, allow_short=True)
        if sig.empty or "Composite" not in sig: continue
        last = float(sig["Composite"].iloc[-1]); total += 1
        if last >= comp_thr: up += 1
        elif last <= -comp_thr: down += 1
    pct_up = 100*up/max(total,1); pct_down = 100*down/max(total,1)
    return {"n": total, "bullish": up, "bearish": down, "pct_up": pct_up, "pct_down": pct_down}


# ───────────────────────────── Tabs ─────────────────────────────
TAB_TITLES = ["🚀 Engine", "🧠 ML Lab", "📡 Scanner", "📉 Regimes", "💼 Portfolio", "❓ Help"]
(tab_engine, tab_ml, tab_scan, tab_regime, tab_port, tab_help) = st.tabs(TAB_TITLES)


# ───────────────────────────── ENGINE ─────────────────────────────
with tab_engine:
    st.title("🚀 QuantaraX — Decision Engine (v23)")

    col_inp, col_mode = st.columns([3,1])
    with col_inp:
        ticker = st.text_input("Symbol (e.g., AAPL or BTC/USDT)", "AAPL", key="inp_engine_ticker").upper()
    with col_mode:
        st.caption("Mode affects how much is shown by default.")
        st.selectbox("Mode", ["Beginner","Advanced"], index=0, key="ui_mode")

    # Live + freshness
    px_live = load_prices(ticker, "10d", "1d")
    if not px_live.empty and "Close" in px_live:
        last_px = _to_float(px_live["Close"].iloc[-1])
        meta = data_health(px_live, "1d")
        c1, c2 = st.columns([1,1])
        c1.metric("💲 Last Close", f"${last_px:.2f}")
        c2.metric("⏱ Freshness", f"{'Fresh' if meta['fresh'] else 'Stale'} • {meta['ago_h']:.1f}h ago")

    # Earnings + News
    earnings_df = render_next_earnings(ticker)

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
            st.markdown(f"- [{t_}]({l_}) {emoji}"); shown += 1
            if shown >= 5: break
    else:
        rss = rss_news(ticker, limit=5)
        if rss:
            st.markdown("#### 📰 Recent News (RSS Fallback)")
            for r in rss: st.markdown(f"- [{r['title']}]({r['link']})")
        else:
            st.info("No recent news found.")

    # Run backtest
    if st.button("▶️ Run Composite Backtest", key="btn_engine_backtest"):
        px = load_prices(ticker, period_sel, interval_sel)
        if px.empty: st.error(f"No data for '{ticker}'"); st.stop()
        df_raw = compute_indicators(px, ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=include_bb)
        if df_raw.empty: st.error("Not enough data after indicators."); st.stop()

        df_sig = build_composite(df_raw, ma_window, rsi_period, use_weighted=use_weighted, w_ma=w_ma, w_rsi=w_rsi,
                                 w_macd=w_macd, w_bb=w_bb, include_bb=include_bb, threshold=comp_thr, allow_short=allow_short)
        if df_sig.empty: st.error("Composite could not be built."); st.stop()

        df_c, max_dd, sharpe, win_rt, trades, tim, cagr = backtest(
            df_sig, allow_short=allow_short, cost_bps=cost_bps, sl_atr_mult=sl_atr_mult,
            tp_atr_mult=tp_atr_mult, vol_target=vol_target, interval=interval_sel
        )

        # Long/Short readout
        comp_last = float(df_sig["Composite"].iloc[-1])
        long_sig  = "🟢 BUY" if comp_last >= comp_thr else ("🟡 HOLD" if comp_last > -comp_thr else "🔴 SELL")
        short_sig = "🔴 SHORT" if comp_last <= -comp_thr else ("🟡 HOLD" if comp_last < comp_thr else "🟢 AVOID")
        colL, colS = st.columns(2)
        colL.success(f"Long Bias: {long_sig}  •  score={comp_last:.2f}")
        colS.warning(f"Short Bias: {short_sig}  •  score={comp_last:.2f}")

        # Explain
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
        bh_last    = float(df_c["CumBH"].tail(1).iloc[0])  if "CumBH" in df_c else 1.0
        strat_last = float(df_c["CumStrat"].tail(1).iloc[0]) if "CumStrat" in df_c else 1.0
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
        if "Composite" in df_c: ax2.bar(idx, df_c["Composite"]); ax2.set_title("Composite (weighted)")
        else: ax2.set_title("Composite (no data)")
        ax3.plot(idx, df_c["CumBH"], ":", label="BH"); ax3.plot(idx, df_c["CumStrat"], "-", label="Strat"); ax3.legend(); ax3.set_title("Equity")
        plt.xticks(rotation=45); plt.tight_layout(); st.pyplot(fig)

    # Extra tools
    st.markdown("---")
    with st.expander("⏱️ Multi-Timeframe Confirmation", expanded=False):
        mtf_symbol = st.text_input("Symbol (MTF)", value=ticker or "AAPL", key="inp_mtf_symbol")
        if st.button("🔍 Check MTF", key="btn_mtf"):
            try:
                d1 = compute_indicators(load_prices(mtf_symbol, "1y", "1d"), ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=True)
                dH = compute_indicators(load_prices(mtf_symbol, "30d", "1h"), ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=True)
                if d1.empty or dH.empty: st.warning("Insufficient data for MTF."); st.stop()
                c1 = build_composite(d1, ma_window, rsi_period, use_weighted=True, w_ma=1.0, w_rsi=1.0, w_macd=1.0, w_bb=0.5, include_bb=True, threshold=1.0)
                cH = build_composite(dH, ma_window, rsi_period, use_weighted=True, w_ma=1.0, w_rsi=1.0, w_macd=1.0, w_bb=0.5, include_bb=True, threshold=1.0)
                daily  = float(c1["Composite"].iloc[-1]); hourly = float(cH["Composite"].iloc[-1])
                st.write(f"**Daily composite:** {daily:.2f}"); st.write(f"**Hourly composite:** {hourly:.2f}")
                if np.sign(daily) == np.sign(hourly): st.success("✅ Signals agree")
                else: st.warning("⚠️ Signals disagree")
            except Exception as e:
                st.error(f"MTF error: {e}")

    with st.expander("📈 Earnings Radar — historical behavior", expanded=False):
        try:
            px_e = load_prices(ticker, "2y", "1d")
            if not px_e.empty and not earnings_df.empty:
                study = earnings_event_study(px_e, earnings_df, window_pre=5, window_post=5)
                if study:
                    st.write(f"Events used: **{study['n_events']}**, Avg gap: **{study['gap_mean_pct']:.2f}%** (σ {study['gap_std_pct']:.2f}%), Next-day mean: **{study['next_day_mean_pct']:.2f}%** (hit-rate {study['next_day_hit_rate']:.1f}%)")
                    series = study["mean_path_pct"]
                    fig, ax = plt.subplots(figsize=(8,3))
                    ax.plot(range(-5,6), series.values); ax.axvline(0, linestyle="--", alpha=0.5)
                    ax.set_xticks(range(-5,6)); ax.set_title("Average % path around earnings (T=0)"); st.pyplot(fig)
                else: st.info("Earnings study unavailable (not enough aligned data).")
            else: st.info("Earnings study unavailable.")
        except Exception as e:
            st.info(f"Earnings study unavailable: {e}")

    with st.expander("📦 Factor Box — exposures via ETF proxies", expanded=False):
        try:
            px_f = load_prices(ticker, "5y", "1d")
            if not px_f.empty:
                fb = factor_exposures(px_f, lookback="3y")
                if not fb.empty: st.dataframe(fb.round(3), use_container_width=True)
                else: st.info("Factor box unavailable (insufficient overlap with proxies).")
            else: st.info("Factor box unavailable (no price data).")
        except Exception as e:
            st.info(f"Factor box unavailable: {e}")


# ───────────────────────────── SCANNER ─────────────────────────────
with tab_scan:
    st.title("📡 Universe Scanner — Long & Short + (optional) ML")

    universe = st.text_area("Tickers (comma-separated)",
        "AAPL, MSFT, NVDA, TSLA, AMZN, GOOGL, META, NFLX, SPY, QQQ",
        key="ta_scan_universe").upper()
    use_ml_scan = st.toggle("Include ML probability (needs scikit-learn)", value=False, key="tg_ml_scan")
    run_scan = st.button("🔎 Scan", key="btn_scan")

    # ETF & Sector quick scanner
    with st.expander("🧺 ETF & Sector Scanner", expanded=False):
        etfs = "SPY, QQQ, IWM, DIA, XLK, XLF, XLY, XLP, XLE, XLV, XLI, XLU, XLB, XLRE, XLC, TLT, HYG, UUP, GLD, SLV, USO"
        etf_txt = st.text_area("ETF list", etfs, key="ta_etf_list").upper()
        if st.button("📊 Scan ETFs", key="btn_etf_scan"):
            rows=[]
            for t in [x.strip() for x in etf_txt.split(",") if x.strip()]:
                try:
                    px = load_prices(t, "1y", "1d")
                    ind = compute_indicators(px, ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=True)
                    sig = build_composite(ind, ma_window, rsi_period, use_weighted=True, w_ma=1.0, w_rsi=1.0, w_macd=1.0, w_bb=0.5, include_bb=True, threshold=comp_thr, allow_short=True)
                    comp = float(sig["Composite"].iloc[-1]) if not sig.empty else 0.0
                    rows.append({"ETF":t, "Composite":comp, "Bias":"Bull" if comp>=comp_thr else ("Bear" if comp<=-comp_thr else "Neutral")})
                except Exception:
                    continue
            if rows:
                df = pd.DataFrame(rows).set_index("ETF").sort_values("Composite", ascending=False)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No ETF results.")

    if run_scan:
        rows=[]
        tickers = [t.strip() for t in universe.split(",") if t.strip()]
        for t in tickers:
            try:
                px = load_prices(t, period_sel, interval_sel)
                if px.empty: continue
                ind = compute_indicators(px, ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=True)
                sig = build_composite(ind, ma_window, rsi_period, use_weighted=use_weighted, w_ma=w_ma, w_rsi=w_rsi, w_macd=w_macd, w_bb=w_bb, include_bb=include_bb, threshold=comp_thr, allow_short=True)
                if sig.empty: continue
                comp = float(sig["Composite"].iloc[-1]) if "Composite" in sig else 0.0
                mlp = np.nan
                if use_ml_scan and SKLEARN_OK:
                    X = pd.DataFrame(index=ind.index)
                    X["ret1"] = ind["Close"].pct_change(); X["rsi"] = ind.get(f"RSI{rsi_period}", np.nan); X["macd"] = ind.get("MACD", np.nan)
                    X = X.dropna(); y = (ind["Close"].pct_change().shift(-1) > 0).reindex(X.index).astype(int)
                    if len(X) > 200 and y.notna().sum() > 100:
                        split = int(len(X)*0.8)
                        clf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=0)
                        clf.fit(X.iloc[:split], y.iloc[:split])
                        mlp = float(clf.predict_proba(X.iloc[split:])[-1,1])
                rows.append({"Ticker":t, "Composite":comp, "Bias":"Bull" if comp>=comp_thr else ("Bear" if comp<=-comp_thr else "Neutral"), "ML P(long)":mlp})
            except Exception:
                continue
        if rows:
            df = pd.DataFrame(rows)
            long_tbl  = df[df["Composite"] >=  comp_thr].sort_values("Composite", ascending=False).set_index("Ticker")
            short_tbl = df[df["Composite"] <= -comp_thr].sort_values("Composite").set_index("Ticker")
            neutral   = df[(df["Composite"] > -comp_thr) & (df["Composite"] < comp_thr)].set_index("Ticker")

            st.subheader("🟢 Long Candidates"); 
            st.dataframe(long_tbl if not long_tbl.empty else pd.DataFrame({"msg":["None hit the long threshold"]}), use_container_width=True)
            st.subheader("🔴 Short Candidates");
            st.dataframe(short_tbl if not short_tbl.empty else pd.DataFrame({"msg":["None hit the short threshold"]}), use_container_width=True)
            with st.expander("➖ Neutral (watchlist)"):
                st.dataframe(neutral.sort_values("Composite", ascending=False), use_container_width=True)

            # Breadth
            br = breadth_from_universe(tickers, period_sel, interval_sel, ma_window, rsi_period, comp_thr)
            st.markdown(f"**Breadth:** {br['pct_up']:.1f}% bullish, {br['pct_down']:.1f}% bearish (universe n={br['n']})")
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
            ax.set_title("Price with Regime Shading"); st.pyplot(fig)
        except Exception as e:
            st.error(f"Regime error: {e}")


# ───────────────────────────── PORTFOLIO ─────────────────────────────
with tab_port:
    st.title("💼 Portfolio — Optimizers, Rotation & Monte Carlo")

    st.subheader("⚖️ Risk Parity Optimizer")
    opt_tickers = st.text_input("Tickers (comma-sep)", "AAPL, MSFT, TSLA, SPY, QQQ", key="inp_opt_tickers").upper()
    if st.button("🧮 Optimize (Risk Parity)", key="btn_opt_rp"):
        try:
            tickers = [t.strip() for t in opt_tickers.split(",") if t.strip()]
            rets = []; valid = []
            for t in tickers:
                px = load_prices(t, "1y", "1d")
                if px.empty: continue
                valid.append(t); rets.append(px["Close"].pct_change().dropna())
            if not rets: st.error("No valid tickers/data."); st.stop()
            R = pd.concat(rets, axis=1); R.columns = valid
            cov = R.cov()
            n = len(valid); w = np.ones(n)/n
            for _ in range(500):
                mrc = cov @ w; rc  = w * mrc; target = rc.mean()
                grad = rc - target; w = np.clip(w - 0.05*grad, 0, None)
                s = w.sum(); w = w / s if s > 1e-12 else np.ones(n)/n
                if np.linalg.norm(grad) < 1e-6: break
            weights = pd.Series(w, index=valid, name="Weight")
            st.dataframe(weights.to_frame().T, use_container_width=True)
            fig, ax = plt.subplots(figsize=(5,5))
            weights.plot.pie(autopct="%.1f%%", ax=ax); ax.set_ylabel(""); ax.set_title("Risk-Parity Weights"); st.pyplot(fig)
        except Exception as e:
            st.error(f"Optimizer error: {e}")

    st.subheader("🔁 ETF Rotation (momentum + composite)")
    ro_etfs = st.text_input("ETF universe", "SPY, QQQ, IWM, TLT, HYG, GLD, UUP, XLE, XLK, XLV", key="inp_rot_etfs").upper()
    topN = st.slider("Hold top-N", 1, 6, 3, key="rot_topN")
    if st.button("Run Rotation", key="btn_rot"):
        try:
            rows=[]
            for t in [x.strip() for x in ro_etfs.split(",") if x.strip()]:
                px = load_prices(t, "1y", "1d")
                if px.empty: continue
                ind = compute_indicators(px, ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=True)
                if ind.empty: continue
                comp = build_composite(ind, ma_window, rsi_period, use_weighted=True, w_ma=1.0, w_rsi=1.0, w_macd=1.0, w_bb=0.5, include_bb=True, threshold=1.0)
                score = float(comp["Composite"].iloc[-1]) if not comp.empty else 0.0
                mom60 = px["Close"].pct_change(60).iloc[-1] if len(px) > 60 else np.nan
                rows.append({"ETF":t, "Momentum60d":mom60, "Composite":score, "Blend": (0.7*(mom60 if pd.notna(mom60) else 0)+0.3*score)})
            if rows:
                df = pd.DataFrame(rows).set_index("ETF").sort_values("Blend", ascending=False)
                st.dataframe(df.round(3), use_container_width=True)
                picks = df.head(int(topN)).index.tolist()
                st.success(f"Suggested rotation picks: {', '.join(picks)}")
            else:
                st.info("No ETF data.")
        except Exception as e:
            st.error(f"Rotation error: {e}")

    st.subheader("🎲 Monte Carlo (Bootstrap) of Strategy Returns")
    mc_symbol = st.text_input("Symbol (MC)", value="AAPL", key="inp_mc_symbol").upper()
    n_paths = st.slider("Paths", 200, 3000, 800, 100, key="mc_paths")
    if st.button("Run Monte Carlo", key="btn_mc"):
        try:
            px = load_prices(mc_symbol, "2y", "1d")
            ind = compute_indicators(px, ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=True)
            sig = build_composite(ind, ma_window, rsi_period, use_weighted=use_weighted, w_ma=w_ma, w_rsi=w_rsi, w_macd=w_macd, w_bb=w_bb,
                                  include_bb=include_bb, threshold=comp_thr, allow_short=allow_short)
            bt, *_ = backtest(sig, allow_short=allow_short, cost_bps=cost_bps, sl_atr_mult=sl_atr_mult, tp_atr_mult=tp_atr_mult, vol_target=vol_target, interval="1d")
            r = bt["StratRet"].dropna().values
            if len(r) < 50: st.warning("Not enough strategy bars to bootstrap."); st.stop()
            N = len(r); endings = []
            for _ in range(int(n_paths)):
                samp = np.random.choice(r, size=N, replace=True)
                eq = (1 + pd.Series(samp)).cumprod().iloc[-1]; endings.append(eq)
            endings = np.array(endings)
            pct = (np.percentile(endings, [5,25,50,75,95]) - 1) * 100
            c1,c2,c3,c4,c5 = st.columns(5)
            c1.metric("P5%",  f"{pct[0]:.1f}%"); c2.metric("P25%", f"{pct[1]:.1f}%"); c3.metric("Median", f"{pct[2]:.1f}%"); c4.metric("P75%", f"{pct[3]:.1f}%"); c5.metric("P95%", f"{pct[4]:.1f}%")
            fig, ax = plt.subplots(figsize=(8,3)); ax.hist((endings-1)*100, bins=30, alpha=0.8); ax.set_title("Monte Carlo: Distribution of End Returns (%)"); st.pyplot(fig)
        except Exception as e:
            st.error(f"Monte Carlo error: {e}")


# ───────────────────────────── HELP ─────────────────────────────
with tab_help:
    st.header("How to Use QuantaraX Pro (v23)")
    st.markdown("""
**Mission:** turn raw market data into **actionable long/short decisions** you can test, explain, and scale.

**Key ideas:**
- A **Composite** score blends MA/RSI/MACD (+ optional Bollinger) events.
- We report **both** sides: **Long Bias** and **Short Bias** from the *same* composite.
- Everything is UTC-safe; earnings handling shows **upcoming** if available, otherwise last one.

### Engine
- Hit **Run Composite Backtest** → you get **Long** & **Short** bias, full metrics (Sharpe, MaxDD, WinRate, Trades, Time in Mkt, CAGR), and charts.
- **Why This Signal** breaks down each driver (great for beginners and PM sign-offs).
- **Multi-Timeframe**: Daily vs Hourly alignment (✅ agree = trend alignment; ⚠ disagree = chop/transition).
- **Earnings Radar**: average path around earnings (T±5), gap stats, next-day odds.
- **Factor Box**: ordinary-least-squares vs proxies (MKT/TECH/SMB/RATES/CREDIT/USD/GOLD). Use it to sanity-check exposures.

### Scanner
- Splits the universe into **🟢 Long** and **🔴 Short** candidates using your threshold (±`Composite`).
- (Optional) **ML P(long)** if scikit-learn is installed.
- **Breadth Monitor**: % of names tripping bullish vs bearish in your universe.
- **ETF & Sector Scanner**: one-click pulse for SPY/QQQ/IWM, sector ETFs (XLK, XLF, …), and macro ETFs (TLT, HYG, UUP, GLD, SLV, USO).

### Regimes
- Clusters price behavior into 3 states using vol/momentum/MA slope; shaded on chart. Handy for flipping between **trend** and **mean-reversion** playbooks.

### Portfolio
- **Risk Parity**: spread risk evenly across chosen tickers.
- **ETF Rotation**: ranks ETFs by a blend of 60-day momentum (70%) and composite (30%); pick top-N.
- **Monte Carlo**: bootstraps your strategy returns to visualize distribution of outcomes.

### Practical playbook
- **Beginners:** Start with Engine → read *Why This Signal*, look for MTF agreement, avoid trading right into earnings gaps.
- **Active traders:** Use Scanner’s long/short lists + Breadth to size risk. Confirm with Factor Box (avoid unintended macro bets).
- **PMs / investors:** Show Regimes & Earnings Radar for narrative; run Monte Carlo to set expectations; Risk Parity / Rotation to deploy.

> Educational & experimental. Always layer risk management, use limits, and diversify sources.
""")
