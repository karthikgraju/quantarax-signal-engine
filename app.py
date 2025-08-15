# app.py — QuantaraX Decision Engine (v24, consolidated & hardened)
# -----------------------------------------------------------------------------
# pip install:
#   streamlit yfinance pandas numpy matplotlib feedparser vaderSentiment
#   scikit-learn (optional for ML), reportlab (optional for PDF exports)

import io
import math
import time
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import feedparser

warnings.simplefilter("ignore", FutureWarning)

# Optional libs
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.inspection import permutation_importance
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False


# ───────────────────────────── Page & Globals ─────────────────────────────
st.set_page_config(page_title="QuantaraX — Decision Engine (v24)", layout="wide")
analyzer = SentimentIntensityAnalyzer()
rec_map = {1: "🟢 BUY", 0: "🟡 HOLD", -1: "🔴 SELL"}

# Tabs
tab_engine, tab_ml, tab_scan, tab_regime, tab_port, tab_macro, tab_help = st.tabs(
    ["🚀 Engine", "🧠 ML Lab", "📡 Scanner", "📉 Regimes", "💼 Portfolio", "🌐 Macro", "❓ Help"]
)

# ───────────────────────────── Sidebar ─────────────────────────────
st.sidebar.header("Global Controls")
DEFAULTS = dict(ma_window=10, rsi_period=14, macd_fast=12, macd_slow=26, macd_signal=9)
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.sidebar.button("🔄 Reset to defaults", key="btn_reset_defaults"):
    for k, v in DEFAULTS.items():
        st.session_state[k] = v

# Beginner / Pro
ui_mode = st.sidebar.selectbox("Mode", ["Beginner", "Pro"], index=0, key="ui_mode")

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
period_sel   = st.sidebar.selectbox("History", ["6mo","1y","2y","5y","10y"], index=1, key="period_sel")
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
            return float(x.iloc[0])
        except Exception:
            return float("nan")

def _utc_now():
    # Return tz-aware UTC timestamp
    return pd.Timestamp.utcnow().tz_localize("UTC")

@st.cache_data(show_spinner=False, ttl=900)
def load_prices(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """Robust loader with retry; auto_adjust=True to avoid warnings."""
    sym = _map_symbol(symbol)
    for attempt in range(3):
        try:
            df = yf.download(sym, period=period, interval=interval,
                             auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            if not df.empty:
                return df.dropna()
        except Exception:
            time.sleep(0.8 * (attempt + 1))
    return pd.DataFrame()

def data_health(df: pd.DataFrame, interval: str):
    """Return dict with last timestamp, recency hours and freshness label."""
    if df.empty:
        return {"last_ts": None, "hours": None, "fresh": False, "label": "No data"}
    last_ts = df.index[-1]
    # Make both tz-naive to subtract safely
    now = _utc_now().tz_convert(None)
    last = pd.Timestamp(last_ts).tz_localize(None) if pd.Timestamp(last_ts).tzinfo else pd.Timestamp(last_ts)
    fresh_hours = max(0.0, (now - last).total_seconds() / 3600.0)
    fresh = fresh_hours < (48 if interval == "1d" else 4)
    return {"last_ts": last, "hours": fresh_hours, "fresh": fresh,
            "label": "Fresh" if fresh else "Stale"}

def safe_get_news(symbol: str) -> list:
    """Try yfinance news; fallback to [] if blocked/timeouts."""
    try:
        return getattr(yf.Ticker(_map_symbol(symbol)), "news", []) or []
    except Exception:
        return []

def rss_news(symbol: str, limit: int = 6) -> list:
    try:
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={_map_symbol(symbol)}&region=US&lang=en-US"
        feed = feedparser.parse(url)
        return [{"title": e.title, "link": e.link} for e in feed.entries[:limit]]
    except Exception:
        return []

def safe_earnings_df(symbol: str) -> pd.DataFrame:
    """Return earnings DF with 'earn_date' (UTC-naive date)."""
    try:
        cal = yf.Ticker(_map_symbol(symbol)).get_earnings_dates(limit=16)
        if isinstance(cal, pd.DataFrame) and not cal.empty:
            df = cal.copy()
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
            # detect date column
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
            df = df.rename(columns={date_col:"earn_date"})
            df["earn_date"] = pd.to_datetime(df["earn_date"], errors="coerce")
            # normalize to UTC-naive DATE (drop tz + time)
            df["earn_date"] = pd.to_datetime(df["earn_date"].dt.date)
            return df.dropna(subset=["earn_date"]).sort_values("earn_date")
    except Exception:
        pass
    return pd.DataFrame()

def next_earnings_date(symbol: str):
    df = safe_earnings_df(symbol)
    if df.empty:
        return None, None
    today = pd.to_datetime(_utc_now().date())
    future = df[df["earn_date"] >= today]
    if not future.empty:
        nxt = future.iloc[0]["earn_date"]
        return pd.to_datetime(nxt).date(), False
    # otherwise last known past
    return pd.to_datetime(df.iloc[-1]["earn_date"]).date(), True


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

    # Mild extras (stable)
    klen = 14
    ll = d["Low"].rolling(klen, min_periods=klen).min()
    hh = d["High"].rolling(klen, min_periods=klen).max()
    rng = (hh - ll).replace(0, np.nan)
    d["STO_K"] = 100 * (d["Close"] - ll) / rng
    d["STO_D"] = d["STO_K"].rolling(3, min_periods=3).mean()

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

def confidence_from_composite(comp_last: float, include_bb: bool,
                              w_ma: float, w_rsi: float, w_macd: float, w_bb: float) -> Tuple[int, int]:
    max_w = (abs(w_ma) + abs(w_rsi) + abs(w_macd) + (abs(w_bb) if include_bb else 0.0))
    if max_w <= 0: return 0, 0
    norm = float(comp_last) / max_w  # [-1, +1]
    long_conf  = int(max(0.0,  norm) * 100)
    short_conf = int(max(0.0, -norm) * 100)
    return long_conf, short_conf


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
    d["CumBH"]    = (1 + ret_bh).cumprod()
    d["CumStrat"] = (1 + ret_st).cumprod()

    max_dd, sharpe, win_rt, trades, tim, cagr, _ = _stats_from_equity(d, interval)
    return d, max_dd, sharpe, win_rt, trades, tim, cagr


# ───────────────────────────── ENGINE ─────────────────────────────
with tab_engine:
    st.title("🚀 QuantaraX — Decision Engine")

    st.markdown("##### Symbol (e.g., AAPL or BTC/USDT)")
    ticker = st.text_input("", "AAPL", key="inp_engine_ticker").upper()

    px_live = load_prices(ticker, "5d", "1d")
    meta = data_health(px_live, "1d")
    colA, colB, colC = st.columns([1.2, 1, 1])
    with colA:
        if not px_live.empty:
            last_px = _to_float(px_live["Close"].iloc[-1])
            st.success(f"💲 Last Close: ${last_px:,.2f}")
        else:
            st.info("No recent price.")
    with colB:
        st.success("✅ Fresh") if meta["fresh"] else st.warning("⚠️ Stale")
    with colC:
        if meta["hours"] is not None:
            st.caption(f"⏱ {meta['hours']:.1f}h ago")

    # Earnings (future only)
    nd, was_past = next_earnings_date(ticker)
    if nd:
        if was_past:
            st.info(f"📅 Last earnings: {nd} (no upcoming date found)")
        else:
            st.info(f"📅 Next earnings: **{nd}**")

    # News
    with st.expander("📰 Recent News & Sentiment", expanded=(ui_mode=="Beginner")):
        shown = 0
        for art in safe_get_news(ticker)[:6]:
            t_ = art.get("title",""); l_ = art.get("link","")
            if not (t_ and l_): continue
            txt = art.get("summary", t_)
            score = analyzer.polarity_scores(txt)["compound"]
            emoji = "🔺" if score>0.1 else ("🔻" if score<-0.1 else "➖")
            st.markdown(f"- [{t_}]({l_}) {emoji}")
            shown += 1
        if shown == 0:
            for r in rss_news(ticker, limit=6):
                st.markdown(f"- [{r['title']}]({r['link']})")
            if shown == 0:
                st.caption("No recent headlines.")

    run_bt = st.button("▶️ Run Composite Backtest", key="btn_engine_backtest")

    if run_bt:
        px = load_prices(ticker, period_sel, interval_sel)
        if px.empty:
            st.error(f"No data for '{ticker}'"); st.stop()

        ind = compute_indicators(px, ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=include_bb)
        if ind.empty:
            st.error("Not enough data after indicators (try longer period or smaller windows)."); st.stop()

        sig = build_composite(ind, ma_window, rsi_period,
                              use_weighted=use_weighted, w_ma=w_ma, w_rsi=w_rsi, w_macd=w_macd, w_bb=w_bb,
                              include_bb=include_bb, threshold=comp_thr, allow_short=allow_short)
        bt, md, sh, wr, trd, tim, cagr = backtest(sig, allow_short=allow_short, cost_bps=cost_bps,
                                                  sl_atr_mult=sl_atr_mult, tp_atr_mult=tp_atr_mult,
                                                  vol_target=vol_target, interval=interval_sel)

        comp_last = float(sig["Composite"].tail(1).iloc[0]) if "Composite" in sig else 0.0
        long_conf, short_conf = confidence_from_composite(comp_last, include_bb, w_ma, w_rsi, w_macd, w_bb)

        # headline signals
        cL, cS = st.columns(2)
        with cL:
            st.info(f"**Long Bias:** {'🟢 BUY' if comp_last>=comp_thr else '🟡 HOLD'} • **confidence={long_conf}/100**")
        with cS:
            st.info(f"**Short Bias:** {'🔴 SELL' if comp_last<=-comp_thr else '🟡 HOLD'} • **confidence={short_conf}/100**")

        # Reasoning
        last = sig.tail(1).iloc[0]
        ma_s  = int(last.get("MA_Signal", 0))
        rsi_s = int(last.get("RSI_Signal", 0))
        macd_s= int(last.get("MACD_Signal2", 0))
        rsi_v = float(last.get(f"RSI{rsi_period}", np.nan))
        ma_txt  = {1:f"Price ↑ crossed **above** MA{ma_window}.", 0:"No MA crossover.",
                   -1:f"Price ↓ crossed **below** MA{ma_window}."}.get(ma_s, "No MA crossover.")
        rsi_txt = "RSI unavailable." if np.isnan(rsi_v) else {
            1:f"RSI ({rsi_v:.1f}) < 30 → **oversold**.",
            0:f"RSI ({rsi_v:.1f}) neutral.",
           -1:f"RSI ({rsi_v:.1f}) > 70 → **overbought**."
        }.get(rsi_s, f"RSI ({rsi_v:.1f}) neutral.")
        macd_txt= {1:"MACD ↑ crossed **above** signal.", 0:"No MACD crossover.",
                   -1:"MACD ↓ crossed **below** signal."}.get(macd_s, "No MACD crossover.")
        with st.expander("🔎 Why This Signal?", expanded=(ui_mode=="Beginner")):
            st.write(f"- **MA:**  {ma_txt}")
            st.write(f"- **RSI:** {rsi_txt}")
            st.write(f"- **MACD:** {macd_txt}")
            if include_bb and "BB_Signal" in sig.columns:
                bb_s = int(last.get("BB_Signal", 0))
                bb_txt = {1:"Close under lower band (mean-revert long).",
                          0:"Inside bands.",
                         -1:"Close over upper band (mean-revert short)."}[bb_s]
                st.write(f"- **BB:** {bb_txt}")
            st.write(f"- **Composite (weighted):** {float(last.get('Composite', 0)):.2f}  (threshold={comp_thr:.1f})")

        # Metrics
        bh_last    = float(bt["CumBH"].tail(1).iloc[0])  if "CumBH" in bt and not bt["CumBH"].empty else 1.0
        strat_last = float(bt["CumStrat"].tail(1).iloc[0]) if "CumStrat" in bt and not bt["CumStrat"].empty else 1.0
        colM = st.columns(6)
        colM[0].metric("CAGR", f"{(0 if np.isnan(cagr) else cagr):.2f}%")
        colM[1].metric("Sharpe", f"{(0 if np.isnan(sh) else sh):.2f}")
        colM[2].metric("Max DD", f"{md:.2f}%")
        colM[3].metric("Win Rate", f"{wr:.1f}%")
        colM[4].metric("Trades", f"{trd}")
        colM[5].metric("Time in Mkt", f"{tim:.1f}%")
        st.caption(f"**Buy & Hold:** {(bh_last-1)*100:.2f}% • **Strategy:** {(strat_last-1)*100:.2f}%")

        # Plots
        idx = bt.index
        fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(11,12), sharex=True)
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
        st.pyplot(fig)

    # Multi-Timeframe Confirmation (fixed output)
    st.markdown("---")
    with st.expander("⏱️ Multi-Timeframe Confirmation", expanded=False):
        mtf_symbol = st.text_input("Symbol (MTF)", value=ticker or "AAPL", key="inp_mtf_symbol")
        if st.button("🔍 Check MTF", key="btn_mtf"):
            try:
                d1 = compute_indicators(load_prices(mtf_symbol, "1y",  "1d"),
                                        ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=True)
                dH = compute_indicators(load_prices(mtf_symbol, "30d", "1h"),
                                        ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=True)
                if d1.empty or dH.empty:
                    st.warning("Insufficient data for MTF."); st.stop()
                c1 = build_composite(d1, ma_window, rsi_period,
                                     use_weighted=True, w_ma=1.0, w_rsi=1.0, w_macd=1.0, w_bb=0.5,
                                     include_bb=True, threshold=1.0)
                cH = build_composite(dH, ma_window, rsi_period,
                                     use_weighted=True, w_ma=1.0, w_rsi=1.0, w_macd=1.0, w_bb=0.5,
                                     include_bb=True, threshold=1.0)
                daily  = float(c1["Composite"].iloc[-1]); st.write(f"**Daily composite:** {daily:.2f}")
                hourly = float(cH["Composite"].iloc[-1]); st.write(f"**Hourly composite:** {hourly:.2f}")
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
            try: auc = roc_auc_score(y_true, proba)
            except Exception: auc = np.nan

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
                long_conf, short_conf = confidence_from_composite(comp, include_bb, w_ma, w_rsi, w_macd, w_bb)
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
                rows.append({"Ticker":t, "Composite":comp, "Signal":rec,
                             "Long conf":long_conf, "Short conf":short_conf, "ML P(long)":mlp})
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
    st.title("💼 Portfolio — Upload • Advise • Export")

    # Upload
    st.markdown("**Upload CSV** with columns: `ticker,shares,cost_basis`")
    upl = st.file_uploader("Upload positions CSV", type=["csv"], key="upl_csv")
    txt = st.text_area("...or paste positions:", "AAPL,10,150\nMSFT,5,300", height=120, key="ta_positions")

    def parse_positions(upl_file, text_area):
        rows=[]
        if upl_file is not None:
            try:
                df = pd.read_csv(upl_file)
                for _, r in df.iterrows():
                    rows.append([str(r[0]), float(r[1]), float(r[2])])
            except Exception:
                st.warning("Uploaded CSV could not be parsed; falling back to text area.")
        if not rows:
            for line in (text_area or "").splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts)==3:
                    try: rows.append([parts[0], float(parts[1]), float(parts[2])])
                    except Exception: pass
        return rows

    rows = parse_positions(upl, txt)

    if st.button("▶️ Analyze Portfolio", key="btn_port_analyze"):
        data=[]
        advices=[]
        for idx, row in enumerate(rows, 1):
            try:
                tkr, shares, cost = row
                tkr = _map_symbol(tkr)
                px_live = load_prices(tkr, "5d", "1d")
                if px_live.empty: 
                    st.warning(f"No price for {tkr}"); 
                    continue
                price = _to_float(px_live["Close"].iloc[-1])
                invested = shares * cost
                value    = shares * price
                pnl      = value - invested
                pnl_pct  = (pnl/invested*100) if invested else np.nan

                # Composite suggestion
                px = load_prices(tkr, period_sel, interval_sel)
                if px.empty:
                    comp_score=0.0; comp_sugg="N/A"; long_conf=short_conf=0
                else:
                    ind = compute_indicators(px, ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=include_bb)
                    if ind.empty:
                        comp_score=0.0; comp_sugg="N/A"; long_conf=short_conf=0
                    else:
                        sig = build_composite(ind, ma_window, rsi_period,
                                              use_weighted=use_weighted, w_ma=w_ma, w_rsi=w_rsi, w_macd=w_macd, w_bb=w_bb,
                                              include_bb=include_bb, threshold=comp_thr, allow_short=allow_short)
                        comp_score = float(sig["Composite"].tail(1).iloc[0]) if "Composite" in sig else 0.0
                        long_conf, short_conf = confidence_from_composite(comp_score, include_bb, w_ma, w_rsi, w_macd, w_bb)
                        comp_sugg = "🟢 BUY" if comp_score>=comp_thr else ("🔴 SELL" if comp_score<=-comp_thr else "🟡 HOLD")

                # Guardrails override
                if pnl_pct > profit_target:     suggestion="🔴 SELL"
                elif pnl_pct < -loss_limit:     suggestion="🟢 BUY"
                else:                           suggestion=comp_sugg

                data.append({
                    "Ticker":tkr,"Shares":shares,"Cost Basis":cost,"Price":price,
                    "Market Value":value,"Invested":invested,"P/L":pnl,"P/L %":pnl_pct,
                    "Composite":comp_score,"Long conf /100":long_conf,"Short conf /100":short_conf,
                    "Suggestion":suggestion
                })

                advices.append(
                    f"{tkr}: MV=${value:,.2f}, P/L={pnl_pct:.1f}%. "
                    f"Signal={comp_sugg} (L:{long_conf}/100, S:{short_conf}/100). "
                    f"Guardrails ⇒ {suggestion}."
                )

            except Exception as e:
                st.warning(f"Row {idx} skipped: {e}")

        if data:
            df_port = pd.DataFrame(data).set_index("Ticker")
            st.dataframe(df_port, use_container_width=True)

            c1,c2,c3 = st.columns(3)
            c1.metric("Total Market Value", f"${df_port['Market Value'].sum():,.2f}")
            c2.metric("Total Invested",     f"${df_port['Invested'].sum():,.2f}")
            c3.metric("Total P/L",          f"${df_port['Market Value'].sum()-df_port['Invested'].sum():,.2f}")

            # CSV download
            st.download_button("⬇️ Download CSV", df_port.to_csv().encode("utf-8"), "portfolio_advice.csv",
                               mime="text/csv", key="dl_port_csv")

            # PDF download (optional)
            report_text = "QuantaraX — Portfolio Advice\n\n" + "\n".join(advices)
            if REPORTLAB_OK:
                buf = io.BytesIO()
                c = canvas.Canvas(buf, pagesize=letter)
                width, height = letter
                y = height - 50
                c.setFont("Helvetica-Bold", 14)
                c.drawString(40, y, "QuantaraX — Portfolio Advice")
                c.setFont("Helvetica", 10)
                y -= 20
                wrap = 90
                for line in advices:
                    # manual wrap
                    words = line.split()
                    cur = ""
                    for w in words:
                        if len(cur + " " + w) > wrap:
                            y -= 14
                            c.drawString(40, y, cur)
                            cur = w
                            if y < 60:
                                c.showPage(); y = height - 50; c.setFont("Helvetica", 10)
                        else:
                            cur = (cur + " " + w).strip()
                    if cur:
                        y -= 14; c.drawString(40, y, cur)
                        if y < 60:
                            c.showPage(); y = height - 50; c.setFont("Helvetica", 10)
                c.showPage(); c.save()
                pdf_bytes = buf.getvalue()
                st.download_button("⬇️ Download PDF", data=pdf_bytes, file_name="portfolio_advice.pdf",
                                   mime="application/pdf", key="dl_port_pdf")
            else:
                st.download_button("⬇️ Download TXT", report_text.encode("utf-8"),
                                   "portfolio_advice.txt", mime="text/plain", key="dl_port_txt")
        else:
            st.error("No valid holdings provided.")


# ───────────────────────────── MACRO (quick snapshot) ─────────────────────────────
with tab_macro:
    st.title("🌐 Macro Snapshot — ETF Proxies & YTD")
    # Simple universe to keep things robust
    tickers = st.text_input("ETF list", "SPY, QQQ, IWM, EFA, EEM, TLT, HYG, DBC, GLD, USO",
                            key="inp_macro_list").upper().replace(" ", "")
    run_macro = st.button("Refresh Macro", key="btn_macro")
    if run_macro:
        names = [t for t in tickers.split(",") if t]
        prices = {}
        for t in names:
            px = load_prices(t, "2y", "1d")
            if not px.empty:
                prices[t] = px["Close"].rename(t)
        if prices:
            df = pd.concat(prices.values(), axis=1).dropna(how="all")
            if df.empty:
                st.info("No common dates for macro set.")
            else:
                # YTD calc: for each series, first obs of current year
                cur_year = df.index[-1].year
                def ytd_one(s: pd.Series):
                    s_this = s[s.index.year == cur_year]
                    if len(s_this) == 0: return np.nan
                    return s_this.iloc[-1]/s_this.iloc[0] - 1.0
                ytd = df.apply(ytd_one, axis=0)
                st.bar_chart(ytd.sort_values(ascending=False))
                st.caption("YTD performance by ETF proxy.")
        else:
            st.info("No macro data available.")


# ───────────────────────────── HELP ─────────────────────────────
with tab_help:
    st.header("How to use QuantaraX (Quick Guide)")
    st.markdown("""
**What it is:** QuantaraX combines a transparent technical **Composite Signal Engine**, optional **ML probabilities**,
and a **risk-aware backtester** to help traders make decisions. It’s built for novices (Beginner mode)
and power users (Pro mode).

### 1) Basic workflow
1. Go to **Engine** → enter a symbol → click **Run Composite Backtest**.  
2. Read the **Long Bias** / **Short Bias** with a **confidence score (0–100)**.  
3. Open **Why This Signal?** to see which indicators triggered.  
4. Review backtest stats: **CAGR, Sharpe, Max Drawdown, Win Rate, Trades, Time in Market**.  
5. (Optional) Use **Multi-Timeframe** to confirm Daily vs Hourly alignment.  
6. (Optional) Add more symbols to **Scanner** to rank opportunities.

### 2) What the composite means
We combine:
- **MA crossover** (trend),  
- **RSI** (overbought/oversold),  
- **MACD signal cross** (momentum),  
- **Bollinger Bands** (mean-reversion, optional).  

Each contributes **−1 / 0 / +1** and can be weighted. The **composite** is the weighted sum.  
We convert it to **confidence** by normalizing to the maximum possible score given your weights.  
**Enter/Exit threshold** (`Composite trigger`) defines when we flip between HOLD vs BUY/SELL.

### 3) Backtesting details
- **Costs**: set per-side costs in **bps**.  
- **Shorting**: toggle to allow negative positions.  
- **Vol targeting**: scales exposure to hit a target annualized vol.  
- **Stops/Targets**: ATR-based exits flatten positions on the next bar.

### 4) ML Lab (optional)
- Trains a RandomForest on price-derived features; reports OOS **accuracy/AUC**.  
- Converts probabilities into trades using your entry/exit thresholds, then backtests.

### 5) Portfolio
- Upload or paste positions (`ticker,shares,cost_basis`).  
- Get **guardrail-aware** advice (profit/loss triggers) blended with **composite signals**.  
- Download **CSV** or **PDF/TXT** explanation for clients/stakeholders.

### 6) Macro
- Quick YTD snapshot of major ETF proxies for context.

**Pro tips**
- Use **Pro** mode (sidebar) to reveal more context while keeping the main UI clean.  
- Try Daily interval for robustness; Hourly for tactical timing.  
- If data providers throttle news/earnings, the app will gracefully fall back or display the last known date.
    """)

# End of file
