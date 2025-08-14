# app.py — QuantaraX Decision Engine (v26, investor-ready+)
# ---------------------------------------------------------------------------------
# pip install:
#   streamlit yfinance pandas numpy matplotlib feedparser vaderSentiment scikit-learn reportlab
# (scikit-learn + reportlab are optional; features degrade gracefully if not present)

import io
import math
import time
import warnings
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.simplefilter("ignore", FutureWarning)


# ───────────────────────────── Optional ML / Report imports ─────────────────────────────
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.inspection import permutation_importance
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

try:
    import reportlab  # noqa: F401
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False


# ───────────────────────────── Page Setup ─────────────────────────────
st.set_page_config(page_title="QuantaraX — Decision Engine (v26)", layout="wide")
analyzer = SentimentIntensityAnalyzer()

rec_map = {1: "🟢 BUY", 0: "🟡 HOLD", -1: "🔴 SELL"}

TAB_TITLES = ["🚀 Engine", "🧠 ML Lab", "📡 Scanner", "📉 Regimes", "💼 Portfolio", "❓ Help"]
(tab_engine, tab_ml, tab_scan, tab_regime, tab_port, tab_help) = st.tabs(TAB_TITLES)


# ───────────────────────────── Sidebar (unique keys) ─────────────────────────────
st.sidebar.header("Global Controls")

MODE = st.sidebar.radio("Mode", ["Beginner", "Pro"], horizontal=True, key="mode")

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
period_sel   = st.sidebar.selectbox("History", ["6mo", "1y", "2y", "5y"], index=1, key="period_sel")
interval_sel = st.sidebar.selectbox("Interval", ["1d", "1h"], index=0, key="interval_sel")

st.sidebar.subheader("Portfolio Guardrails")
profit_target = st.sidebar.slider("Profit target (%)", 1, 100, 10, key="profit_target")
loss_limit    = st.sidebar.slider("Loss limit (%)",  1, 100, 5,  key="loss_limit")


# ───────────────────────────── Helpers ─────────────────────────────
def _map_symbol(sym: str) -> str:
    s = (sym or "").strip().upper()
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
            return float(getattr(x, "iloc", [x])[-1])
        except Exception:
            return float("nan")


def _now_naive() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").tz_convert("UTC").tz_localize(None)


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


def earnings_calendar(symbol: str, lookback=12, lookahead=12) -> pd.DataFrame:
    try:
        cal = yf.Ticker(_map_symbol(symbol)).get_earnings_dates(limit=max(lookback, lookahead))
        if isinstance(cal, pd.DataFrame) and not cal.empty:
            df = cal.copy()
            if isinstance(df.index, (pd.DatetimeIndex, pd.Index)):
                df = df.reset_index()
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
            df = df.rename(columns={date_col: "event_date"})
            df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce").dt.tz_localize(None)
            df = df.dropna(subset=["event_date"]).sort_values("event_date")
            return df
    except Exception:
        pass
    return pd.DataFrame(columns=["event_date"])


def next_earnings_text(symbol: str) -> str:
    cal = earnings_calendar(symbol)
    if cal.empty:
        return "Earnings: unavailable"
    now_d = _now_naive().date()
    future = cal[cal["event_date"].dt.date > now_d]
    if not future.empty:
        nxt = future.iloc[0]["event_date"].date()
        return f"Next earnings: {nxt}"
    last = cal.iloc[-1]["event_date"].date()
    return f"Last earnings: {last} (no upcoming date found)"


def data_health(df: pd.DataFrame, interval: str) -> dict:
    meta = {"fresh": False, "age_hours": None, "label": ""}
    if df.empty:
        meta["label"] = "No data"; return meta
    last_ts = pd.to_datetime(df.index[-1]).tz_localize(None)
    now = _now_naive()
    fresh_hours = max(0.0, (now - last_ts).total_seconds() / 3600.0)
    meta["age_hours"] = fresh_hours
    if interval == "1d":
        meta["fresh"] = fresh_hours <= 24 + 6
    else:
        meta["fresh"] = fresh_hours <= 2.5
    meta["label"] = "Fresh" if meta["fresh"] else f"{fresh_hours:.1f}h ago"
    return meta


# ─────────── Indicators / Composite ───────────
def compute_indicators(df: pd.DataFrame, ma_w: int, rsi_p: int, mf: int, ms: int, sig: int,
                       use_bb: bool = True) -> pd.DataFrame:
    d = df.copy()
    if d.empty or not set(["Open", "High", "Low", "Close"]).issubset(d.columns):
        return pd.DataFrame()

    d[f"MA{ma_w}"] = d["Close"].rolling(ma_w, min_periods=ma_w).mean()

    chg = d["Close"].diff()
    up, dn = chg.clip(lower=0), -chg.clip(upper=0)
    ema_up   = up.ewm(com=rsi_p - 1, adjust=False).mean()
    ema_down = dn.ewm(com=rsi_p - 1, adjust=False).mean()
    rs = ema_up / ema_down.replace(0, np.nan)
    d[f"RSI{rsi_p}"] = 100 - 100 / (1 + rs)

    ema_f = d["Close"].ewm(span=mf, adjust=False).mean()
    ema_s = d["Close"].ewm(span=ms, adjust=False).mean()
    macd_line = ema_f - ema_s
    d["MACD"] = macd_line
    d["MACD_Signal"] = macd_line.ewm(span=sig, adjust=False).mean()

    pc = d["Close"].shift(1)
    tr = pd.concat([(d["High"] - d["Low"]).abs(), (d["High"] - pc).abs(), (d["Low"] - pc).abs()], axis=1).max(axis=1)
    d["ATR"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    if use_bb:
        w = 20; k = 2.0
        mid = d["Close"].rolling(w, min_periods=w).mean()
        sd = d["Close"].rolling(w, min_periods=w).std(ddof=0)
        d["BB_M"], d["BB_U"], d["BB_L"] = mid, mid + k * sd, mid - k * sd

    klen = 14
    ll = d["Low"].rolling(klen, min_periods=klen).min()
    hh = d["High"].rolling(klen, min_periods=klen).max()
    rng = (hh - ll).replace(0, np.nan)
    d["STO_K"] = 100 * (d["Close"] - ll) / rng
    d["STO_D"] = d["STO_K"].rolling(3, min_periods=3).mean()

    adx_n = 14
    up_move = d["High"].diff()
    dn_move = -d["Low"].diff()
    plus_dm = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
    tr_sm = tr.ewm(alpha=1 / adx_n, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=d.index).ewm(alpha=1 / adx_n, adjust=False).mean() / tr_sm
    minus_di = 100 * pd.Series(minus_dm, index=d.index).ewm(alpha=1 / adx_n, adjust=False).mean() / tr_sm
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan) * 100
    d["ADX"] = dx.ewm(alpha=1 / adx_n, adjust=False).mean()

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
    ma = d[f"MA{ma_w}"].to_numpy()
    rsi = d[f"RSI{rsi_p}"].to_numpy()
    macd = d["MACD"].to_numpy()
    sigl = d["MACD_Signal"].to_numpy()

    ma_sig = np.zeros(n, int)
    rsi_sig = np.zeros(n, int)
    macd_sig2 = np.zeros(n, int)
    bb_sig = np.zeros(n, int)

    for i in range(1, n):
        if close[i - 1] < ma[i - 1] and close[i] > ma[i]:   ma_sig[i] = 1
        elif close[i - 1] > ma[i - 1] and close[i] < ma[i]: ma_sig[i] = -1
        if rsi[i] < 30:   rsi_sig[i] = 1
        elif rsi[i] > 70: rsi_sig[i] = -1
        if macd[i - 1] < sigl[i - 1] and macd[i] > sigl[i]:   macd_sig2[i] = 1
        elif macd[i - 1] > sigl[i - 1] and macd[i] < sigl[i]: macd_sig2[i] = -1
        if include_bb and {"BB_U", "BB_L"}.issubset(d.columns):
            if close[i] < d["BB_L"].iloc[i]: bb_sig[i] = 1
            elif close[i] > d["BB_U"].iloc[i]: bb_sig[i] = -1

    comp = (w_ma * ma_sig + w_rsi * rsi_sig + w_macd * macd_sig2 + (w_bb * bb_sig if include_bb else 0)) if use_weighted \
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


def backtest(df: pd.DataFrame, *, allow_short=False, cost_bps=0.0,
             sl_atr_mult=0.0, tp_atr_mult=0.0, vol_target=0.0, interval="1d"):
    d = df.copy()
    if d.empty or "Close" not in d:
        sk = d.copy()
        for col in ["Return", "Position", "StratRet", "CumBH", "CumStrat"]:
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
        ann = 252 if interval == "1d" else 252 * 6
        realized = daily_vol * math.sqrt(ann)
        scale = (vol_target / realized).clip(0, 3.0).fillna(0.0)
        base_ret = base_ret * scale

    cost = cost_bps / 10000.0
    pos_change = d["Position"].diff().fillna(0).abs()
    tcost = -2.0 * cost * (pos_change > 0).astype(float)
    d["StratRet"] = pd.Series(base_ret, index=d.index).fillna(0.0) + tcost

    if (sl_atr_mult > 0 or tp_atr_mult > 0) and "ATR" in d.columns:
        flat = np.zeros(len(d), dtype=int)
        entry = np.nan
        for i in range(len(d)):
            p, c = d["Position"].iat[i], d["Close"].iat[i]
            a = d["ATR"].iat[i]
            if p != 0 and np.isnan(entry): entry = c
            if p == 0: entry = np.nan
            if p != 0 and not np.isnan(a):
                if p == 1 and (c <= entry - sl_atr_mult * a or c >= entry + tp_atr_mult * a):
                    flat[i] = 1; entry = np.nan
                if p == -1 and (c >= entry + sl_atr_mult * a or c <= entry - tp_atr_mult * a):
                    flat[i] = 1; entry = np.nan
        if flat.any(): d.loc[flat == 1, "Position"] = 0

    ret_bh = d["Return"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ret_st = d["StratRet"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    d["CumBH"] = (1 + ret_bh).cumprod()
    d["CumStrat"] = (1 + ret_st).cumprod()

    max_dd, sharpe, win_rt, trades, tim, cagr, _ = _stats_from_equity(d, interval)
    return d, max_dd, sharpe, win_rt, trades, tim, cagr


def confidence_from_score(score: float, thr: float) -> float:
    base = abs(score) / max(thr, 1e-6)
    conf = 100.0 * (0.5 + 0.5 * np.tanh(base - 1.0))
    return float(np.clip(conf, 0, 100))


# ─────────── Factor data & exposures ───────────
FACTOR_ETFS: Dict[str, str] = {
    "SPY": "Market",
    "QQQ": "Growth/Tech",
    "IWM": "Small Size",
    "MTUM": "Momentum",
    "QUAL": "Quality",
    "VLUE": "Value",
    "SIZE": "Size",
    "USMV": "Min Vol",
    "UUP": "US Dollar",
    "TLT": "Duration",
    "GLD": "Gold"
}

@st.cache_data(show_spinner=False, ttl=3600)
def load_factors(period="2y", interval="1d") -> pd.DataFrame:
    syms = list(FACTOR_ETFS.keys())
    df = yf.download(syms, period=period, interval=interval, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df = df["Close"]
    df = df.dropna(how="all").fillna(method="ffill")
    return df

def _ols_beta(y: pd.Series, X: pd.DataFrame):
    X_ = pd.concat([pd.Series(1.0, index=X.index, name="const"), X], axis=1).dropna()
    y_ = y.reindex(X_.index).dropna()
    X_, y_ = X_.reindex(y_.index), y_
    if len(X_) < X_.shape[1] + 10:
        return None, None
    try:
        b = np.linalg.lstsq(X_.values, y_.values, rcond=None)[0]
        coef = pd.Series(b[1:], index=X.columns)
        alpha = float(b[0])
        return alpha, coef
    except Exception:
        return None, None

def factor_exposures_for_symbol(sym: str, period="2y", interval="1d"):
    px = load_prices(sym, period, interval)
    fac = load_factors(period, interval)
    if px.empty or fac.empty or "Close" not in px:
        return None, None
    r_sym = px["Close"].pct_change().dropna()
    r_fac = fac.pct_change().dropna()
    common = r_fac.join(r_sym.to_frame("asset"), how="inner").dropna()
    alpha, coef = _ols_beta(common["asset"], common.drop(columns=["asset"]))
    return alpha, coef


# ───────────────────────────── ENGINE ─────────────────────────────
with tab_engine:
    st.title("QuantaraX — Decision Engine (v26)")

    ticker = st.text_input("Symbol (e.g., AAPL or BTC/USDT)", "AAPL", key="inp_engine_ticker").upper()

    # Live price & freshness
    px_live = load_prices(ticker, "5d", "1d")
    if not px_live.empty and "Close" in px_live:
        last_px = _to_float(px_live["Close"].iloc[-1])
        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader(f"💲 Last Close: ${last_px:.2f}")
        with c2:
            meta = data_health(px_live, "1d")
            if meta["fresh"]:
                st.success(f"✅ Fresh • {meta['label']}")
            else:
                st.warning(f"⏱️ {meta['label']}")

    # Earnings
    st.info(f"📅 {next_earnings_text(ticker)}")

    # News (safe → RSS fallback)
    st.markdown("### 📰 Recent News & Sentiment")
    news = safe_get_news(ticker)
    shown = 0
    if news:
        lim = 3 if MODE == "Beginner" else 8
        for art in news:
            t_ = art.get("title", ""); l_ = art.get("link", "")
            if not (t_ and l_): continue
            txt = art.get("summary", t_)
            score = analyzer.polarity_scores(txt)["compound"]
            emoji = "🔺" if score > 0.1 else ("🔻" if score < -0.1 else "➖")
            st.markdown(f"- [{t_}]({l_}) {emoji}")
            shown += 1
            if shown >= lim: break
    if shown == 0:
        rss = rss_news(ticker, limit=(3 if MODE == "Beginner" else 8))
        if rss:
            for r in rss: st.markdown(f"- [{r['title']}]({r['link']})")
        else:
            st.info("No recent news found.")

    # Composite Backtest
    st.markdown("---")
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

        last_trade = int(df_sig["Trade"].tail(1).iloc[0]) if "Trade" in df_sig.columns and not df_sig.empty else 0
        last_comp  = float(df_sig["Composite"].tail(1).iloc[0]) if "Composite" in df_sig.columns else 0.0
        rec = rec_map.get(1 if last_trade > 0 else (-1 if last_trade < 0 else 0), "🟡 HOLD")

        # Bias + explicit confidence /100
        conf_long  = confidence_from_score(max(0,  last_comp), comp_thr)
        conf_short = confidence_from_score(max(0, -last_comp), comp_thr)

        lcol, rcol = st.columns(2)
        with lcol:
            st.info(f"**Long Bias:** {'🟢 BUY' if last_comp >= comp_thr else '🟡 WAIT'} • score={last_comp:+.2f}")
            st.metric("Long Confidence (0–100)", f"{conf_long:.0f}/100")
            st.progress(int(round(conf_long))/100)
        with rcol:
            st.warning(f"**Short Bias:** {'🔴 SHORT' if last_comp <= -comp_thr else '🟡 WAIT'} • score={last_comp:+.2f}")
            st.metric("Short Confidence (0–100)", f"{conf_short:.0f}/100")
            st.progress(int(round(conf_short))/100, text="")

        st.success(f"**Recommendation:** {rec}")

        # Reasoning
        last = df_sig.tail(1).iloc[0]
        ma_s  = int(last.get("MA_Signal", 0))
        rsi_s = int(last.get("RSI_Signal", 0))
        macd_s= int(last.get("MACD_Signal2", 0))
        rsi_v = float(last.get(f"RSI{rsi_period}", np.nan))
        ma_txt  = {1: f"Price ↑ crossed above MA{ma_window}.", 0: "No MA crossover.",
                   -1: f"Price ↓ crossed below MA{ma_window}."}.get(ma_s, "No MA crossover.")
        rsi_txt = "RSI data unavailable." if np.isnan(rsi_v) else {
            1: f"RSI ({rsi_v:.1f}) < 30 → oversold.",
            0: f"RSI ({rsi_v:.1f}) neutral.",
            -1: f"RSI ({rsi_v:.1f}) > 70 → overbought."
        }.get(rsi_s, f"RSI ({rsi_v:.1f}) neutral.")
        macd_txt= {1: "MACD ↑ crossed above signal.", 0: "No MACD crossover.",
                   -1: "MACD ↓ crossed below signal."}.get(macd_s, "No MACD crossover.")
        with st.expander("🔎 Why This Signal?"):
            st.write(f"- **MA:**  {ma_txt}")
            st.write(f"- **RSI:** {rsi_txt}")
            st.write(f"- **MACD:** {macd_txt}")
            if include_bb and "BB_Signal" in df_sig.columns:
                bb_s = int(last.get("BB_Signal", 0))
                bb_txt = {1: "Close under lower band (mean-revert long).", 0: "Inside bands.",
                          -1: "Close over upper band (mean-revert short)."}[bb_s]
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

        st.markdown(f"- **Buy & Hold:** {(bh_last - 1) * 100:.2f}%  \n- **Strategy:** {(strat_last - 1) * 100:.2f}%")

        # Charts: Price+signals, Composite, Equity, Drawdown; mark entries/exits
        idx = df_c.index
        fig, axes = plt.subplots(4, 1, figsize=(11, 15), sharex=True)
        ax1, ax2, ax3, ax4 = axes

        ax1.plot(idx, df_c["Close"], label="Close")
        if f"MA{ma_window}" in df_c: ax1.plot(idx, df_c[f"MA{ma_window}"], label=f"MA{ma_window}")
        if include_bb and {"BB_U", "BB_L"}.issubset(df_c.columns):
            ax1.plot(idx, df_c["BB_U"], label="BB Upper"); ax1.plot(idx, df_c["BB_L"], label="BB Lower")

        # entry/exit markers
        chg = df_c["Position"].diff().fillna(0)
        buys = df_c.index[chg > 0]
        sells = df_c.index[chg < 0]
        ax1.scatter(buys, df_c.loc[buys, "Close"], marker="^", s=50, label="Entry", zorder=3)
        ax1.scatter(sells, df_c.loc[sells, "Close"], marker="v", s=50, label="Exit",  zorder=3)
        ax1.legend(); ax1.set_title("Price, Indicators & Trades")

        ax2.bar(idx, df_c["Composite"])
        ax2.axhline(comp_thr, color="gray", linestyle="--", linewidth=0.8)
        ax2.axhline(-comp_thr, color="gray", linestyle="--", linewidth=0.8)
        ax2.set_title("Composite (weighted)")

        ax3.plot(idx, df_c["CumBH"], ":", label="BH")
        ax3.plot(idx, df_c["CumStrat"], "-", label="Strat"); ax3.legend(); ax3.set_title("Equity")

        dd = (df_c["CumStrat"] / df_c["CumStrat"].cummax() - 1.0).fillna(0.0)
        ax4.fill_between(idx, dd, 0, color="tab:red", alpha=0.3)
        ax4.set_title("Drawdown (Strategy)")

        plt.xticks(rotation=45); plt.tight_layout()
        st.pyplot(fig)

        # Factor exposures
        with st.expander("📈 Factor Exposures (beta)"):
            try:
                alpha, coef = factor_exposures_for_symbol(ticker, period="2y", interval="1d")
                if coef is not None and len(coef) > 0:
                    fac_table = pd.DataFrame({
                        "Factor ETF": coef.index,
                        "Beta": coef.values,
                        "Interpretation": [FACTOR_ETFS.get(x, "") for x in coef.index]
                    }).set_index("Factor ETF").sort_values("Beta", ascending=False)
                    st.dataframe(fac_table)
                    st.caption("Method: OLS on daily returns vs factor ETFs; alpha is intercept (omitted here).")
                else:
                    st.info("Not enough data to compute factor exposures.")
            except Exception as e:
                st.info(f"Factor exposure unavailable: {e}")

        # Position sizing helper
        with st.expander("🎯 Position Sizing"):
            try:
                risk_pct = st.slider("Risk per trade (% of portfolio)", 0.1, 5.0, 1.0, 0.1, key="ps_risk_pct")
                acct_val = st.number_input("Portfolio value ($)", min_value=1000.0, value=100000.0, step=1000.0, key="ps_port_val")
                atr = float(df_raw["ATR"].iloc[-1]) if "ATR" in df_raw else np.nan
                price = float(df_raw["Close"].iloc[-1])
                stop_mult = st.slider("ATR Stop ×", 0.5, 5.0, max(1.0, sl_atr_mult or 1.0), 0.5, key="ps_atr_mult")
                if not np.isnan(atr) and price > 0:
                    stop_dist = atr * stop_mult
                    dollar_risk = acct_val * (risk_pct / 100.0)
                    shares = (dollar_risk / stop_dist) if stop_dist > 0 else 0
                    notional = shares * price
                    st.write(f"- ATR ≈ {atr:.2f}, price ≈ {price:.2f}, stop distance ≈ {stop_dist:.2f}")
                    st.success(f"Suggested size: **{shares:.0f} shares** (~${notional:,.0f}) for {risk_pct:.1f}% risk")
                else:
                    st.info("Need ATR/price to compute sizing.")
            except Exception as e:
                st.info(f"Sizing unavailable: {e}")

        # Export PDF summary
        def build_pdf_bytes():
            try:
                from reportlab.lib.pagesizes import letter
                from reportlab.lib.styles import getSampleStyleSheet
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
                from reportlab.lib import colors

                buf = io.BytesIO()
                doc = SimpleDocTemplate(buf, pagesize=letter)
                styles = getSampleStyleSheet()
                flow = []

                flow.append(Paragraph(f"QuantaraX Report — {ticker}", styles["Title"]))
                flow.append(Paragraph(next_earnings_text(ticker), styles["Normal"]))
                flow.append(Spacer(1, 10))

                flow.append(Paragraph(f"Recommendation: <b>{rec}</b>", styles["Heading2"]))
                flow.append(Paragraph(
                    f"Composite score: {last_comp:+.2f} (thr={comp_thr:.1f}). "
                    f"Long confidence: {conf_long:.0f}/100 | Short confidence: {conf_short:.0f}/100.",
                    styles["Normal"]
                ))
                flow.append(Spacer(1, 8))

                flow.append(Paragraph("Why this signal?", styles["Heading3"]))
                flow.append(Paragraph(f"• MA — {ma_txt}", styles["Normal"]))
                flow.append(Paragraph(f"• RSI — {rsi_txt}", styles["Normal"]))
                flow.append(Paragraph(f"• MACD — {macd_txt}", styles["Normal"]))
                flow.append(Spacer(1, 8))

                tbl = Table([
                    ["CAGR", "Sharpe", "Max DD", "Win Rate", "Trades", "Time in Mkt"],
                    [f"{(0 if np.isnan(cagr) else cagr):.2f}%",
                     f"{(0 if np.isnan(sharpe) else sharpe):.2f}",
                     f"{max_dd:.2f}%",
                     f"{win_rt:.1f}%",
                     f"{trades}",
                     f"{tim:.1f}%"]
                ])
                tbl.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ]))
                flow.append(tbl)
                flow.append(Spacer(1, 8))
                flow.append(Paragraph("Disclaimer: Educational only; not investment advice.", styles["Normal"]))

                doc.build(flow)
                buf.seek(0)
                return buf.getvalue()
            except Exception:
                txt = io.StringIO()
                txt.write(f"QuantaraX Report — {ticker}\n")
                txt.write(next_earnings_text(ticker) + "\n\n")
                txt.write(f"Recommendation: {rec}\n")
                txt.write(f"Composite: {last_comp:+.2f} (thr={comp_thr:.1f})\n")
                txt.write(f"Long conf: {conf_long:.0f}/100 | Short conf: {conf_short:.0f}/100\n")
                txt.write(f"Sharpe={sharpe:.2f} | MaxDD={max_dd:.2f}% | WinRate={win_rt:.1f}% | Trades={trades} | TimeInMkt={tim:.1f}%\n")
                return txt.getvalue().encode()

        pdf_bytes = build_pdf_bytes()
        st.download_button(
            "⬇️ Download PDF summary" if REPORTLAB_OK else "⬇️ Download TXT summary",
            data=pdf_bytes,
            file_name=(f"{ticker}_QuantaraX_Report.pdf" if REPORTLAB_OK else f"{ticker}_QuantaraX_Report.txt"),
            mime=("application/pdf" if REPORTLAB_OK else "text/plain"),
            key="dl_single_pdf"
        )

    # Extra — Multi-Timeframe
    st.markdown("---")
    with st.expander("⏱️ Multi-Timeframe Confirmation", expanded=False):
        mtf_symbol = st.text_input("Symbol (MTF)", value=ticker or "AAPL", key="inp_mtf_symbol")
        if st.button("🔍 Check MTF", key="btn_mtf"):
            try:
                d1 = compute_indicators(load_prices(mtf_symbol, "1y", "1d"), ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=True)
                dH = compute_indicators(load_prices(mtf_symbol, "30d", "1h"), ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=True)
                if d1.empty or dH.empty: st.warning("Insufficient data for MTF.")
                else:
                    c1 = build_composite(d1, ma_window, rsi_period, use_weighted=True, w_ma=1.0, w_rsi=1.0, w_macd=1.0, w_bb=0.5, include_bb=True, threshold=1.0)
                    cH = build_composite(dH, ma_window, rsi_period, use_weighted=True, w_ma=1.0, w_rsi=1.0, w_macd=1.0, w_bb=0.5, include_bb=True, threshold=1.0)
                    daily, hourly = float(c1["Composite"].iloc[-1]), float(cH["Composite"].iloc[-1])
                    st.write(f"**Daily composite:** {daily:+.2f}")
                    st.write(f"**Hourly composite:** {hourly:+.2f}")
                    if np.sign(daily) == np.sign(hourly): st.success("✅ Signals agree")
                    else: st.warning("⚠️ Signals disagree")
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
        if {"BB_U", "BB_L"}.issubset(d.columns):
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
            proba = clf.predict_proba(test.drop(columns=["y"]))[:, 1]
            y_true = test["y"].values
            acc = accuracy_score(y_true, (proba > 0.5).astype(int))
            try: auc = roc_auc_score(y_true, proba)
            except Exception: auc = np.nan

            st.subheader("Out-of-sample performance")
            c1, c2 = st.columns(2)
            c1.metric("Accuracy (0.5)", f"{acc * 100:.1f}%")
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
            bt, md, sh, wr, trd, tim, cagr = backtest(
                ml_df, allow_short=allow_short, cost_bps=cost_bps,
                sl_atr_mult=sl_atr_mult, tp_atr_mult=tp_atr_mult,
                vol_target=vol_target, interval=interval_sel
            )
            st.markdown(
                f"**ML Strategy OOS:** Return={(bt['CumStrat'].iloc[-1] - 1) * 100:.2f}% | Sharpe={sh:.2f} | MaxDD={md:.2f}% | Trades={trd}"
            )
            fig, ax = plt.subplots(figsize=(9, 3))
            ax.plot(bt.index, bt["CumBH"], ":", label="BH")
            ax.plot(bt.index, bt["CumStrat"], label="ML Strat")
            ax.legend(); ax.set_title("ML OOS Equity")
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
                if px.empty: continue
                ind = compute_indicators(px, ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=True)
                sig = build_composite(
                    ind, ma_window, rsi_period,
                    use_weighted=use_weighted, w_ma=w_ma, w_rsi=w_rsi, w_macd=w_macd, w_bb=w_bb,
                    include_bb=include_bb, threshold=comp_thr, allow_short=allow_short
                )
                if sig.empty: continue
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
            if ind.empty: st.error("Not enough data.")
            else:
                feat = pd.DataFrame(index=ind.index)
                feat["vol20"] = ind["Close"].pct_change().rolling(20).std()
                feat["mom20"] = ind["Close"].pct_change(20)
                feat["ma_slope"] = ind[f"MA{ma_window}"].diff()
                feat = feat.dropna()

                if SKLEARN_OK and len(feat) >= 60:
                    from sklearn.cluster import KMeans
                    km = KMeans(n_clusters=3, n_init=10, random_state=42)
                    lab = km.fit_predict(feat)
                else:
                    q = feat.rank(pct=True).mean(axis=1)
                    lab = (q > 0.66).astype(int) + (q < 0.33).astype(int) * 2

                reg = pd.Series(lab, index=feat.index, name="Regime")
                joined = ind.join(reg, how="right")
                ret = joined["Close"].pct_change().groupby(joined["Regime"]).mean().sort_values()
                ord_map = {old: i for i, old in enumerate(ret.index)}
                joined["Regime"] = joined["Regime"].map(ord_map)

                st.dataframe(joined[["Close", "Regime"]].tail(10))
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(joined.index, joined["Close"], label="Close")
                for r in sorted([x for x in joined["Regime"].dropna().unique() if pd.notna(x)]):
                    seg = joined[joined["Regime"] == r]
                    ax.fill_between(seg.index, seg["Close"].min(), seg["Close"].max(), alpha=0.08)
                ax.set_title("Price with Regime Shading")
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Regime error: {e}")


# ───────────────────────────── PORTFOLIO (Advisor + Factors + PDF) ─────────────────────────────
with tab_port:
    st.title("💼 Portfolio — Advisor, Factors, Optimizer & Monte Carlo")

    # Advisor
    st.subheader("🧭 Advisor")
    st.caption("Paste holdings as CSV: ticker,shares,cost_basis — or upload a CSV with those columns.")
    up = st.file_uploader("Upload CSV (optional)", type=["csv"], key="upl_holdings")
    default_csv = "AAPL,10,150\nMSFT,5,300\nSPY,8,430"
    text_default = st.text_area("or paste here", default_csv, height=120, key="ta_portfolio")

    def parse_holdings():
        if up is not None:
            try:
                dfu = pd.read_csv(up)
                dfu.columns = [c.strip().lower() for c in dfu.columns]
                if not {"ticker", "shares", "cost_basis"}.issubset(dfu.columns):
                    st.error("CSV must include columns: ticker, shares, cost_basis")
                    return []
                return dfu[["ticker", "shares", "cost_basis"]].values.tolist()
            except Exception as e:
                st.error(f"CSV parse error: {e}")
                return []
        rows = []
        for i, r in enumerate(text_default.splitlines(), 1):
            if not r.strip(): continue
            parts = [x.strip() for x in r.split(",")]
            if len(parts) != 3:
                st.warning(f"Skipping invalid row {i}: {r}"); continue
            rows.append(parts)
        return rows

    if st.button("▶️ Analyze Portfolio", key="btn_port_analyze"):
        rows = parse_holdings()
        data = []
        for idx, row in enumerate(rows, 1):
            try:
                ticker_, shares, cost = row
                tkr = _map_symbol(ticker_.upper().strip())
                s = float(shares); c = float(cost)
            except Exception:
                st.warning(f"Invalid numbers on row {idx}: {row}"); continue

            hist = load_prices(tkr, "5d", "1d")
            if hist.empty:
                st.warning(f"No price for {tkr}"); continue
            price = _to_float(hist["Close"].iloc[-1])
            invested = s * c; value = s * price; pnl = value - invested
            pnl_pct = (pnl / invested * 100) if invested else np.nan

            # Composite suggestion
            px = load_prices(tkr, period_sel, interval_sel)
            comp_sugg = "N/A"; comp_score = 0.0
            if not px.empty:
                df_i = compute_indicators(px, ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=include_bb)
                if not df_i.empty:
                    df_csig = build_composite(
                        df_i, ma_window, rsi_period,
                        use_weighted=use_weighted, w_ma=w_ma, w_rsi=w_rsi, w_macd=w_macd, w_bb=w_bb,
                        include_bb=include_bb, threshold=comp_thr, allow_short=allow_short
                    )
                    if not df_csig.empty:
                        comp_score = float(df_csig["Composite"].tail(1).iloc[0]) if "Composite" in df_csig else 0.0
                        comp_sugg = "🟢 BUY" if comp_score >= comp_thr else ("🔴 SELL" if comp_score <= -comp_thr else "🟡 HOLD")

            # Guardrails override
            if pnl_pct > profit_target:     suggestion = "🔴 SELL (take profit)"
            elif pnl_pct < -loss_limit:     suggestion = "🟢 BUY (average judiciously)"
            else:                           suggestion = comp_sugg

            data.append({
                "Ticker": tkr, "Shares": s, "Cost Basis": c, "Price": price,
                "Market Value": value, "Invested": invested, "P/L": pnl, "P/L %": pnl_pct,
                "Composite": comp_score, "Composite Sig": comp_sugg, "Suggestion": suggestion
            })

        if data:
            df_port = pd.DataFrame(data).set_index("Ticker")
            st.dataframe(df_port, use_container_width=True)

            c1, c2, c3 = st.columns(3)
            total_mv = df_port["Market Value"].sum()
            total_inv = df_port["Invested"].sum()
            c1.metric("Total Market Value", f"${total_mv:,.2f}")
            c2.metric("Total Invested",     f"${total_inv:,.2f}")
            c3.metric("Total P/L",          f"${(total_mv-total_inv):,.2f}")

            # Portfolio factor tilt (value-weighted betas)
            with st.expander("🧬 Portfolio Factor Tilt"):
                try:
                    fac = load_factors("2y", "1d")
                    betas = {}
                    for tkr, row in df_port.iterrows():
                        alpha, coef = factor_exposures_for_symbol(tkr, "2y", "1d")
                        if coef is not None: betas[tkr] = coef
                    if betas:
                        # value weights
                        vw = df_port["Market Value"] / df_port["Market Value"].sum()
                        # aggregate beta per factor
                        agg = {}
                        for tkr, coef in betas.items():
                            for f, b in coef.items():
                                agg[f] = agg.get(f, 0.0) + float(b) * float(vw.get(tkr, 0.0))
                        tilt = pd.Series(agg).sort_values(ascending=False)
                        if not tilt.empty:
                            df_tilt = tilt.rename("Portfolio Beta").to_frame()
                            df_tilt["Factor"] = [FACTOR_ETFS.get(x, "") for x in df_tilt.index]
                            st.dataframe(df_tilt)
                        else:
                            st.info("No factor betas computed.")
                    else:
                        st.info("Unable to compute factor betas for the holdings.")
                except Exception as e:
                    st.info(f"Factor tilt unavailable: {e}")

            # Simple hedge sizing vs SPY
            with st.expander("🛡️ Hedge Suggestion (beta vs SPY)"):
                try:
                    rets = {}
                    for tkr in list(df_port.index) + ["SPY"]:
                        px = load_prices(tkr, "1y", "1d")
                        if px.empty: continue
                        rets[tkr] = px["Close"].pct_change().dropna()
                    if {"SPY"}.issubset(rets.keys()):
                        R = pd.concat(rets, axis=1).dropna()
                        b = {}
                        for tkr in df_port.index:
                            if tkr in R:
                                cov = np.cov(R[tkr], R["SPY"])[0, 1]
                                var = np.var(R["SPY"])
                                b[tkr] = cov / var if var > 0 else np.nan
                        port_beta = np.nansum([b.get(t, np.nan) * (df_port.loc[t, "Market Value"] / total_mv) for t in df_port.index])
                        spy_price = _to_float(load_prices("SPY", "5d", "1d")["Close"].iloc[-1])
                        hedge_shares = -round((port_beta * total_mv) / spy_price, 2) if spy_price > 0 else 0
                        st.write(f"Estimated portfolio beta: **{port_beta:.2f}**")
                        st.write(f"Suggested SPY hedge: **{hedge_shares} shares** (short if negative)")
                    else:
                        st.info("Could not compute hedge (SPY data unavailable).")
                except Exception as e:
                    st.info(f"Hedge calc unavailable: {e}")

            # Scenario shocks
            with st.expander("🌪️ Scenario Shocks"):
                try:
                    shock_spy = st.slider("Shock SPY (%)", -5.0, 5.0, -2.0, 0.5, key="shock_spy")
                    shock_tlt = st.slider("Shock TLT (%)", -5.0, 5.0, 1.0, 0.5, key="shock_tlt")
                    # crude mapping via factor betas (if available)
                    exp_spy = 0.0; exp_tlt = 0.0
                    # estimate exposures as before
                    rets = {}
                    for tkr in list(df_port.index) + ["SPY", "TLT"]:
                        px = load_prices(tkr, "1y", "1d")
                        if px.empty: continue
                        rets[tkr] = px["Close"].pct_change().dropna()
                    if {"SPY", "TLT"}.issubset(rets.keys()):
                        R = pd.concat(rets, axis=1).dropna()
                        betas = {}
                        for tkr in df_port.index:
                            if tkr in R:
                                X = R[["SPY", "TLT"]]
                                a, c = _ols_beta(R[tkr], X)
                                if c is not None: betas[tkr] = c
                        vw = df_port["Market Value"] / df_port["Market Value"].sum()
                        for tkr, c in betas.items():
                            exp_spy += float(c.get("SPY", 0.0)) * float(vw.get(tkr, 0.0))
                            exp_tlt += float(c.get("TLT", 0.0)) * float(vw.get(tkr, 0.0))
                    pnl_pct = exp_spy * shock_spy + exp_tlt * shock_tlt
                    st.write(f"Approx. portfolio P/L for scenario: **{pnl_pct:.2f}%**")
                except Exception as e:
                    st.info(f"Scenario unavailable: {e}")

            # Portfolio PDF download (per-holding reasoning)
            def build_portfolio_pdf_bytes(dfp: pd.DataFrame):
                try:
                    from reportlab.lib.pagesizes import letter
                    from reportlab.lib.styles import getSampleStyleSheet
                    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
                    from reportlab.lib import colors

                    buf = io.BytesIO()
                    doc = SimpleDocTemplate(buf, pagesize=letter)
                    styles = getSampleStyleSheet()
                    flow = []

                    flow.append(Paragraph("QuantaraX — Portfolio Advisor", styles["Title"]))
                    flow.append(Spacer(1, 6))
                    tbl_data = [["Ticker", "Shares", "Cost", "Price", "P/L %", "Composite", "Suggestion"]]
                    for t, row in dfp.iterrows():
                        tbl_data.append([
                            t,
                            f"{row['Shares']:.2f}",
                            f"{row['Cost Basis']:.2f}",
                            f"{row['Price']:.2f}",
                            f"{(0 if np.isnan(row['P/L %']) else row['P/L %']):.2f}%",
                            f"{row['Composite']:+.2f}",
                            row["Suggestion"]
                        ])
                    tbl = Table(tbl_data, repeatRows=1)
                    tbl.setStyle(TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ]))
                    flow.append(tbl)
                    flow.append(Spacer(1, 8))
                    flow.append(Paragraph("Guardrails:", styles["Heading3"]))
                    flow.append(Paragraph(f"• Profit target: {profit_target}% | Loss limit: {loss_limit}% ", styles["Normal"]))
                    flow.append(Paragraph("• Suggestions reflect composite signal and guardrails.", styles["Normal"]))
                    flow.append(Spacer(1, 6))
                    flow.append(Paragraph("Disclaimer: Educational only; not investment advice.", styles["Normal"]))

                    doc.build(flow)
                    buf.seek(0)
                    return buf.getvalue()
                except Exception:
                    return dfp.to_csv(index=True).encode()

            port_pdf = build_portfolio_pdf_bytes(df_port)
            st.download_button(
                "⬇️ Download Portfolio Report" if REPORTLAB_OK else "⬇️ Download Portfolio CSV",
                data=port_pdf,
                file_name=("QuantaraX_Portfolio_Report.pdf" if REPORTLAB_OK else "QuantaraX_Portfolio.csv"),
                mime=("application/pdf" if REPORTLAB_OK else "text/csv"),
                key="dl_port_pdf"
            )
        else:
            st.error("No valid holdings provided.")

    st.markdown("---")
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
            if not rets: st.error("No valid tickers/data.")
            else:
                R = pd.concat(rets, axis=1); R.columns = valid
                cov = R.cov(); n = len(valid); w = np.ones(n) / n
                for _ in range(500):
                    mrc = cov @ w; rc = w * mrc; target = rc.mean(); grad = rc - target
                    w = np.clip(w - 0.05 * grad, 0, None)
                    s = w.sum(); w = w / s if s > 1e-12 else np.ones(n) / n
                    if np.linalg.norm(grad) < 1e-6: break
                weights = pd.Series(w, index=valid, name="Weight")
                st.dataframe(weights.to_frame().T, use_container_width=True)
                fig, ax = plt.subplots(figsize=(5, 5))
                weights.plot.pie(autopct="%.1f%%", ax=ax); ax.set_ylabel(""); ax.set_title("Risk-Parity Weights")
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
                ind, ma_window, rsi_period,
                use_weighted=use_weighted, w_ma=w_ma, w_rsi=w_rsi, w_macd=w_macd, w_bb=w_bb,
                include_bb=include_bb, threshold=comp_thr, allow_short=allow_short
            )
            bt, *_ = backtest(
                sig, allow_short=allow_short, cost_bps=cost_bps,
                sl_atr_mult=sl_atr_mult, tp_atr_mult=tp_atr_mult,
                vol_target=vol_target, interval="1d"
            )
            r = bt["StratRet"].dropna().values
            if len(r) < 50: st.warning("Not enough strategy bars to bootstrap.")
            else:
                N = len(r); endings = []
                for _ in range(int(n_paths)):
                    samp = np.random.choice(r, size=N, replace=True)
                    eq = (1 + pd.Series(samp)).cumprod().iloc[-1]
                    endings.append(eq)
                endings = np.array(endings)
                pct = (np.percentile(endings, [5, 25, 50, 75, 95]) - 1) * 100
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("P5%",  f"{pct[0]:.1f}%"); c2.metric("P25%", f"{pct[1]:.1f}%")
                c3.metric("Median", f"{pct[2]:.1f}%"); c4.metric("P75%", f"{pct[3]:.1f}%"); c5.metric("P95%", f"{pct[4]:.1f}%")
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.hist((endings - 1) * 100, bins=30, alpha=0.85)
                ax.set_title("Monte Carlo: Distribution of End Returns (%)")
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Monte Carlo error: {e}")


# ───────────────────────────── HELP ─────────────────────────────
with tab_help:
    st.header("How to use QuantaraX (v26)")
    st.markdown("""
**What is this?**  
QuantaraX blends classic technicals (MA/RSI/MACD, bands & channels) with a robust backtester, optional ML probabilities, factor analysis, and a portfolio advisor. It’s designed for **both beginners** (simple calls-to-action) and **advanced traders** (deep research tools).

---

### Quick Start (Beginner)
1) **Type a symbol** (e.g., AAPL) and click **Run Composite Backtest**.  
2) Read the **Recommendation** (BUY/SELL/HOLD), and check the **Confidence (0–100)**.  
3) Expand **Why This Signal?** to see which indicators triggered.  
4) Use **Position Sizing** to decide how many shares to buy, using your risk % and ATR-based stops.  
5) Click **Download PDF** for a neat summary you can share.

**What does Confidence mean?**  
It measures how far the composite score is beyond the threshold (smoothed with `tanh`).  
- 0–39: weak → wait for confirmation  
- 40–69: moderate → consider partial size  
- 70–100: strong → consider full size (within risk limits)

---

### Interpreting the Composite (Pro & Beginner)
- **MA**: Crosses indicate trend changes.  
- **RSI**: <30 oversold (bullish), >70 overbought (bearish).  
- **MACD**: Line crossing its signal is momentum change.  
- **Bollinger** (optional): Mean-reversion hint at band extremes.  
Weighted sum → **Composite**. Trades fire when `|Composite| ≥ threshold`.

---

### Backtesting Controls (Pro)
- **Allow shorts**: Enables long/short testing.  
- **Trading costs**: Enter BPS per side (open+close).  
- **ATR exits**: Stop/target in ATR units.  
- **Vol targeting**: Scales daily exposure to a target annualized vol.

Metrics you’ll see:
- **Sharpe** (risk-adjusted return), **MaxDD**, **Win rate**, **Time in market**, **CAGR**.  
- Equity + **Drawdown** chart to visualize pain.

---

### Multi-Timeframe Confirmation
Check daily vs hourly composite agreement:
- **Agree** → higher conviction  
- **Disagree** → reduce size or wait

---

### Factor Exposures
We estimate betas vs ETFs (SPY, QQQ, IWM, MTUM, QUAL, VLUE, SIZE, USMV, UUP, TLT, GLD).  
Use this to understand **macro sensitivity** (equity beta, duration, dollar, gold, momentum, value, etc.).

---

### Portfolio Advisor
Paste or upload `ticker,shares,cost_basis`. You get:
- **Per-holding** P/L, composite score, BUY/SELL/HOLD suggestion  
- **Guardrails** (take-profit / cut-loss) override signals when tripped  
- **Portfolio factor tilt** (value-weighted betas)  
- **Hedge sizing** vs SPY (beta-based shares)  
- **Scenario shocks** (e.g., SPY -2%, TLT +1%) → approximate P/L impact  
- **PDF report** for your portfolio

---

### ML Lab (optional)
RandomForest predicts **P(up in N bars)** and tests a proba-based strategy out-of-sample, with feature importance.

---

### Regimes
Clustering by volatility, momentum, and MA slope helps explain why a system under/over-performs in different cycles.

---

### Tips
- Crypto pairs: type `BTC/USDT` (auto-mapped to `BTC-USD`).  
- Freshness badge shows if data is timely (tolerates weekends/holidays).  
- If a library/data source fails, the app **degrades gracefully** without crashing.

**Disclaimer**: Educational use only. Not investment advice. Position sizing and risk management are your responsibility.
""")
