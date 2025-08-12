# app.py — QuantaraX Ω (v7)
# ---------------------------------------------------------------------------------
# Baseline deps:
#   streamlit yfinance pandas numpy matplotlib feedparser vaderSentiment
# Optional (auto-fallback if missing):
#   duckdb optuna scipy statsmodels hmmlearn scikit-learn reportlab

import os, io, json, math, time, textwrap, warnings
from typing import Dict, List, Tuple, Callable

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore", category=FutureWarning)

# ───────────────────────────── Optional imports ─────────────────────────────
try:
    import duckdb
    DUCK_OK = True
except Exception:
    DUCK_OK = False

try:
    import optuna
    OPTUNA_OK = True
except Exception:
    OPTUNA_OK = False

try:
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import squareform
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

try:
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import coint
    SM_OK = True
except Exception:
    SM_OK = False

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_OK = True
except Exception:
    HMM_OK = False

try:
    from sklearn.cluster import KMeans
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score
    SK_OK = True
except Exception:
    SK_OK = False

try:
    from reportlab.pdfgen import canvas as pdf_canvas
    RL_OK = True
except Exception:
    RL_OK = False

# ───────────────────────────── Page config ─────────────────────────────
st.set_page_config(page_title="QuantaraX Ω (v7)", layout="wide")
analyzer = SentimentIntensityAnalyzer()
rec_map = {1: "🟢 BUY", 0: "🟡 HOLD", -1: "🔴 SELL"}

# ───────────────────────────── Registries (plugin architecture) ─────────────────────────────
SIGNALS: Dict[str, Callable] = {}
RISKS: Dict[str, Callable] = {}
OPTIMS: Dict[str, Callable] = {}
EXECUTORS: Dict[str, Callable] = {}

def register_signal(name: str):
    def deco(f):
        SIGNALS[name] = f
        return f
    return deco

def register_risk(name: str):
    def deco(f):
        RISKS[name] = f
        return f
    return deco

def register_optim(name: str):
    def deco(f):
        OPTIMS[name] = f
        return f
    return deco

def register_executor(name: str):
    def deco(f):
        EXECUTORS[name] = f
        return f
    return deco

# ───────────────────────────── Helpers ─────────────────────────────
def _map_symbol(sym: str) -> str:
    s = sym.strip().upper()
    if "/" in s:
        base, quote = s.split("/")
        quote = "USD" if quote in ("USDT", "USD") else quote
        return f"{base}-{quote}"
    return s

@st.cache_data(show_spinner=False, ttl=600)
def safe_yf_download(symbol: str, period: str, interval: str, retries: int = 2) -> pd.DataFrame:
    sym = _map_symbol(symbol)
    last_exc = None
    for _ in range(max(1, retries)):
        try:
            df = yf.download(sym, period=period, interval=interval, auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            if not df.empty:
                return df.dropna()
        except Exception as e:
            last_exc = e
            time.sleep(0.8)
    return pd.DataFrame()

def safe_news(symbol: str, max_items: int = 5) -> List[Tuple[str, str, float]]:
    """Return [(title, link, sentiment_score), ...] with yfinance->RSS fallback."""
    sym = _map_symbol(symbol)
    out = []
    # yfinance → try/catch to avoid curl_cffi exceptions
    try:
        tk = yf.Ticker(sym)
        news = getattr(tk, "news", []) or []
        for art in news[:max_items]:
            t_ = art.get("title", ""); l_ = art.get("link", "")
            if not (t_ and l_): continue
            txt = art.get("summary", t_)
            s = analyzer.polarity_scores(txt)["compound"]
            out.append((t_, l_, s))
        if out:
            return out
    except Exception:
        pass
    # RSS fallback
    rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={sym}&region=US&lang=en-US"
    try:
        feed = feedparser.parse(rss_url)
        for e in feed.entries[:max_items]:
            t_ = e.title
            l_ = e.link
            s = analyzer.polarity_scores(t_)["compound"]
            out.append((t_, l_, s))
    except Exception:
        pass
    return out

# Optional DuckDB cache layer (feature-store feel)
DUCK_PATH = "qx_cache.duckdb"
def duckdb_store(df: pd.DataFrame, key: str):
    if not DUCK_OK or df.empty: return
    con = duckdb.connect(DUCK_PATH)
    con.execute("CREATE TABLE IF NOT EXISTS prices (key TEXT, ts TIMESTAMP, o DOUBLE, h DOUBLE, l DOUBLE, c DOUBLE, v DOUBLE)")
    d = df.reset_index().rename(columns={"Date":"ts","Open":"o","High":"h","Low":"l","Close":"c","Volume":"v"})
    d["key"] = key
    con.execute("DELETE FROM prices WHERE key = ?", [key])
    con.execute("INSERT INTO prices SELECT * FROM d")
    con.close()

def duckdb_load(key: str) -> pd.DataFrame:
    if not DUCK_OK: return pd.DataFrame()
    if not os.path.exists(DUCK_PATH): return pd.DataFrame()
    con = duckdb.connect(DUCK_PATH)
    try:
        d = con.execute("SELECT ts as Date, o as Open, h as High, l as Low, c as Close, v as Volume FROM prices WHERE key=?", [key]).df()
        con.close()
        if d.empty: return pd.DataFrame()
        d["Date"] = pd.to_datetime(d["Date"])
        d = d.set_index("Date").sort_index()
        return d
    except Exception:
        con.close()
        return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=600)
def load_prices(symbol: str, period: str, interval: str) -> pd.DataFrame:
    key = f"{_map_symbol(symbol)}|{period}|{interval}"
    # Try DuckDB cache first
    if DUCK_OK:
        d = duckdb_load(key)
        if not d.empty:
            return d
    # Pull from yfinance
    df = safe_yf_download(symbol, period, interval)
    if not df.empty and DUCK_OK:
        duckdb_store(df, key)
    return df

# ───────────────────────────── Indicators ─────────────────────────────
def compute_indicators(df: pd.DataFrame, ma_w: int, rsi_p: int, mf: int, ms: int, sig: int,
                       use_bb: bool = True) -> pd.DataFrame:
    d = df.copy()
    if d.empty or not set(["Open","High","Low","Close"]).issubset(d.columns):
        return pd.DataFrame()

    d[f"MA{ma_w}"] = d["Close"].rolling(ma_w).mean()

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

    # Extra set for Ω
    klen = 14
    ll = d["Low"].rolling(klen).min(); hh = d["High"].rolling(klen).max()
    d["STO_K"] = 100 * (d["Close"] - ll) / (hh - ll)
    d["STO_D"] = d["STO_K"].rolling(3).mean()
    dc_n = 20
    d["DC_U"] = d["High"].rolling(dc_n).max()
    d["DC_L"] = d["Low"].rolling(dc_n).min()

    if use_bb:
        w = 20; k = 2.0
        mid = d["Close"].rolling(w).mean()
        sd  = d["Close"].rolling(w).std(ddof=0)
        d["BB_M"], d["BB_U"], d["BB_L"] = mid, mid + k*sd, mid - k*sd
    return d.dropna()

# ───────────────────────────── Signals (plugins) ─────────────────────────────
@register_signal("composite_v2")
def sig_composite(df: pd.DataFrame, *, ma_w=10, rsi_p=14, use_weighted=True,
                  w_ma=1.0, w_rsi=1.0, w_macd=1.0, w_bb=0.5, include_bb=True,
                  threshold=1.0, allow_short=False) -> pd.DataFrame:
    d = df.copy()
    if d.empty: return d
    n = len(d)
    close = d["Close"].to_numpy()
    ma    = d.get(f"MA{ma_w}", d["Close"]).to_numpy()
    rsi   = d.get(f"RSI{rsi_p}", pd.Series(index=d.index, data=50.0)).to_numpy()
    macd  = d.get("MACD", pd.Series(index=d.index, data=0.0)).to_numpy()
    sigl  = d.get("MACD_Signal", pd.Series(index=d.index, data=0.0)).to_numpy()

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

@register_signal("donchian_breakout")
def sig_dc_breakout(df: pd.DataFrame, *, look=20, allow_short=False) -> pd.DataFrame:
    d = df.copy()
    if d.empty: return d
    hh = d["High"].rolling(look).max().shift(1)
    ll = d["Low"].rolling(look).min().shift(1)
    long = d["Close"] > hh
    short = d["Close"] < ll
    d["Trade"] = 0
    d.loc[long, "Trade"] = 1
    if allow_short:
        d.loc[short, "Trade"] = -1
    d["Composite"] = (d["Trade"]).astype(float)
    return d

# ───────────────────────────── Risk overlays (plugins) ─────────────────────────────
@register_risk("atr_exits")
def risk_atr_exits(d: pd.DataFrame, *, sl_mult=2.0, tp_mult=3.0) -> pd.DataFrame:
    if d.empty or "ATR" not in d or "Trade" not in d: return d
    out = d.copy()
    pos = out["Trade"].shift(1).fillna(0)
    flat = np.zeros(len(out), dtype=int)
    entry = np.nan
    for i in range(len(out)):
        p = pos.iat[i]
        c = out["Close"].iat[i]
        a = out["ATR"].iat[i]
        if p != 0 and np.isnan(entry): entry = c
        if p == 0: entry = np.nan
        if p != 0 and not np.isnan(a):
            if p == 1 and (c <= entry - sl_mult*a or c >= entry + tp_mult*a):
                flat[i] = 1; entry = np.nan
            if p == -1 and (c >= entry + sl_mult*a or c <= entry - tp_mult*a):
                flat[i] = 1; entry = np.nan
    if flat.any():
        out.loc[flat==1, "Trade"] = 0
    return out

@register_risk("vol_target")
def risk_vol_target(d: pd.DataFrame, *, target=0.0, interval="1d") -> pd.Series:
    """Returns a scaling series; multiply returns by scale."""
    if target <= 0: return pd.Series(1.0, index=d.index)
    ann = 252 if interval == "1d" else 252*6
    vol = d["Close"].pct_change().rolling(20).std(ddof=0) * math.sqrt(ann)
    scale = (target / vol).clip(0, 3.0).fillna(0.0)
    return scale

# ───────────────────────────── Backtest & execution sim ─────────────────────────────
def backtest(d: pd.DataFrame, *, allow_short=False, cost_bps=0.0, scale_series: pd.Series = None, interval="1d"):
    x = d.copy()
    if x.empty or "Close" not in x:
        zero = x.copy()
        zero["Return"] = 0.0
        zero["Position"] = 0.0
        zero["StratRet"] = 0.0
        zero["CumBH"] = 1.0
        zero["CumStrat"] = 1.0
        return zero, dict(max_dd=0.0, sharpe=np.nan, win=0.0, trades=0, tim=0.0, cagr=np.nan)

    x["Return"] = x["Close"].pct_change().replace([np.inf,-np.inf], np.nan).fillna(0.0)
    pos = x.get("Trade", 0).shift(1).fillna(0).clip(-1 if allow_short else 0, 1)
    base_ret = np.where(pos>=0, x["Return"], -x["Return"])
    if scale_series is not None:
        base_ret = base_ret * scale_series.reindex(x.index).fillna(0.0).values

    cost = cost_bps/10000.0
    pos_change = pos.diff().fillna(0).abs()
    tcost = -2.0*cost*(pos_change > 0).astype(float)  # open+close per flip
    x["Position"] = pos
    x["StratRet"] = pd.Series(base_ret, index=x.index) + tcost

    bh = (1 + x["Return"]).cumprod()
    stg = (1 + x["StratRet"]).cumprod()
    x["CumBH"], x["CumStrat"] = bh, stg

    # Stats
    ann = 252 if interval == "1d" else 252*6
    dd = stg / stg.cummax() - 1
    max_dd = float(dd.min()*100)
    mean_ann = float(x["StratRet"].mean()*ann)
    vol_ann  = float(x["StratRet"].std(ddof=0)*math.sqrt(ann))
    sharpe = mean_ann/vol_ann if vol_ann>0 else np.nan
    win = float((x["StratRet"]>0).mean()*100)
    trades = int((pos_change>0).sum())
    tim = float((pos!=0).mean()*100)
    n_eff = max(1, x["StratRet"].notna().sum())
    last = float(stg.iloc[-1])
    cagr = (last ** (ann/n_eff) - 1) * 100 if n_eff>0 else np.nan
    return x, dict(max_dd=max_dd, sharpe=sharpe, win=win, trades=trades, tim=tim, cagr=cagr)

# Simple execution simulator (order log)
def simulate_execution(d: pd.DataFrame) -> pd.DataFrame:
    if d.empty or "Position" not in d or "Close" not in d: return pd.DataFrame()
    df = d.copy()
    pos = df["Position"].fillna(0)
    flips = pos.diff().fillna(0)
    orders = []
    for i, val in enumerate(flips):
        if val == 0: continue
        side = "BUY" if val>0 else "SELL"
        px = float(df["Close"].iloc[i])
        atr = float(df.get("ATR", pd.Series(index=df.index, data=np.nan)).iloc[i])
        slip = 0.0001 + (atr/px if not np.isnan(atr) and px>0 else 0.0)*0.02
        fill = px*(1+slip) if side=="BUY" else px*(1-slip)
        orders.append({"ts": df.index[i], "side": side, "change": float(val), "mkt_px": px, "slip": slip, "fill_px": fill})
    return pd.DataFrame(orders)

# ───────────────────────────── HRP/BL Hybrid (fallbacks) ─────────────────────────────
def inv_vol_weights(rets: pd.DataFrame) -> pd.Series:
    vol = rets.std().replace(0, np.nan)
    w = (1/vol)
    w = w / w.sum()
    return w.fillna(0)

def hrp_weights(rets: pd.DataFrame) -> pd.Series:
    if not SCIPY_OK or rets.shape[1] < 2:
        return inv_vol_weights(rets)
    corr = rets.corr().fillna(0)
    dist = ((1 - corr).clip(0, 2)) ** 0.5
    distsq = squareform(dist.values, checks=False)
    Z = linkage(distsq, method='single')
    # Quasi-deneric order from dendrogram
    order = dendrogram(Z, no_plot=True)['leaves']
    cov = rets.cov().values
    cov_ = cov[np.ix_(order, order)]
    w = _hrp_recursive_bisect(cov_)
    idx = rets.columns[order]
    return pd.Series(w, index=idx).reindex(rets.columns).fillna(0)

def _hrp_recursive_bisect(cov):
    w = np.ones(cov.shape[0])
    clusters = [np.arange(cov.shape[0])]
    while len(clusters) > 0:
        cl = clusters.pop(0)
        if len(cl) <= 1:
            continue
        split = int(len(cl)/2)
        c1, c2 = cl[:split], cl[split:]
        v1 = _cluster_variance(cov, c1)
        v2 = _cluster_variance(cov, c2)
        alpha = 1 - v1/(v1+v2)
        w[c1] *= alpha
        w[c2] *= 1-alpha
        clusters.extend([c1, c2])
    return w / w.sum()

def _cluster_variance(cov, cluster_idx):
    sub = cov[np.ix_(cluster_idx, cluster_idx)]
    ivp = 1/np.diag(sub)
    ivp /= ivp.sum()
    return ivp @ sub @ ivp

def bl_tilt(weights: pd.Series, views: pd.Series, strength: float = 0.15) -> pd.Series:
    """Very light BL: tilt weights toward positive 'alpha' (views) with cap."""
    vw = weights.copy()
    if views is None or views.empty:
        return vw
    v = (views - views.mean()).fillna(0.0)
    v = v / (abs(v).sum() + 1e-9)
    tilted = vw + strength * v
    tilted = tilted.clip(lower=0)
    return (tilted / tilted.sum()).fillna(0)

# ───────────────────────────── Optuna WFO (graceful fallback) ─────────────────────────────
def objective_factory(px: pd.DataFrame, base_params: dict, allow_short: bool, interval: str):
    def obj(trial):
        ma_w = trial.suggest_int("ma", 5, 40)
        rsi_p = trial.suggest_int("rsi", 5, 25)
        mf = trial.suggest_int("macd_f", 8, 18)
        ms = trial.suggest_int("macd_s", 20, 40)
        sg = trial.suggest_int("macd_sig", 5, 16)
        th = trial.suggest_float("thr", 0.5, 2.0)
        w_ma = trial.suggest_float("w_ma", 0.0, 2.0)
        w_rsi = trial.suggest_float("w_rsi", 0.0, 2.0)
        w_macd= trial.suggest_float("w_macd", 0.0, 2.0)
        w_bb  = trial.suggest_float("w_bb", 0.0, 1.5)
        cost  = trial.suggest_float("cost", 0.0, 0.0005)
        vt    = trial.suggest_float("vol_t", 0.0, 0.25)

        ind = compute_indicators(px, ma_w, rsi_p, mf, ms, sg, use_bb=True)
        sig = sig_composite(ind, ma_w=ma_w, rsi_p=rsi_p, use_weighted=True, w_ma=w_ma, w_rsi=w_rsi,
                            w_macd=w_macd, w_bb=w_bb, include_bb=True, threshold=th, allow_short=allow_short)
        scale = risk_vol_target(sig, target=vt, interval=interval)
        bt, stats = backtest(sig, allow_short=allow_short, cost_bps=cost*10000, scale_series=scale, interval=interval)
        # Multi-objective (Sharpe ↑, MaxDD ↓, Trades ↓)
        return -stats["max_dd"], stats["sharpe"], -stats["trades"]
    return obj

def run_optuna_wfo(symbol: str, period="2y", interval="1d", n_trials=60, allow_short=False):
    if not OPTUNA_OK:
        return None, "Optuna not installed."
    px = load_prices(symbol, period, interval)
    if px.empty:
        return None, f"No data for {symbol}"
    obj = objective_factory(px, {}, allow_short, interval)
    study = optuna.create_study(directions=["maximize","maximize","minimize"])
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    return study, None

# ───────────────────────────── DSL Runner (JSON) ─────────────────────────────
DEFAULT_DSL = {
  "data": {"symbol":"AAPL","period":"1y","interval":"1d"},
  "indicators": {"ma_window":10,"rsi_period":14,"macd":[12,26,9],"include_bb": True},
  "signal": {"name":"composite_v2",
             "params":{"use_weighted":True,"w_ma":1.0,"w_rsi":1.0,"w_macd":1.0,"w_bb":0.5,"threshold":1.0,"allow_short":False}},
  "risk": {"atr":{"sl":2.0,"tp":3.0},"vol_target":0.0,"cost_bps":5.0},
  "report": {"title":"QuantaraX Strategy Report"}
}

def run_dsl(dsl: dict):
    dcfg = dsl.get("data", {})
    sym = dcfg.get("symbol","AAPL"); period = dcfg.get("period","1y"); interval = dcfg.get("interval","1d")
    px = load_prices(sym, period, interval)
    if px.empty: return None, None, "No data."

    icfg = dsl.get("indicators", {})
    ma_w = int(icfg.get("ma_window",10)); rsi_p = int(icfg.get("rsi_period",14))
    mf, ms, sg = icfg.get("macd",[12,26,9])
    include_bb = bool(icfg.get("include_bb", True))
    ind = compute_indicators(px, ma_w, rsi_p, mf, ms, sg, use_bb=include_bb)
    if ind.empty: return None, None, "Not enough data post indicators."

    scfg = dsl.get("signal", {"name":"composite_v2","params":{}})
    sname = scfg.get("name","composite_v2")
    spar  = scfg.get("params",{})
    sig_fn = SIGNALS.get(sname)
    if not sig_fn: return None, None, f"Unknown signal '{sname}'"
    sig = sig_fn(ind, ma_w=ma_w, rsi_p=rsi_p, **spar)

    rcfg = dsl.get("risk", {})
    cost_bps = float(rcfg.get("cost_bps", 0.0))
    vt = float(rcfg.get("vol_target", 0.0))
    scale = risk_vol_target(sig, target=vt, interval=interval)
    if "atr" in rcfg:
        at = rcfg["atr"]
        sig = risk_atr_exits(sig, sl_mult=float(at.get("sl",2.0)), tp_mult=float(at.get("tp",3.0)))

    bt, stats = backtest(sig, allow_short=bool(spar.get("allow_short",False)), cost_bps=cost_bps, scale_series=scale, interval=interval)
    return bt, stats, None

# ───────────────────────────── UI ─────────────────────────────
TABS = [
    "🚀 Engine",
    "🧩 DSL Runner",
    "🧠 ML Lab",
    "📡 Scanner",
    "📉 Regimes",
    "🔗 Pairs Lab",
    "💼 Portfolio",
    "📄 Reports",
    "❓ Help"
]
(tab_engine, tab_dsl, tab_ml, tab_scan, tab_regime, tab_pairs, tab_port, tab_reports, tab_help) = st.tabs(TABS)

# ─────────── Sidebar (unique keys!) ───────────
st.sidebar.header("Global Controls")
DEFAULTS = dict(ma_window=10, rsi_period=14, macd_fast=12, macd_slow=26, macd_signal=9)
if st.sidebar.button("🔄 Reset to defaults", key="btn_reset_defaults"):
    for k, v in DEFAULTS.items():
        st.session_state[k] = v

ma_window   = st.sidebar.slider("MA window", 5, 60, st.session_state.get("ma_window",10), key="sl_ma")
rsi_period  = st.sidebar.slider("RSI lookback", 5, 30, st.session_state.get("rsi_period",14), key="sl_rsi")
macd_fast   = st.sidebar.slider("MACD fast span", 5, 20, st.session_state.get("macd_fast",12), key="sl_mf")
macd_slow   = st.sidebar.slider("MACD slow span", 20, 50, st.session_state.get("macd_slow",26), key="sl_ms")
macd_signal = st.sidebar.slider("MACD sig span", 5, 20, st.session_state.get("macd_signal",9), key="sl_sig")

use_weighted = st.sidebar.toggle("Use weighted composite", value=True, key="tg_weight")
include_bb   = st.sidebar.toggle("Include Bollinger Bands", value=True, key="tg_bb")
w_ma   = st.sidebar.slider("Weight • MA",   0.0, 2.0, 1.0, 0.1, key="w_ma")
w_rsi  = st.sidebar.slider("Weight • RSI",  0.0, 2.0, 1.0, 0.1, key="w_rsi")
w_macd = st.sidebar.slider("Weight • MACD", 0.0, 2.0, 1.0, 0.1, key="w_macd")
w_bb   = st.sidebar.slider("Weight • BB",   0.0, 2.0, 0.5, 0.1, key="w_bb") if include_bb else 0.0
comp_thr = st.sidebar.slider("Composite trigger", 0.0, 3.0, 1.0, 0.1, key="thr")

allow_short = st.sidebar.toggle("Allow shorts", value=False, key="tg_short")
cost_bps    = st.sidebar.slider("Trading cost (bps/side)", 0.0, 25.0, 5.0, 0.5, key="cost")
sl_atr_mult = st.sidebar.slider("Stop • ATR ×", 0.0, 5.0, 2.0, 0.1, key="slatr")
tp_atr_mult = st.sidebar.slider("Target • ATR ×", 0.0, 8.0, 3.0, 0.1, key="tpatr")
vol_target  = st.sidebar.slider("Vol targeting (annual)", 0.0, 0.5, 0.0, 0.05, key="vt")

period_sel   = st.sidebar.selectbox("History", ["6mo","1y","2y","5y"], index=1, key="sb_period")
interval_sel = st.sidebar.selectbox("Interval", ["1d","1h"], index=0, key="sb_interval")

profit_target = st.sidebar.slider("Profit target (%)", 1, 100, 10, key="ptgt")
loss_limit    = st.sidebar.slider("Loss limit (%)",  1, 100, 5, key="llim")

# ───────────────────────────── ENGINE ─────────────────────────────
with tab_engine:
    st.title("🚀 QuantaraX — Engine (v7)")

    colA, colB = st.columns([2,1])
    with colA:
        ticker = st.text_input("Symbol", "AAPL", key="eng_sym").upper()
    with colB:
        run_btn = st.button("▶️ Run Composite Backtest", key="eng_run")

    # Price / news header
    if ticker:
        h = safe_yf_download(ticker, period="1d", interval="1d")
        if not h.empty and "Close" in h:
            last_px = float(h["Close"].iloc[-1])
            st.subheader(f"💲 Live Price: ${last_px:.2f}")
        news = safe_news(ticker, 5)
        if news:
            st.markdown("#### 📰 Recent News & Sentiment")
            for t_, l_, sc in news:
                emoji = "🔺" if sc>0.1 else ("🔻" if sc<-0.1 else "➖")
                st.markdown(f"- [{t_}]({l_}) {emoji}")
        else:
            st.info("No recent news found.")

    if run_btn:
        px = load_prices(ticker, period_sel, interval_sel)
        if px.empty:
            st.error(f"No data for '{ticker}'"); st.stop()
        ind = compute_indicators(px, ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=include_bb)
        sig = sig_composite(ind, ma_w=ma_window, rsi_p=rsi_period, use_weighted=use_weighted,
                            w_ma=w_ma, w_rsi=w_rsi, w_macd=w_macd, w_bb=w_bb, include_bb=include_bb,
                            threshold=comp_thr, allow_short=allow_short)
        # Apply risk overlays
        if sl_atr_mult>0 or tp_atr_mult>0:
            sig = risk_atr_exits(sig, sl_mult=sl_atr_mult, tp_mult=tp_atr_mult)
        scale = risk_vol_target(sig, target=vol_target, interval=interval_sel)
        bt, stats = backtest(sig, allow_short=allow_short, cost_bps=cost_bps, scale_series=scale, interval=interval_sel)

        last_trade = int(sig["Trade"].iloc[-1]) if not sig.empty else 0
        rec = rec_map.get(np.sign(last_trade), "🟡 HOLD")
        st.success(f"**{ticker}**: {rec}")

        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.metric("CAGR", f"{stats['cagr']:.2f}%")
        c2.metric("Sharpe", f"{0 if np.isnan(stats['sharpe']) else stats['sharpe']:.2f}")
        c3.metric("Max DD", f"{stats['max_dd']:.2f}%")
        c4.metric("Win Rate", f"{stats['win']:.1f}%")
        c5.metric("Trades", f"{stats['trades']}")
        c6.metric("Time in Mkt", f"{stats['tim']:.1f}%")

        fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(11,12), sharex=True)
        ax1.plot(bt.index, bt["Close"], label="Close")
        ma_col = f"MA{ma_window}"
        if ma_col in bt: ax1.plot(bt.index, bt[ma_col], label=ma_col)
        if include_bb and {"BB_U","BB_L"}.issubset(bt.columns):
            ax1.plot(bt.index, bt["BB_U"], label="BB_U"); ax1.plot(bt.index, bt["BB_L"], label="BB_L")
        ax1.legend(); ax1.set_title("Price & Indicators")
        ax2.bar(bt.index, sig["Composite"] if "Composite" in sig else 0); ax2.set_title("Composite")
        ax3.plot(bt.index, bt["CumBH"], ":", label="BH")
        ax3.plot(bt.index, bt["CumStrat"], label="Strat"); ax3.legend(); ax3.set_title("Equity")
        plt.xticks(rotation=45); plt.tight_layout()
        st.pyplot(fig)

        # Execution log
        od = simulate_execution(bt)
        if not od.empty:
            st.markdown("#### Execution log (simulated)")
            st.dataframe(od, use_container_width=True)

# ───────────────────────────── DSL Runner ─────────────────────────────
with tab_dsl:
    st.title("🧩 Strategy DSL (JSON)")
    st.caption("Describe your whole pipeline in JSON. We parse → run → report. Keep it simple; YAML not required.")
    dsl_text = st.text_area("DSL JSON", json.dumps(DEFAULT_DSL, indent=2), height=260, key="dsl_text")
    run_dsl_btn = st.button("▶️ Run DSL", key="dsl_run")
    if run_dsl_btn:
        try:
            spec = json.loads(dsl_text)
        except Exception as e:
            st.error(f"JSON parse error: {e}")
            spec = None
        if spec:
            bt, stats, err = run_dsl(spec)
            if err:
                st.error(err)
            else:
                st.success("DSL executed.")
                if stats:
                    c1,c2,c3,c4 = st.columns(4)
                    c1.metric("CAGR", f"{stats['cagr']:.2f}%")
                    c2.metric("Sharpe", f"{0 if np.isnan(stats['sharpe']) else stats['sharpe']:.2f}")
                    c3.metric("Max DD", f"{stats['max_dd']:.2f}%")
                    c4.metric("Win Rate", f"{stats['win']:.1f}%")
                fig, ax = plt.subplots(figsize=(10,4))
                ax.plot(bt.index, bt["CumBH"], ":", label="BH"); ax.plot(bt.index, bt["CumStrat"], label="Strategy")
                ax.legend(); ax.set_title("DSL Equity")
                st.pyplot(fig)

# ───────────────────────────── ML Lab (OOS) ─────────────────────────────
with tab_ml:
    st.title("🧠 ML Lab — RandomForest (OOS)")
    if not SK_OK:
        st.warning("scikit-learn not installed. `pip install scikit-learn` to enable.")
    ml_sym = st.text_input("Symbol (ML)", "AAPL", key="ml_sym").upper()
    horizon = st.slider("Prediction horizon (bars)", 1, 5, 1, key="ml_h")
    train_frac = st.slider("Train fraction", 0.5, 0.95, 0.8, key="ml_tf")
    proba_enter = st.slider("Enter if P(long) ≥", 0.50, 0.80, 0.55, 0.01, key="ml_pe")
    proba_exit  = st.slider("Enter short if P(long) ≤", 0.20, 0.50, 0.45, 0.01, key="ml_px")
    ml_btn = st.button("🤖 Train & Backtest", key="ml_btn")

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

    if ml_btn and SK_OK:
        px = load_prices(ml_sym, period_sel, interval_sel)
        ind = compute_indicators(px, ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=True)
        X = _ml_features(ind)
        y = (ind["Close"].pct_change(horizon).shift(-horizon) > 0).reindex(X.index).astype(int)
        data = pd.concat([X, y.rename("y")], axis=1).dropna()
        if len(data) < 200:
            st.warning("Not enough rows for ML. Try longer history or daily interval.")
        else:
            split = int(len(data)*float(train_frac))
            train, test = data.iloc[:split], data.iloc[split:]
            clf = RandomForestClassifier(n_estimators=400, max_depth=6, random_state=42, n_jobs=-1)
            clf.fit(train.drop(columns=["y"]), train["y"])
            proba = clf.predict_proba(test.drop(columns=["y"]))[:,1]
            y_true = test["y"].values
            acc = accuracy_score(y_true, (proba>0.5).astype(int))
            auc = roc_auc_score(y_true, proba) if len(np.unique(y_true))>1 else np.nan
            c1,c2 = st.columns(2)
            c1.metric("Accuracy (0.5)", f"{acc*100:.1f}%")
            c2.metric("ROC-AUC", f"{0 if np.isnan(auc) else auc:.3f}")
            sig = np.where(proba>=proba_enter, 1, (np.where(proba<=proba_exit, -1, 0) if allow_short else 0))
            ml_df = ind.loc[test.index].copy()
            ml_df["Trade"] = sig.astype(int)
            bt, stats = backtest(ml_df, allow_short=allow_short, cost_bps=cost_bps, interval=interval_sel)
            st.markdown(f"**ML OOS:** Return={(bt['CumStrat'].iloc[-1]-1)*100:.2f}% | Sharpe={0 if np.isnan(stats['sharpe']) else stats['sharpe']:.2f} | MaxDD={stats['max_dd']:.2f}%")
            fig, ax = plt.subplots(figsize=(9,3))
            ax.plot(bt.index, bt["CumBH"], ":", label="BH"); ax.plot(bt.index, bt["CumStrat"], label="ML Strat"); ax.legend(); ax.set_title("ML OOS Equity")
            st.pyplot(fig)

# ───────────────────────────── Scanner ─────────────────────────────
with tab_scan:
    st.title("📡 Universe Scanner")
    universe = st.text_area("Tickers (comma-separated)", "AAPL, MSFT, NVDA, TSLA, AMZN, GOOGL, META, NFLX, SPY, QQQ", height=80, key="sc_uni").upper()
    scan_btn = st.button("🔎 Scan", key="sc_btn")
    if scan_btn:
        rows=[]
        for t in [x.strip() for x in universe.split(",") if x.strip()]:
            try:
                px = load_prices(t, period_sel, interval_sel)
                if px.empty: continue
                ind = compute_indicators(px, ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=True)
                sig = sig_composite(ind, ma_w=ma_window, rsi_p=rsi_period, use_weighted=use_weighted,
                                    w_ma=w_ma, w_rsi=w_rsi, w_macd=w_macd, w_bb=w_bb, include_bb=True,
                                    threshold=comp_thr, allow_short=allow_short)
                comp = float(sig.get("Composite", pd.Series(index=sig.index, data=0.0)).iloc[-1]) if not sig.empty else 0.0
                rec = rec_map.get(int(np.sign(comp)), "🟡 HOLD")
                rows.append({"Ticker":t,"Composite":comp,"Signal":rec})
            except Exception:
                continue
        if rows:
            df = pd.DataFrame(rows).set_index("Ticker").sort_values(["Signal","Composite"], ascending=[True,False])
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No results. Check tickers or increase history.")

# ───────────────────────────── Regimes ─────────────────────────────
with tab_regime:
    st.title("📉 Regime Detection")
    sym = st.text_input("Symbol (Regime)", "SPY", key="rg_sym").upper()
    rg_btn = st.button("Cluster Regimes", key="rg_btn")
    if rg_btn:
        px = load_prices(sym, "2y", "1d")
        ind = compute_indicators(px, ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=False)
        if ind.empty:
            st.error("Not enough data.")
        else:
            feat = pd.DataFrame(index=ind.index)
            feat["vol20"] = ind["Close"].pct_change().rolling(20).std()
            feat["mom20"] = ind["Close"].pct_change(20)
            feat["ma_slope"] = ind[f"MA{ma_window}"].diff()
            feat = feat.dropna()
            if HMM_OK:
                X = feat.fillna(0).values
                hmm = GaussianHMM(n_components=3, covariance_type="diag", n_iter=100, random_state=0)
                hmm.fit(X)
                lab = hmm.predict(X)
            elif SK_OK:
                km = KMeans(n_clusters=3, n_init=10, random_state=42)
                lab = km.fit_predict(feat.fillna(0).values)
            else:
                q = feat.rank(pct=True).mean(axis=1)
                lab = (q>0.66).astype(int) + 2*(q<0.33).astype(int)
            reg = pd.Series(lab, index=feat.index, name="Regime")
            joined = ind.join(reg, how="right")
            # Sort regime labels by avg return
            ret = joined["Close"].pct_change().groupby(joined["Regime"]).mean().sort_values()
            remap = {old:i for i, old in enumerate(ret.index)}
            joined["Regime"] = joined["Regime"].map(remap)
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(joined.index, joined["Close"], label="Close")
            for r in sorted(joined["Regime"].dropna().unique()):
                seg = joined[joined["Regime"]==r]
                ax.fill_between(seg.index, seg["Close"].min(), seg["Close"].max(), alpha=0.08)
            ax.set_title("Price with Regime Shading")
            st.pyplot(fig)

# ───────────────────────────── Pairs Lab ─────────────────────────────
with tab_pairs:
    st.title("🔗 Pairs Lab (Cointegration)")
    pair_a = st.text_input("Symbol A", "MSFT", key="pair_a").upper()
    pair_b = st.text_input("Symbol B", "AAPL", key="pair_b").upper()
    pairs_btn = st.button("Analyze Pair", key="pair_btn")
    if pairs_btn:
        A = load_prices(pair_a, "1y", "1d")["Close"].rename("A")
        B = load_prices(pair_b, "1y", "1d")["Close"].rename("B")
        df = pd.concat([A,B], axis=1).dropna()
        if df.empty:
            st.error("Not enough overlapping data.")
        else:
            beta = np.polyfit(df["B"], df["A"], 1)[0]
            spread = df["A"] - beta*df["B"]
            z = (spread - spread.rolling(60).mean()) / spread.rolling(60).std(ddof=0)
            c_adf = None
            if SM_OK:
                score, pval, _ = coint(df["A"], df["B"])
                c_adf = pval
            st.line_chart(z.rename("Spread Z-Score"))
            st.write(f"β (A~B): {beta:.3f} | CI p-value: {('%.3f'%c_adf) if c_adf is not None else 'n/a'}")
            st.info("Rule of thumb: ±2σ bands for entries; flat when mean reverts to 0.")

# ───────────────────────────── Portfolio ─────────────────────────────
with tab_port:
    st.title("💼 Portfolio — HRP/BL + Simulator")
    opt_tickers = st.text_input("Tickers (comma-sep)", "AAPL, MSFT, TSLA, SPY, QQQ", key="pt_tix").upper()
    pt_btn = st.button("🧮 Optimize", key="pt_btn")
    if pt_btn:
        tickers = [t.strip() for t in opt_tickers.split(",") if t.strip()]
        rets=[]; valid=[]
        for t in tickers:
            px = load_prices(t, "1y", "1d")
            if px.empty: continue
            valid.append(t); rets.append(px["Close"].pct_change().dropna())
        if not rets:
            st.error("No valid data.")
        else:
            R = pd.concat(rets, axis=1); R.columns = valid
            base = hrp_weights(R)  # fallback to INVVOL if SCIPY missing
            # Tiny BL tilt using latest composite signal
            views = pd.Series(0.0, index=valid)
            for t in valid:
                ind = compute_indicators(load_prices(t, "6mo", "1d"), ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=True)
                if ind.empty: continue
                sig = sig_composite(ind, ma_w=ma_window, rsi_p=rsi_period, use_weighted=True,
                                    w_ma=1.0, w_rsi=1.0, w_macd=1.0, w_bb=0.5, include_bb=True, threshold=1.0, allow_short=False)
                views[t] = float(sig["Composite"].iloc[-1]) if not sig.empty else 0.0
            w = bl_tilt(base, views, strength=0.15)
            st.subheader("Weights")
            st.dataframe(w.to_frame("Weight").T, use_container_width=True)
            fig, ax = plt.subplots(figsize=(5,5)); w.plot.pie(autopct="%.1f%%", ax=ax); ax.set_ylabel("")
            st.pyplot(fig)

    st.subheader("📊 Portfolio Simulator (positions CSV)")
    holdings = st.text_area("e.g.\nAAPL,10,150\nMSFT,5,300", height=100, key="pt_pos")
    sim_btn = st.button("▶️ Simulate Portfolio", key="pt_sim")
    if sim_btn:
        rows = [r.strip().split(",") for r in holdings.splitlines() if r.strip()]
        data=[]
        for idx, row in enumerate(rows, 1):
            if len(row) != 3:
                st.warning(f"Skipping invalid row {idx}: {row}"); continue
            ticker_, shares, cost = row
            tkr = _map_symbol(ticker_.upper().strip())
            try: s=float(shares); c=float(cost)
            except: st.warning(f"Invalid numbers on row {idx}: {row}"); continue
            hist = safe_yf_download(tkr, "1d", "1d")
            if hist.empty:
                st.warning(f"No price for {tkr}"); continue
            price=float(hist["Close"].iloc[-1])
            invested=s*c; value=s*price; pnl=value-invested
            pnl_pct=(pnl/invested*100) if invested else np.nan
            # Composite suggestion
            px = load_prices(tkr, period_sel, interval_sel)
            comp_sugg="N/A"
            if not px.empty:
                ind = compute_indicators(px, ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=include_bb)
                sig = sig_composite(ind, ma_w=ma_window, rsi_p=rsi_period, use_weighted=use_weighted,
                                    w_ma=w_ma, w_rsi=w_rsi, w_macd=w_macd, w_bb=w_bb, include_bb=include_bb,
                                    threshold=comp_thr, allow_short=allow_short)
                if not sig.empty:
                    score = float(sig["Composite"].iloc[-1]) if "Composite" in sig else 0.0
                    comp_sugg = "🟢 BUY" if score>=comp_thr else ("🔴 SELL" if score<=-comp_thr else "🟡 HOLD")
            if pnl_pct > profit_target:     suggestion="🔴 SELL"
            elif pnl_pct < -loss_limit:     suggestion="🟢 BUY"
            else:                           suggestion=comp_sugg
            data.append({"Ticker":tkr,"Shares":s,"Cost Basis":c,"Price":price,"Market Value":value,"Invested":invested,
                         "P/L":pnl,"P/L %":pnl_pct,"Composite Sig":comp_sugg,"Suggestion":suggestion})
        if data:
            dfp = pd.DataFrame(data).set_index("Ticker")
            st.dataframe(dfp, use_container_width=True)
            c1,c2,c3 = st.columns(3)
            c1.metric("Total MV", f"${dfp['Market Value'].sum():,.2f}")
            c2.metric("Total Invested", f"${dfp['Invested'].sum():,.2f}")
            c3.metric("Total P/L", f"${(dfp['Market Value'].sum()-dfp['Invested'].sum()):,.2f}")
            fig, ax = plt.subplots(figsize=(5,5)); dfp["Market Value"].plot.pie(autopct="%.1f%%", ax=ax); ax.set_ylabel("")
            st.pyplot(fig)
        else:
            st.error("No valid holdings provided.")

# ───────────────────────────── Reports ─────────────────────────────
with tab_reports:
    st.title("📄 Auto-Reports")
    rep_sym = st.text_input("Symbol (Report)", "AAPL", key="rp_sym").upper()
    rep_btn = st.button("Generate quick HTML report", key="rp_btn")
    if rep_btn:
        px = load_prices(rep_sym, "1y", "1d")
        ind = compute_indicators(px, 10, 14, 12, 26, 9, use_bb=True)
        sig = sig_composite(ind, ma_w=10, rsi_p=14, use_weighted=True, w_ma=1, w_rsi=1, w_macd=1, w_bb=0.5, include_bb=True, threshold=1.0, allow_short=False)
        bt, stats = backtest(sig, allow_short=False, cost_bps=5.0, interval="1d")
        html = f"""
        <html><body>
        <h2>QuantaraX Report — {rep_sym}</h2>
        <p>Sharpe: {0 if np.isnan(stats['sharpe']) else round(stats['sharpe'],2)} | MaxDD: {round(stats['max_dd'],2)}% | CAGR: {round(stats['cagr'],2)}%</p>
        <p>Bars: {len(bt)}</p>
        </body></html>
        """
        st.download_button("Download HTML", data=html, file_name=f"qx_report_{rep_sym}.html", mime="text/html")
        if RL_OK:
            # Tiny PDF header (optional)
            fn = f"qx_report_{rep_sym}.pdf"
            buf = io.BytesIO()
            c = pdf_canvas.Canvas(buf)
            c.setFont("Helvetica-Bold", 14)
            c.drawString(72, 760, f"QuantaraX Report — {rep_sym}")
            c.setFont("Helvetica", 12)
            c.drawString(72, 740, f"Sharpe: {0 if np.isnan(stats['sharpe']) else round(stats['sharpe'],2)}")
            c.drawString(72, 725, f"MaxDD: {round(stats['max_dd'],2)}%  |  CAGR: {round(stats['cagr'],2)}%")
            c.save()
            st.download_button("Download PDF", data=buf.getvalue(), file_name=fn, mime="application/pdf")
        else:
            st.info("Install reportlab for PDF export (`pip install reportlab`).")

# ───────────────────────────── Help ─────────────────────────────
with tab_help:
    st.header("How QuantaraX Ω (v7) Works")
    st.markdown("""
- **Plugin Architecture** — Add signals/risks/optimizers with `@register_*` and call by name.
- **Cache** — DuckDB-backed (optional) + Streamlit cache, less API thrash.
- **Signals** — Composite v2, Donchian breakout; extend easily.
- **Risk** — ATR exits, volatility targeting; execution slip sim + order log.
- **DSL** — Describe the whole pipeline in JSON and run it.
- **Optimizer** — Optuna multi-objective (Sharpe↑, Drawdown↓, Trades↓) with WFO hook (see code).
- **Regimes** — HMM/KMeans/quantile fallback; shaded plot.
- **Pairs** — Cointegration stats + z-score bands.
- **Portfolio** — HRP (fallback INVVOL) + Black-Litterman-style tilt from live composite views.
- **Reports** — One-click HTML (and PDF if reportlab).
- **Resilience** — Safe yfinance wrappers, RSS fallback for news, unique widget keys to avoid duplicate-ID errors.
""")
