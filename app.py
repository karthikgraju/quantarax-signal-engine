# app.py — QuantaraX ∞ (v8)
# ---------------------------------------------------------------------------------
# Core deps:
#   streamlit yfinance pandas numpy matplotlib feedparser vaderSentiment
# Optional (auto-fallback if missing):
#   duckdb optuna scipy statsmodels hmmlearn scikit-learn reportlab

import os, io, json, math, time, textwrap, warnings, datetime as dt
from typing import Dict, List, Tuple, Callable

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore", category=FutureWarning)
np.seterr(all="ignore")

# ───────────────────────────── optional imports ─────────────────────────────
def _try_import(mod, names=None):
    try:
        m = __import__(mod, fromlist=names or [])
        return m, True
    except Exception:
        return None, False

duckdb, DUCK_OK = _try_import("duckdb")
optuna, OPTUNA_OK = _try_import("optuna")
scipy, SCIPY_OK = _try_import("scipy")
statsmodels, SM_OK = _try_import("statsmodels")
hmmlearn, HMM_OK = _try_import("hmmlearn")
sk, SK_OK = _try_import("sklearn")
reportlab, RL_OK = _try_import("reportlab")

if SK_OK:
    from sklearn.cluster import KMeans
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score
if SCIPY_OK:
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import squareform
if SM_OK:
    from statsmodels.tsa.stattools import coint
if HMM_OK:
    from hmmlearn.hmm import GaussianHMM
if RL_OK:
    from reportlab.pdfgen import canvas as pdf_canvas

# ───────────────────────────── page config ─────────────────────────────
st.set_page_config(page_title="QuantaraX ∞ (v8)", layout="wide")
analyzer = SentimentIntensityAnalyzer()
rec_map = {1:"🟢 BUY", 0:"🟡 HOLD", -1:"🔴 SELL"}

# ───────────────────────────── plugin registries ─────────────────────────────
SIGNALS: Dict[str, Callable] = {}
SIZERS: Dict[str, Callable] = {}
RISKS: Dict[str, Callable] = {}

def register_signal(name):  return lambda f: (SIGNALS.setdefault(name, f) or f)
def register_sizer(name):   return lambda f: (SIZERS.setdefault(name, f) or f)
def register_risk(name):    return lambda f: (RISKS.setdefault(name, f) or f)

# ───────────────────────────── helpers ─────────────────────────────
def _map_symbol(sym: str) -> str:
    s = sym.strip().upper()
    if "/" in s:
        base, quote = s.split("/")
        quote = "USD" if quote in ("USDT","USD") else quote
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

def safe_news(symbol: str, max_items: int = 5) -> List[Tuple[str,str,float]]:
    sym = _map_symbol(symbol)
    out = []
    try:
        tk = yf.Ticker(sym)
        news = getattr(tk, "news", []) or []
        for art in news[:max_items]:
            t_ = art.get("title",""); l_ = art.get("link","")
            if not (t_ and l_): continue
            txt = art.get("summary", t_)
            s = analyzer.polarity_scores(txt)["compound"]
            out.append((t_, l_, s))
        if out: return out
    except Exception:
        pass
    try:
        rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={sym}&region=US&lang=en-US"
        feed = feedparser.parse(rss_url)
        for e in feed.entries[:max_items]:
            t_ = e.title; l_ = e.link
            s = analyzer.polarity_scores(t_)["compound"]
            out.append((t_, l_, s))
    except Exception:
        pass
    return out

def safe_earnings(symbol: str) -> pd.DataFrame:
    try:
        cal = yf.Ticker(_map_symbol(symbol)).get_earnings_dates(limit=8)
        if isinstance(cal, pd.DataFrame) and not cal.empty:
            return cal.reset_index().rename(columns={"index":"EarningsDate"})
    except Exception:
        pass
    return pd.DataFrame()

# ───────── optional DuckDB cache ─────────
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
    if not DUCK_OK or not os.path.exists(DUCK_PATH): return pd.DataFrame()
    con = duckdb.connect(DUCK_PATH)
    try:
        d = con.execute("SELECT ts as Date, o as Open, h as High, l as Low, c as Close, v as Volume FROM prices WHERE key=?", [key]).df()
        con.close()
        if d.empty: return pd.DataFrame()
        d["Date"] = pd.to_datetime(d["Date"])
        return d.set_index("Date").sort_index()
    except Exception:
        con.close()
        return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=600)
def load_prices(symbol: str, period: str, interval: str) -> pd.DataFrame:
    key = f"{_map_symbol(symbol)}|{period}|{interval}"
    if DUCK_OK:
        d = duckdb_load(key)
        if not d.empty: return d
    df = safe_yf_download(symbol, period, interval)
    if not df.empty and DUCK_OK: duckdb_store(df, key)
    return df

# ───────────────────────────── indicators ─────────────────────────────
def compute_indicators(df: pd.DataFrame, ma_w: int, rsi_p: int, mf: int, ms: int, sig: int,
                       use_bb: bool = True) -> pd.DataFrame:
    d = df.copy()
    if d.empty or not set(["Open","High","Low","Close"]).issubset(d.columns): return pd.DataFrame()
    d[f"MA{ma_w}"] = d["Close"].rolling(ma_w).mean()
    chg = d["Close"].diff()
    up, dn = chg.clip(lower=0), -chg.clip(upper=0)
    ema_up = up.ewm(com=rsi_p-1, adjust=False).mean()
    ema_dn = dn.ewm(com=rsi_p-1, adjust=False).mean()
    rs = ema_up / ema_dn.replace(0, np.nan)
    d[f"RSI{rsi_p}"] = 100 - 100/(1+rs)
    ema_f = d["Close"].ewm(span=mf, adjust=False).mean()
    ema_s = d["Close"].ewm(span=ms, adjust=False).mean()
    d["MACD"] = ema_f - ema_s
    d["MACD_Signal"] = d["MACD"].ewm(span=sig, adjust=False).mean()
    pc = d["Close"].shift(1)
    tr = pd.concat([(d["High"]-d["Low"]).abs(), (d["High"]-pc).abs(), (d["Low"]-pc).abs()], axis=1).max(axis=1)
    d["ATR"] = tr.ewm(alpha=1/14, adjust=False).mean()
    # extras
    klen = 14
    ll = d["Low"].rolling(klen).min(); hh = d["High"].rolling(klen).max()
    d["STO_K"] = 100*(d["Close"]-ll)/(hh-ll)
    d["STO_D"] = d["STO_K"].rolling(3).mean()
    dc_n=20; d["DC_U"]=d["High"].rolling(dc_n).max(); d["DC_L"]=d["Low"].rolling(dc_n).min()
    if use_bb:
        w=20;k=2.0; mid=d["Close"].rolling(w).mean(); sd=d["Close"].rolling(w).std(ddof=0)
        d["BB_M"],d["BB_U"],d["BB_L"]=mid,mid+k*sd,mid-k*sd
    return d.dropna()

# ───────────────────────────── signal plugins ─────────────────────────────
@register_signal("composite_v2")
def sig_composite(df: pd.DataFrame, *, ma_w=10, rsi_p=14, use_weighted=True,
                  w_ma=1.0, w_rsi=1.0, w_macd=1.0, w_bb=0.5, include_bb=True,
                  threshold=1.0, allow_short=False) -> pd.DataFrame:
    d = df.copy()
    if d.empty: return d
    n = len(d)
    close = d["Close"].to_numpy()
    ma = d.get(f"MA{ma_w}", d["Close"]).to_numpy()
    rsi = d.get(f"RSI{rsi_p}", pd.Series(index=d.index, data=50.0)).to_numpy()
    macd = d.get("MACD", pd.Series(index=d.index, data=0.0)).to_numpy()
    sigl = d.get("MACD_Signal", pd.Series(index=d.index, data=0.0)).to_numpy()
    ma_sig = np.zeros(n,int); rsi_sig=np.zeros(n,int); macd_sig2=np.zeros(n,int); bb_sig=np.zeros(n,int)
    for i in range(1,n):
        if close[i-1]<ma[i-1] and close[i]>ma[i]: ma_sig[i]=1
        elif close[i-1]>ma[i-1] and close[i]<ma[i]: ma_sig[i]=-1
        if rsi[i]<30: rsi_sig[i]=1
        elif rsi[i]>70: rsi_sig[i]=-1
        if macd[i-1]<sigl[i-1] and macd[i]>sigl[i]: macd_sig2[i]=1
        elif macd[i-1]>sigl[i-1] and macd[i]<sigl[i]: macd_sig2[i]=-1
        if include_bb and {"BB_U","BB_L"}.issubset(d.columns):
            if close[i] < d["BB_L"].iloc[i]: bb_sig[i]=1
            elif close[i] > d["BB_U"].iloc[i]: bb_sig[i]=-1
    comp = (w_ma*ma_sig + w_rsi*rsi_sig + w_macd*macd_sig2 + (w_bb*bb_sig if include_bb else 0)) if use_weighted else (ma_sig+rsi_sig+macd_sig2)
    trade = np.where(comp>=threshold,1,np.where(comp<=-threshold,-1 if allow_short else 0,0))
    d["MA_Signal"], d["RSI_Signal"], d["MACD_Signal2"] = ma_sig, rsi_sig, macd_sig2
    if include_bb: d["BB_Signal"]=bb_sig
    d["Composite"]=comp.astype(float); d["Trade"]=trade.astype(int)
    return d

@register_signal("donchian_breakout")
def sig_dc_breakout(df: pd.DataFrame, *, look=20, allow_short=False) -> pd.DataFrame:
    d=df.copy(); 
    if d.empty: return d
    hh=d["High"].rolling(look).max().shift(1); ll=d["Low"].rolling(look).min().shift(1)
    d["Trade"]=0; d.loc[d["Close"]>hh,"Trade"]=1
    if allow_short: d.loc[d["Close"]<ll,"Trade"]=-1
    d["Composite"]=d["Trade"].astype(float)
    return d

# ───────────────────────────── sizing plugins ─────────────────────────────
@register_sizer("fixed_1x")
def size_fixed(sig_df: pd.DataFrame, **_) -> pd.Series:
    return pd.Series(1.0, index=sig_df.index)

@register_sizer("kelly_capped")
def size_kelly(sig_df: pd.DataFrame, cap=1.5) -> pd.Series:
    r = sig_df["Close"].pct_change().rolling(40)
    mu = r.mean(); sd = r.std(ddof=0)
    f = (mu/(sd**2)).clip(lower=-cap, upper=cap)
    return f.fillna(0.0).abs()

@register_sizer("composite_scaled")
def size_comp_scaled(sig_df: pd.DataFrame, scale=0.5) -> pd.Series:
    c = sig_df.get("Composite", pd.Series(index=sig_df.index, data=0.0)).abs()
    c = (c / (c.rolling(50).quantile(0.9)+1e-9)).clip(0,2.0)
    return (scale*c).fillna(0.0)

# ───────────────────────────── risk overlays ─────────────────────────────
@register_risk("atr_trailing")
def risk_atr_trailing(d: pd.DataFrame, mult=2.0) -> pd.DataFrame:
    if d.empty or "ATR" not in d or "Trade" not in d: return d
    out=d.copy(); pos=out["Trade"].shift(1).fillna(0)
    trail=np.full(len(out), np.nan)
    for i in range(1,len(out)):
        p=pos.iat[i]; c=out["Close"].iat[i]; a=out["ATR"].iat[i]
        if p>0:
            trail[i]=max(trail[i-1] if not np.isnan(trail[i-1]) else -1e9, c-mult*a)
            if c < trail[i]: out.loc[out.index[i],"Trade"]=0
        elif p<0:
            trail[i]=min(trail[i-1] if not np.isnan(trail[i-1]) else 1e9, c+mult*a)
            if c > trail[i]: out.loc[out.index[i],"Trade"]=0
        else:
            trail[i]=np.nan
    return out

def risk_vol_target(d: pd.DataFrame, *, target=0.0, interval="1d") -> pd.Series:
    if target<=0: return pd.Series(1.0, index=d.index)
    ann=252 if interval=="1d" else 252*6
    vol=d["Close"].pct_change().rolling(20).std(ddof=0)*math.sqrt(ann)
    scale=(target/vol).clip(0,3.0).fillna(0.0)
    return scale

def risk_drawdown_throttle(equity: pd.Series, *, soft=-0.1, hard=-0.2) -> float:
    """Return multiplier (0..1) based on current drawdown."""
    if equity.empty: return 1.0
    dd = equity.iloc[-1]/equity.cummax().iloc[-1] - 1
    if dd <= hard: return 0.0
    if dd <= soft: return max(0.2, 1 - (abs(dd)-abs(soft))/(abs(hard)-abs(soft)))
    return 1.0

# ───────────────────────────── backtest / execution ─────────────────────────────
def backtest(d: pd.DataFrame, *, allow_short=False, cost_bps=0.0,
             scale_series: pd.Series = None, sizer: pd.Series = None, interval="1d"):
    x=d.copy()
    if x.empty or "Close" not in x:
        zero=x.copy()
        for c in ["Return","Position","StratRet","CumBH","CumStrat"]: zero[c]=0.0
        zero["CumBH"]=1.0; zero["CumStrat"]=1.0
        return zero, dict(max_dd=0.0, sharpe=np.nan, win=0.0, trades=0, tim=0.0, cagr=np.nan)

    x["Return"]=x["Close"].pct_change().replace([np.inf,-np.inf], np.nan).fillna(0.0)
    raw = x.get("Trade",0).shift(1).fillna(0).clip(-1 if allow_short else 0, 1)
    # apply sizer
    size = sizer.reindex(x.index).fillna(0.0) if isinstance(sizer, pd.Series) else pd.Series(1.0, index=x.index)
    pos = raw*size
    base_ret = np.where(pos>=0, x["Return"], -x["Return"]) * pos.abs()
    if scale_series is not None:
        base_ret *= scale_series.reindex(x.index).fillna(0.0).values

    cost = cost_bps/10000.0
    flips = pos.diff().fillna(0).abs()
    tcost = -2.0*cost*(flips>0).astype(float)
    x["Position"]=pos
    x["StratRet"]=pd.Series(base_ret, index=x.index)+tcost

    bh=(1+x["Return"]).cumprod(); stg=(1+x["StratRet"]).cumprod()
    x["CumBH"],x["CumStrat"]=bh,stg

    ann=252 if interval=="1d" else 252*6
    dd = stg/stg.cummax()-1
    max_dd=float(dd.min()*100)
    mean_ann=float(x["StratRet"].mean()*ann)
    vol_ann=float(x["StratRet"].std(ddof=0)*math.sqrt(ann))
    sharpe=mean_ann/vol_ann if vol_ann>0 else np.nan
    win=float((x["StratRet"]>0).mean()*100)
    trades=int((flips>0).sum())
    tim=float((pos!=0).mean()*100)
    n_eff=max(1, x["StratRet"].notna().sum())
    last=float(stg.iloc[-1]); cagr=(last**(ann/n_eff)-1)*100 if n_eff>0 else np.nan
    return x, dict(max_dd=max_dd, sharpe=sharpe, win=win, trades=trades, tim=tim, cagr=cagr)

def simulate_execution(d: pd.DataFrame) -> pd.DataFrame:
    if d.empty or "Position" not in d or "Close" not in d: return pd.DataFrame()
    df=d.copy(); flips = df["Position"].fillna(0).diff().fillna(0)
    orders=[]
    for i,chg in enumerate(flips):
        if chg==0: continue
        side = "BUY" if chg>0 else "SELL"
        px=float(df["Close"].iloc[i])
        atr=float(df.get("ATR", pd.Series(index=df.index,data=np.nan)).iloc[i])
        slip=0.0001 + (atr/px if not np.isnan(atr) and px>0 else 0.0)*0.02
        fill=px*(1+slip) if side=="BUY" else px*(1-slip)
        orders.append({"ts":df.index[i], "side":side, "delta_pos":float(chg), "mkt_px":px, "slip":slip, "fill_px":fill})
    return pd.DataFrame(orders)

# ───────────────────────────── HRP/BL utilities ─────────────────────────────
def inv_vol_weights(rets: pd.DataFrame) -> pd.Series:
    vol=rets.std().replace(0,np.nan)
    w=(1/vol); w=w/w.sum()
    return w.fillna(0)

def _hrp_recursive_bisect(cov):
    w=np.ones(cov.shape[0]); clusters=[np.arange(cov.shape[0])]
    def cl_var(cidx):
        sub=cov[np.ix_(cidx,cidx)]
        ivp=1/np.diag(sub); ivp/=ivp.sum()
        return ivp@sub@ivp
    while clusters:
        cl=clusters.pop(0)
        if len(cl)<=1: continue
        split=len(cl)//2; c1,c2=cl[:split],cl[split:]
        v1,v2=cl_var(c1),cl_var(c2)
        a=1 - v1/(v1+v2)
        w[c1]*=a; w[c2]*=(1-a)
        clusters.extend([c1,c2])
    return w/w.sum()

def hrp_weights(rets: pd.DataFrame) -> pd.Series:
    if not SCIPY_OK or rets.shape[1]<2: return inv_vol_weights(rets)
    corr=rets.corr().fillna(0); dist=((1-corr).clip(0,2))**0.5
    distsq=squareform(dist.values, checks=False)
    Z=linkage(distsq, method="single")
    order=dendrogram(Z, no_plot=True)["leaves"]
    cov=rets.cov().values; cov_=cov[np.ix_(order,order)]
    w=_hrp_recursive_bisect(cov_)
    return pd.Series(w, index=rets.columns[order]).reindex(rets.columns).fillna(0)

def bl_tilt(weights: pd.Series, views: pd.Series, strength: float=0.15) -> pd.Series:
    vw=weights.copy()
    if views is None or views.empty: return vw
    v=(views-views.mean()).fillna(0.0); v=v/(abs(v).sum()+1e-9)
    tilted=(vw + strength*v).clip(lower=0)
    return (tilted/tilted.sum()).fillna(0)

# ───────────────────────────── Optuna WFO ─────────────────────────────
def objective_factory(px: pd.DataFrame, allow_short: bool, interval: str):
    def obj(trial):
        ma_w=trial.suggest_int("ma", 5, 40)
        rsi_p=trial.suggest_int("rsi", 5, 25)
        mf=trial.suggest_int("macd_f", 8, 18)
        ms=trial.suggest_int("macd_s", 20, 40)
        sg=trial.suggest_int("macd_sig", 5, 16)
        th=trial.suggest_float("thr", 0.5, 2.0)
        w_ma=trial.suggest_float("w_ma", 0.0, 2.0)
        w_rsi=trial.suggest_float("w_rsi", 0.0, 2.0)
        w_macd=trial.suggest_float("w_macd", 0.0, 2.0)
        w_bb=trial.suggest_float("w_bb", 0.0, 1.5)
        cost=trial.suggest_float("cost", 0.0, 0.0005)
        vt=trial.suggest_float("vol_t", 0.0, 0.25)
        ind=compute_indicators(px, ma_w, rsi_p, mf, ms, sg, use_bb=True)
        sig=sig_composite(ind, ma_w=ma_w, rsi_p=rsi_p, use_weighted=True, w_ma=w_ma, w_rsi=w_rsi,
                          w_macd=w_macd, w_bb=w_bb, include_bb=True, threshold=th, allow_short=allow_short)
        scale=risk_vol_target(sig, target=vt, interval=interval)
        bt, stats=backtest(sig, allow_short=allow_short, cost_bps=cost*10000, scale_series=scale, interval=interval)
        return -stats["max_dd"], stats["sharpe"], -stats["trades"]
    return obj

def run_optuna(symbol: str, period="2y", interval="1d", n_trials=60, allow_short=False):
    if not OPTUNA_OK: return None, "Optuna not installed."
    px=load_prices(symbol, period, interval)
    if px.empty: return None, f"No data for {symbol}."
    study=optuna.create_study(directions=["maximize","maximize","minimize"])
    study.optimize(objective_factory(px, allow_short, interval), n_trials=n_trials, show_progress_bar=False)
    return study, None

# ───────────────────────────── Strategy Router (regime-aware) ─────────────────────────────
def infer_regime(ind: pd.DataFrame) -> int:
    """0=bear/volatile, 1=neutral, 2=bull/trend — simple fallback."""
    vol = ind["Close"].pct_change().rolling(20).std()
    mom = ind["Close"].pct_change(20)
    score = (mom.rank(pct=True) + (1-vol.rank(pct=True)))/2
    last = score.dropna().iloc[-1] if not score.dropna().empty else 0.5
    if last < 0.33: return 0
    if last > 0.66: return 2
    return 1

def route_strategy(ind: pd.DataFrame, allow_short: bool, params: dict) -> pd.DataFrame:
    r = infer_regime(ind)
    if r==2:
        return sig_dc_breakout(ind, look=20, allow_short=allow_short)
    elif r==0:
        p=dict(params); p["threshold"]=max(1.5, p.get("threshold",1.0))
        return sig_composite(ind, **p)
    else:
        return sig_composite(ind, **params)

# ───────────────────────────── UI ─────────────────────────────
TABS = [
    "🚀 Engine", "🧩 DSL", "🧠 ML", "📡 Scanner", "📉 Regimes",
    "🔗 Pairs", "💼 Portfolio", "🎛️ Optimizer", "📄 Reports", "🩺 Health", "❓ Help"
]
(tab_engine, tab_dsl, tab_ml, tab_scan, tab_regime, tab_pairs,
 tab_port, tab_opt, tab_reports, tab_health, tab_help) = st.tabs(TABS)

# ─────────── sidebar (unique keys) ───────────
st.sidebar.header("Global Controls")
DEFAULTS = dict(ma_window=10, rsi_period=14, macd_fast=12, macd_slow=26, macd_signal=9)
if st.sidebar.button("🔄 Reset to defaults", key="sb_reset"):
    for k,v in DEFAULTS.items(): st.session_state[k]=v

ma_window   = st.sidebar.slider("MA window", 5, 60, st.session_state.get("ma_window",10), key="sb_ma")
rsi_period  = st.sidebar.slider("RSI lookback", 5, 30, st.session_state.get("rsi_period",14), key="sb_rsi")
macd_fast   = st.sidebar.slider("MACD fast", 5, 20, st.session_state.get("macd_fast",12), key="sb_mf")
macd_slow   = st.sidebar.slider("MACD slow", 20, 50, st.session_state.get("macd_slow",26), key="sb_ms")
macd_signal = st.sidebar.slider("MACD signal", 5, 20, st.session_state.get("macd_signal",9), key="sb_sig")

use_weighted = st.sidebar.toggle("Weighted composite", True, key="sb_wt")
include_bb   = st.sidebar.toggle("Include Bollinger", True, key="sb_bb")
w_ma   = st.sidebar.slider("W•MA",   0.0, 2.0, 1.0, 0.1, key="sb_w_ma")
w_rsi  = st.sidebar.slider("W•RSI",  0.0, 2.0, 1.0, 0.1, key="sb_w_rsi")
w_macd = st.sidebar.slider("W•MACD", 0.0, 2.0, 1.0, 0.1, key="sb_w_macd")
w_bb   = st.sidebar.slider("W•BB",   0.0, 2.0, 0.5, 0.1, key="sb_w_bb") if include_bb else 0.0
comp_thr = st.sidebar.slider("Composite trigger", 0.0, 3.0, 1.0, 0.1, key="sb_thr")

allow_short = st.sidebar.toggle("Allow shorts", False, key="sb_short")
cost_bps    = st.sidebar.slider("Trading cost (bps/side)", 0.0, 25.0, 5.0, 0.5, key="sb_cost")
sl_atr_mult = st.sidebar.slider("Stop ATR×", 0.0, 5.0, 2.0, 0.1, key="sb_sl")
tp_atr_mult = st.sidebar.slider("Target ATR×", 0.0, 8.0, 3.0, 0.1, key="sb_tp")
vol_target  = st.sidebar.slider("Vol target (ann.)", 0.0, 0.5, 0.0, 0.05, key="sb_vt")

period_sel   = st.sidebar.selectbox("History", ["6mo","1y","2y","5y"], index=1, key="sb_period")
interval_sel = st.sidebar.selectbox("Interval", ["1d","1h"], index=0, key="sb_interval")

profit_target = st.sidebar.slider("Profit target (%)", 1, 100, 10, key="sb_ptgt")
loss_limit    = st.sidebar.slider("Loss limit (%)",  1, 100, 5,  key="sb_llim")

# Profiles
st.sidebar.markdown("---")
st.sidebar.subheader("Profiles")
if st.sidebar.button("💾 Download profile", key="sb_prof_save"):
    prof = {
        "ma_window":ma_window, "rsi_period":rsi_period, "macd_fast":macd_fast, "macd_slow":macd_slow,
        "macd_signal":macd_signal, "use_weighted":use_weighted, "include_bb":include_bb,
        "w": [w_ma, w_rsi, w_macd, float(w_bb) if isinstance(w_bb,float) else 0.0],
        "thr": comp_thr, "allow_short":allow_short, "cost_bps":cost_bps,
        "sl": sl_atr_mult, "tp": tp_atr_mult, "vt": vol_target,
        "period": period_sel, "interval": interval_sel
    }
    st.sidebar.download_button("Download JSON", data=json.dumps(prof, indent=2), file_name="qx_profile.json", key="sb_dl_prof")
upf = st.sidebar.file_uploader("Upload profile JSON", type=["json"], key="sb_prof_up")
if upf:
    try:
        cfg=json.load(upf)
        st.sidebar.success("Profile loaded. Apply by re-setting sliders.")
    except Exception as e:
        st.sidebar.error(f"Bad profile: {e}")

# ───────────────────────────── ENGINE ─────────────────────────────
with tab_engine:
    st.title("🚀 QuantaraX — Engine (v8)")
    cA,cB,cC = st.columns([2,1,1])
    with cA: ticker = st.text_input("Symbol", "AAPL", key="eng_sym").upper()
    with cB: sizer_sel = st.selectbox("Sizer", list(SIZERS.keys()), index=0, key="eng_sizer")
    with cC: router_on = st.toggle("Regime Router", True, key="eng_router")

    r1,r2 = st.columns([3,2])
    with r1:
        run_btn = st.button("▶️ Run Backtest", key="eng_run")
    with r2:
        st.caption("News & Earnings")
        news = safe_news(ticker, 5)
        if news:
            for t_,l_,sc in news:
                emoji = "🔺" if sc>0.1 else ("🔻" if sc<-0.1 else "➖")
                st.markdown(f"- [{t_}]({l_}) {emoji}")
        er = safe_earnings(ticker)
        if not er.empty:
            nxt = er.sort_values("EarningsDate").head(1)
            ed = pd.to_datetime(nxt["EarningsDate"].iloc[0]).date()
            st.info(f"Next Earnings: **{ed}**")

    if run_btn:
        px=load_prices(ticker, period_sel, interval_sel)
        if px.empty:
            st.error(f"No data for '{ticker}'."); st.stop()
        ind=compute_indicators(px, ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=include_bb)
        if router_on:
            params=dict(ma_w=ma_window, rsi_p=rsi_period, use_weighted=use_weighted, w_ma=w_ma, w_rsi=w_rsi,
                        w_macd=w_macd, w_bb=w_bb, include_bb=include_bb, threshold=comp_thr, allow_short=allow_short)
            sig=route_strategy(ind, allow_short, params)
        else:
            sig=sig_composite(ind, ma_w=ma_window, rsi_p=rsi_period, use_weighted=use_weighted, w_ma=w_ma, w_rsi=w_rsi,
                              w_macd=w_macd, w_bb=w_bb, include_bb=include_bb, threshold=comp_thr, allow_short=allow_short)

        # exits overlays
        if sl_atr_mult>0 or tp_atr_mult>0:
            sig = RISKS.get("atr_trailing")(sig, mult=max(sl_atr_mult,1e-9))
        scale = risk_vol_target(sig, target=vol_target, interval=interval_sel)

        # position sizer
        sizer_fn = SIZERS.get(sizer_sel, size_fixed)
        size_series = sizer_fn(sig)

        bt, stats = backtest(sig, allow_short=allow_short, cost_bps=cost_bps,
                             scale_series=scale, sizer=size_series, interval=interval_sel)

        # dd throttle
        mult = risk_drawdown_throttle(bt["CumStrat"], soft=-0.1, hard=-0.25)
        if mult < 1.0:
            st.warning(f"De-risk throttle active: multiplier {mult:.2f}")

        last_trade = int(sig["Trade"].iloc[-1]) if not sig.empty else 0
        rec = rec_map.get(np.sign(last_trade), "🟡 HOLD")
        st.success(f"**{ticker}**: {rec}")

        m1,m2,m3,m4,m5,m6 = st.columns(6)
        m1.metric("CAGR", f"{stats['cagr']:.2f}%")
        m2.metric("Sharpe", f"{0 if np.isnan(stats['sharpe']) else stats['sharpe']:.2f}")
        m3.metric("Max DD", f"{stats['max_dd']:.2f}%")
        m4.metric("Win Rate", f"{stats['win']:.1f}%")
        m5.metric("Trades", f"{stats['trades']}")
        m6.metric("Time in Mkt", f"{stats['tim']:.1f}%")

        fig,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(11,12), sharex=True)
        ax1.plot(bt.index, bt["Close"], label="Close")
        macol=f"MA{ma_window}"
        if macol in bt: ax1.plot(bt.index, bt[macol], label=macol)
        if include_bb and {"BB_U","BB_L"}.issubset(bt.columns):
            ax1.plot(bt.index, bt["BB_U"], label="BB_U"); ax1.plot(bt.index, bt["BB_L"], label="BB_L")
        ax1.legend(); ax1.set_title("Price & Indicators")
        ax2.bar(bt.index, sig.get("Composite", pd.Series(0,index=bt.index)))
        ax2.set_title("Composite / Router Signal")
        ax3.plot(bt.index, bt["CumBH"], ":", label="BH")
        ax3.plot(bt.index, bt["CumStrat"], label="Strat"); ax3.legend(); ax3.set_title("Equity")
        plt.xticks(rotation=45); plt.tight_layout()
        st.pyplot(fig)

        od=simulate_execution(bt)
        if not od.empty:
            st.markdown("#### Execution log (simulated)")
            st.dataframe(od, use_container_width=True)

# ───────────────────────────── DSL Runner ─────────────────────────────
with tab_dsl:
    st.title("🧩 Strategy DSL (JSON)")
    DEFAULT_DSL = {
      "data": {"symbol":"AAPL","period":"1y","interval":"1d"},
      "indicators": {"ma_window":10,"rsi_period":14,"macd":[12,26,9],"include_bb": True},
      "signal": {"name":"composite_v2",
                 "params":{"use_weighted":True,"w_ma":1.0,"w_rsi":1.0,"w_macd":1.0,"w_bb":0.5,"threshold":1.0,"allow_short":False}},
      "sizer": {"name":"fixed_1x","params":{}},
      "risk": {"atr_trailing":{"mult":2.0},"vol_target":0.0,"cost_bps":5.0},
      "report": {"title":"QuantaraX Strategy Report"}
    }
    dsl_text = st.text_area("DSL JSON", json.dumps(DEFAULT_DSL, indent=2), height=260, key="dsl_text")
    if st.button("▶️ Run DSL", key="dsl_run"):
        try:
            spec=json.loads(dsl_text)
        except Exception as e:
            st.error(f"JSON error: {e}"); spec=None
        if spec:
            dcfg=spec.get("data",{}); sym=dcfg.get("symbol","AAPL"); period=dcfg.get("period","1y"); interval=dcfg.get("interval","1d")
            px=load_prices(sym, period, interval)
            if px.empty: st.error("No data."); st.stop()
            ic=spec.get("indicators",{}); ma=ic.get("ma_window",10); rp=ic.get("rsi_period",14); mf,ms,sg=ic.get("macd",[12,26,9]); inc=ic.get("include_bb",True)
            ind=compute_indicators(px, ma, rp, mf, ms, sg, use_bb=inc)
            sc=spec.get("signal",{"name":"composite_v2","params":{}}); sname=sc.get("name"); spar=sc.get("params",{})
            sig_fn=SIGNALS.get(sname)
            if not sig_fn: st.error(f"Unknown signal {sname}"); st.stop()
            sig=sig_fn(ind, ma_w=ma, rsi_p=rp, **spar)
            # risk
            rcfg=spec.get("risk",{}); cost=rcfg.get("cost_bps",0.0); vt=rcfg.get("vol_target",0.0)
            if "atr_trailing" in rcfg:
                sig = RISKS["atr_trailing"](sig, mult=float(rcfg["atr_trailing"].get("mult",2.0)))
            scale = risk_vol_target(sig, target=float(vt), interval=interval)
            # sizer
            scfg=spec.get("sizer", {"name":"fixed_1x","params":{}})
            sfn=SIZERS.get(scfg.get("name","fixed_1x"), size_fixed)
            sser=sfn(sig, **scfg.get("params",{}))
            bt, stats = backtest(sig, allow_short=bool(spar.get("allow_short",False)), cost_bps=float(cost),
                                 scale_series=scale, sizer=sser, interval=interval)
            c1,c2,c3,c4=st.columns(4)
            c1.metric("CAGR", f"{stats['cagr']:.2f}%")
            c2.metric("Sharpe", f"{0 if np.isnan(stats['sharpe']) else stats['sharpe']:.2f}")
            c3.metric("Max DD", f"{stats['max_dd']:.2f}%")
            c4.metric("Win", f"{stats['win']:.1f}%")
            fig, ax=plt.subplots(figsize=(10,4))
            ax.plot(bt.index, bt["CumBH"], ":", label="BH"); ax.plot(bt.index, bt["CumStrat"], label="Strategy")
            ax.legend(); ax.set_title("DSL Equity"); st.pyplot(fig)

# ───────────────────────────── ML (same as v7, hardened) ─────────────────────────────
with tab_ml:
    st.title("🧠 ML Lab — RF (OOS)")
    if not SK_OK: st.warning("Install scikit-learn to enable ML.")
    ml_sym = st.text_input("Symbol (ML)", "AAPL", key="ml_sym").upper()
    horizon = st.slider("Prediction horizon", 1, 5, 1, key="ml_h")
    train_frac = st.slider("Train fraction", 0.5, 0.95, 0.8, key="ml_tf")
    proba_enter = st.slider("Enter if P(long) ≥", 0.50, 0.85, 0.55, 0.01, key="ml_pe")
    proba_exit  = st.slider("Enter short if P(long) ≤", 0.15, 0.50, 0.45, 0.01, key="ml_px")
    if st.button("🤖 Train & Backtest", key="ml_btn") and SK_OK:
        px=load_prices(ml_sym, period_sel, interval_sel)
        ind=compute_indicators(px, ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=True)
        def feats(d):
            out=pd.DataFrame(index=d.index)
            out["ret1"]=d["Close"].pct_change(); out["ret5"]=d["Close"].pct_change(5); out["vol20"]=d["Close"].pct_change().rolling(20).std()
            out["rsi"]=d.get(f"RSI{rsi_period}", np.nan); out["macd"]=d.get("MACD", np.nan); out["sto_k"]=d.get("STO_K", np.nan)
            if {"BB_U","BB_L"}.issubset(d.columns):
                rng=(d["BB_U"]-d["BB_L"]).replace(0,np.nan); out["bb_pos"]=(d["Close"]-d["BB_L"])/rng
            else: out["bb_pos"]=np.nan
            return out.dropna()
        X=feats(ind)
        y=(ind["Close"].pct_change(horizon).shift(-horizon)>0).reindex(X.index).astype(int)
        data=pd.concat([X, y.rename("y")], axis=1).dropna()
        if len(data)<200: st.warning("Not enough rows for ML."); st.stop()
        split=int(len(data)*float(train_frac)); train,test=data.iloc[:split], data.iloc[split:]
        clf=RandomForestClassifier(n_estimators=400, max_depth=6, random_state=42, n_jobs=-1)
        clf.fit(train.drop(columns=["y"]), train["y"])
        proba=clf.predict_proba(test.drop(columns=["y"]))[:,1]; y_true=test["y"].values
        acc=accuracy_score(y_true,(proba>0.5).astype(int)); auc=roc_auc_score(y_true, proba) if len(np.unique(y_true))>1 else np.nan
        c1,c2=st.columns(2); c1.metric("Accuracy", f"{acc*100:.1f}%"); c2.metric("ROC-AUC", f"{0 if np.isnan(auc) else auc:.3f}")
        sig=np.where(proba>=proba_enter,1,(np.where(proba<=proba_exit,-1,0) if allow_short else 0))
        ml_df=ind.loc[test.index].copy(); ml_df["Trade"]=sig.astype(int)
        bt, stats = backtest(ml_df, allow_short=allow_short, cost_bps=cost_bps, interval=interval_sel)
        st.markdown(f"**ML OOS:** Return={(bt['CumStrat'].iloc[-1]-1)*100:.2f}% | Sharpe={0 if np.isnan(stats['sharpe']) else stats['sharpe']:.2f} | MaxDD={stats['max_dd']:.2f}%")
        fig, ax=plt.subplots(figsize=(9,3)); ax.plot(bt.index, bt["CumBH"], ":", label="BH"); ax.plot(bt.index, bt["CumStrat"], label="ML Strat"); ax.legend(); ax.set_title("ML OOS"); st.pyplot(fig)

# ───────────────────────────── Scanner ─────────────────────────────
with tab_scan:
    st.title("📡 Universe Scanner")
    universe = st.text_area("Tickers (comma-separated)", "AAPL, MSFT, NVDA, TSLA, AMZN, GOOGL, META, NFLX, SPY, QQQ", key="sc_uni").upper()
    if st.button("🔎 Scan", key="sc_btn"):
        rows=[]
        for t in [x.strip() for x in universe.split(",") if x.strip()]:
            try:
                px=load_prices(t, period_sel, interval_sel)
                if px.empty: continue
                ind=compute_indicators(px, ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=True)
                sig=sig_composite(ind, ma_w=ma_window, rsi_p=rsi_period, use_weighted=use_weighted,
                                  w_ma=w_ma, w_rsi=w_rsi, w_macd=w_macd, w_bb=w_bb, include_bb=True,
                                  threshold=comp_thr, allow_short=allow_short)
                comp=float(sig.get("Composite", pd.Series(index=sig.index,data=0.0)).iloc[-1]) if not sig.empty else 0.0
                rec=rec_map.get(int(np.sign(comp)), "🟡 HOLD")
                rows.append({"Ticker":t, "Composite":comp, "Signal":rec})
            except Exception:
                continue
        if rows:
            df=pd.DataFrame(rows).set_index("Ticker").sort_values(["Signal","Composite"], ascending=[True,False])
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No results.")

# ───────────────────────────── Regimes ─────────────────────────────
with tab_regime:
    st.title("📉 Regime Detection")
    sym = st.text_input("Symbol (Regime)", "SPY", key="rg_sym").upper()
    if st.button("Cluster Regimes", key="rg_btn"):
        px=load_prices(sym, "2y", "1d")
        ind=compute_indicators(px, ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=False)
        if ind.empty: st.error("Not enough data.")
        else:
            feat=pd.DataFrame(index=ind.index)
            feat["vol20"]=ind["Close"].pct_change().rolling(20).std()
            feat["mom20"]=ind["Close"].pct_change(20)
            feat["ma_slope"]=ind[f"MA{ma_window}"].diff()
            feat=feat.dropna()
            if HMM_OK:
                X=feat.fillna(0).values; hmm=GaussianHMM(n_components=3, covariance_type="diag", n_iter=100, random_state=0); hmm.fit(X); lab=hmm.predict(X)
            elif SK_OK:
                km=KMeans(n_clusters=3, n_init=10, random_state=42); lab=km.fit_predict(feat.fillna(0).values)
            else:
                q=feat.rank(pct=True).mean(axis=1); lab=(q>0.66).astype(int)+2*(q<0.33).astype(int)
            reg=pd.Series(lab, index=feat.index, name="Regime")
            joined=ind.join(reg, how="right")
            ret=joined["Close"].pct_change().groupby(joined["Regime"]).mean().sort_values()
            remap={old:i for i,old in enumerate(ret.index)}
            joined["Regime"]=joined["Regime"].map(remap)
            fig, ax=plt.subplots(figsize=(10,4))
            ax.plot(joined.index, joined["Close"], label="Close")
            for r in sorted(joined["Regime"].dropna().unique()):
                seg=joined[joined["Regime"]==r]
                ax.fill_between(seg.index, seg["Close"].min(), seg["Close"].max(), alpha=0.08)
            ax.set_title("Price with Regime Shading"); st.pyplot(fig)

# ───────────────────────────── Pairs Lab ─────────────────────────────
with tab_pairs:
    st.title("🔗 Pairs Lab (CI + Trade Sim)")
    a = st.text_input("Symbol A", "MSFT", key="pair_a").upper()
    b = st.text_input("Symbol B", "AAPL", key="pair_b").upper()
    if st.button("Analyze Pair", key="pair_btn"):
        A=load_prices(a,"1y","1d")["Close"].rename("A")
        B=load_prices(b,"1y","1d")["Close"].rename("B")
        df=pd.concat([A,B],axis=1).dropna()
        if df.empty: st.error("Not enough overlap.")
        else:
            beta=np.polyfit(df["B"], df["A"], 1)[0]
            spread=df["A"]-beta*df["B"]
            z=(spread-spread.rolling(60).mean())/spread.rolling(60).std(ddof=0)
            ci_p=None
            if SM_OK:
                _, ci_p, _ = coint(df["A"], df["B"])
            st.line_chart(z.rename("Spread Z"))
            st.write(f"β (A~B): {beta:.3f} | CI p={('%.3f'%ci_p) if ci_p is not None else 'n/a'}")
            # toy rules
            up=z>2; dn=z<-2; exit_=z.abs()<0.5
            trades=pd.DataFrame({"longA_shortB":dn, "shortA_longB":up, "exit":exit_})
            st.dataframe(trades.tail(15), use_container_width=True)

# ───────────────────────────── Portfolio ─────────────────────────────
with tab_port:
    st.title("💼 Portfolio — HRP/BL + Simulator")
    opt_tickers = st.text_input("Tickers", "AAPL, MSFT, TSLA, SPY, QQQ", key="pt_tix").upper()
    if st.button("🧮 Optimize", key="pt_btn"):
        tks=[t.strip() for t in opt_tickers.split(",") if t.strip()]
        rets=[]; valid=[]
        for t in tks:
            px=load_prices(t,"1y","1d")
            if px.empty: continue
            valid.append(t); rets.append(px["Close"].pct_change().dropna())
        if not rets: st.error("No valid data.")
        else:
            R=pd.concat(rets,axis=1); R.columns=valid
            base=hrp_weights(R)
            views=pd.Series(0.0,index=valid)
            for t in valid:
                ind=compute_indicators(load_prices(t,"6mo","1d"), ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=True)
                if ind.empty: continue
                sig=sig_composite(ind, ma_w=ma_window, rsi_p=rsi_period, use_weighted=True, w_ma=1, w_rsi=1, w_macd=1, w_bb=0.5, include_bb=True, threshold=1.0, allow_short=False)
                views[t]=float(sig["Composite"].iloc[-1]) if not sig.empty else 0.0
            w=bl_tilt(base, views, strength=0.15)
            st.dataframe(w.to_frame("Weight").T, use_container_width=True)
            fig, ax=plt.subplots(figsize=(5,5)); w.plot.pie(autopct="%.1f%%", ax=ax); ax.set_ylabel(""); st.pyplot(fig)

    st.subheader("📊 Portfolio Simulator (positions CSV)")
    holdings = st.text_area("e.g.\nAAPL,10,150\nMSFT,5,300", height=100, key="pt_pos")
    if st.button("▶️ Simulate Portfolio", key="pt_sim"):
        rows=[r.strip().split(",") for r in holdings.splitlines() if r.strip()]
        data=[]
        for idx,row in enumerate(rows,1):
            if len(row)!=3: st.warning(f"Skip row {idx}: {row}"); continue
            tkr,shares,cost = row; tkr=_map_symbol(tkr)
            try: s=float(shares); c=float(cost)
            except: st.warning(f"Bad numbers row {idx}"); continue
            hist=safe_yf_download(tkr,"1d","1d")
            if hist.empty: st.warning(f"No price {tkr}"); continue
            price=float(hist["Close"].iloc[-1]); invested=s*c; value=s*price; pnl=value-invested
            pnl_pct=(pnl/invested*100) if invested else np.nan
            # composite suggestion
            px=load_prices(tkr, period_sel, interval_sel); comp_sugg="N/A"
            if not px.empty:
                ind=compute_indicators(px, ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=include_bb)
                sig=sig_composite(ind, ma_w=ma_window, rsi_p=rsi_period, use_weighted=use_weighted,
                                  w_ma=w_ma, w_rsi=w_rsi, w_macd=w_macd, w_bb=w_bb, include_bb=include_bb,
                                  threshold=comp_thr, allow_short=allow_short)
                if not sig.empty:
                    score=float(sig["Composite"].iloc[-1]); comp_sugg="🟢 BUY" if score>=comp_thr else ("🔴 SELL" if score<=-comp_thr else "🟡 HOLD")
            suggestion="🔴 SELL" if pnl_pct>profit_target else ("🟢 BUY" if pnl_pct<-loss_limit else comp_sugg)
            data.append({"Ticker":tkr,"Shares":s,"Cost Basis":c,"Price":price,"Market Value":value,"Invested":invested,"P/L":pnl,"P/L %":pnl_pct,"Composite Sig":comp_sugg,"Suggestion":suggestion})
        if data:
            dfp=pd.DataFrame(data).set_index("Ticker")
            st.dataframe(dfp, use_container_width=True)
            c1,c2,c3=st.columns(3)
            c1.metric("Total MV", f"${dfp['Market Value'].sum():,.2f}")
            c2.metric("Total Invested", f"${dfp['Invested'].sum():,.2f}")
            c3.metric("Total P/L", f"${(dfp['Market Value'].sum()-dfp['Invested'].sum()):,.2f}")
            fig, ax=plt.subplots(figsize=(5,5)); dfp["Market Value"].plot.pie(autopct="%.1f%%", ax=ax); ax.set_ylabel(""); st.pyplot(fig)
        else:
            st.error("No valid holdings.")

# ───────────────────────────── Optimizer (Optuna UI) ─────────────────────────────
with tab_opt:
    st.title("🎛️ Walk-Forward Optimizer (Optuna)")
    if not OPTUNA_OK:
        st.warning("Install `optuna` to enable optimizer.")
    sym = st.text_input("Symbol", "AAPL", key="opt_sym").upper()
    n_trials = st.slider("Trials", 20, 200, 60, 10, key="opt_trials")
    if st.button("Run Study", key="opt_run") and OPTUNA_OK:
        study, err = run_optuna(sym, period_sel, interval_sel, n_trials, allow_short)
        if err: st.error(err)
        else:
            st.success(f"Study complete: {len(study.trials)} trials.")
            best = max(study.best_trials, key=lambda t: (t.values[1], -t.values[0]))  # Sharpe first, then DD
            st.json(best.params)
            st.download_button("Download best params", data=json.dumps(best.params, indent=2), file_name=f"qx_best_{sym}.json")

# ───────────────────────────── Reports ─────────────────────────────
with tab_reports:
    st.title("📄 Auto-Reports")
    rep_sym = st.text_input("Symbol (Report)", "AAPL", key="rp_sym").upper()
    if st.button("Generate HTML/PDF", key="rp_btn"):
        px=load_prices(rep_sym,"1y","1d")
        ind=compute_indicators(px, 10, 14, 12, 26, 9, use_bb=True)
        sig=sig_composite(ind, ma_w=10, rsi_p=14, use_weighted=True, w_ma=1, w_rsi=1, w_macd=1, w_bb=0.5, include_bb=True, threshold=1.0, allow_short=False)
        bt, stats = backtest(sig, allow_short=False, cost_bps=5.0, interval="1d")
        html=f"""<html><body><h2>QuantaraX Report — {rep_sym}</h2>
        <p>Sharpe: {0 if np.isnan(stats['sharpe']) else round(stats['sharpe'],2)} | MaxDD: {round(stats['max_dd'],2)}% | CAGR: {round(stats['cagr'],2)}%</p>
        <p>Bars: {len(bt)}</p></body></html>"""
        st.download_button("Download HTML", data=html, file_name=f"qx_report_{rep_sym}.html", mime="text/html", key="rp_dl_html")
        if RL_OK:
            buf=io.BytesIO(); c=pdf_canvas.Canvas(buf)
            c.setFont("Helvetica-Bold",14); c.drawString(72,760,f"QuantaraX Report — {rep_sym}")
            c.setFont("Helvetica",12); c.drawString(72,740,f"Sharpe: {0 if np.isnan(stats['sharpe']) else round(stats['sharpe'],2)}")
            c.drawString(72,725,f"MaxDD: {round(stats['max_dd'],2)}%  |  CAGR: {round(stats['cagr'],2)}%")
            c.save(); st.download_button("Download PDF", data=buf.getvalue(), file_name=f"qx_report_{rep_sym}.pdf", mime="application/pdf", key="rp_dl_pdf")
        else:
            st.info("Install reportlab for PDF export.")

# ───────────────────────────── Health ─────────────────────────────
with tab_health:
    st.title("🩺 Data & System Health")
    h_sym = st.text_input("Symbol (health)", "AAPL", key="hl_sym").upper()
    if st.button("Check", key="hl_btn"):
        d1=load_prices(h_sym,"6mo","1d"); dH=load_prices(h_sym,"5d","1h")
        issues=[]
        if d1.empty: issues.append("Daily data empty.")
        if not d1.empty and d1['Close'].isna().mean()>0.05: issues.append("Daily data: >5% NaNs.")
        if dH.empty: issues.append("Hourly data empty.")
        if not dH.empty and dH.index.duplicated().any(): issues.append("Hourly index duplicates.")
        if not issues:
            st.success("All checks passed.")
        else:
            st.error(" • " + " | ".join(issues))
    st.caption(f"Optional modules — duckdb:{DUCK_OK} optuna:{OPTUNA_OK} scipy:{SCIPY_OK} statsmodels:{SM_OK} hmmlearn:{HMM_OK} sklearn:{SK_OK} reportlab:{RL_OK}")

# ───────────────────────────── Help ─────────────────────────────
with tab_help:
    st.header("How QuantaraX ∞ (v8) Works")
    st.markdown("""
- **Router**: picks trend/breakout vs composite based on simple regime score.
- **Sizer plugins**: fixed / capped Kelly / composite-scaled.
- **Risk**: trailing ATR exits, vol target, drawdown throttle.
- **Optimizer**: Optuna multi-objective; export best params.
- **Cache**: DuckDB (optional) + Streamlit cache.
- **News & Earnings**: safe wrappers, no crashes.
- **Profiles**: save/load your whole setup as JSON.
- **No duplicate widget IDs**: every widget has a unique key.
""")
