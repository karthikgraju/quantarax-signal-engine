import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="QuantaraX Composite Signals", layout="wide")

# ─── Defaults ─────────────────────────────────────────────────────────────────────
DEFAULTS = dict(
    ma_window=10,
    rsi_period=14,
    macd_fast=12,
    macd_slow=26,
    macd_signal=9
)
# initialize session state
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── Sidebar ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")
    if st.button("🔄 Reset to defaults"):
        for k, v in DEFAULTS.items():
            st.session_state[k] = v

    st.header("Hyper-parameters")
    ma_window   = st.slider("MA window",        5, 50, key="ma_window")
    rsi_period  = st.slider("RSI lookback",     5, 30, key="rsi_period")
    macd_fast   = st.slider("MACD fast span",   5, 20, key="macd_fast")
    macd_slow   = st.slider("MACD slow span",  20, 40, key="macd_slow")
    macd_signal = st.slider("MACD signal span", 5, 20, key="macd_signal")

    st.markdown("---")
    st.header("Grid Search Ranges")
    grid_ma    = st.multiselect("MA windows to test",      [5,10,15,20,30],    default=[10])
    grid_rsi   = st.multiselect("RSI lookbacks to test",   [5,10,14,20,30],    default=[14])
    grid_fast  = st.multiselect("MACD fast spans to test",[5,10,12,15,20],    default=[12])
    grid_slow  = st.multiselect("MACD slow spans to test",[20,26,30,35,40],  default=[26])
    grid_sig   = st.multiselect("MACD sig spans to test", [5,9,12,15,20],    default=[9])

# ─── Title ───────────────────────────────────────────────────────────────────────
st.title("🚀 QuantaraX — Composite Signal Engine")
st.write("MA + RSI + MACD Signals & Backtesting")

# ─── Indicator Computation ───────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_compute(ticker, ma_w, rsi_p, mf, ms, sig):
    # 1) fetch
    df = yf.download(ticker, period="6mo", progress=False)
    if df.empty or "Close" not in df:
        return pd.DataFrame()

    # require at least enough rows for the largest lookback
    min_rows = max(ma_w, rsi_p, ms+sig)
    if len(df) < min_rows:
        return pd.DataFrame()

    # 2) moving average
    df["MA"] = df["Close"].rolling(ma_w).mean()

    # 3) RSI
    delta    = df["Close"].diff()
    up       = delta.clip(lower=0)
    down     = -delta.clip(upper=0)
    ema_up   = up.ewm(com=rsi_p-1, adjust=False).mean()
    ema_dn   = down.ewm(com=rsi_p-1, adjust=False).mean()
    df["RSI"] = 100 - 100/(1 + ema_up/ema_dn)

    # 4) MACD & signal
    ema_f = df["Close"].ewm(span=mf, adjust=False).mean()
    ema_s = df["Close"].ewm(span=ms, adjust=False).mean()
    macd  = ema_f - ema_s
    df["MACD"]        = macd
    df["MACD_Signal"] = macd.ewm(span=sig, adjust=False).mean()

    # 5) drop any rows missing those four
    req = ["MA","RSI","MACD","MACD_Signal"]
    df = df.dropna(subset=req).reset_index(drop=True)
    return df

# ─── Build Composite Signals ─────────────────────────────────────────────────────
def build_signals(df):
    n      = len(df)
    close  = df["Close"].to_numpy()
    ma     = df["MA"].to_numpy()
    rsi    = df["RSI"].to_numpy()
    macd   = df["MACD"].to_numpy()
    sig    = df["MACD_Signal"].to_numpy()

    ma_s    = np.zeros(n, int)
    rsi_s   = np.zeros(n, int)
    macd_s  = np.zeros(n, int)
    comp    = np.zeros(n, int)
    trade   = np.zeros(n, int)

    for i in range(1,n):
        # MA crossover
        if close[i-1]<ma[i-1] and close[i]>ma[i]:
            ma_s[i]= 1
        elif close[i-1]>ma[i-1] and close[i]<ma[i]:
            ma_s[i]=-1

        # RSI oversold/overbought
        if rsi[i]<30:
            rsi_s[i]= 1
        elif rsi[i]>70:
            rsi_s[i]=-1

        # MACD crossover
        if macd[i-1]<sig[i-1] and macd[i]>sig[i]:
            macd_s[i]= 1
        elif macd[i-1]>sig[i-1] and macd[i]<sig[i]:
            macd_s[i]=-1

        comp[i]  = ma_s[i]+rsi_s[i]+macd_s[i]
        trade[i] = np.sign(comp[i])

    df["MA_Signal"   ] = ma_s
    df["RSI_Signal"  ] = rsi_s
    df["MACD_Signal2"] = macd_s
    df["Composite"   ] = comp
    df["Trade"       ] = trade
    return df

# ─── Backtest & Stats ───────────────────────────────────────────────────────────
def backtest(df):
    df = df.copy()
    df["Ret"     ] = df["Close"].pct_change().fillna(0)
    df["Pos"     ] = df["Trade"].shift(1).fillna(0).clip(0,1)
    df["StratRet"] = df["Pos"] * df["Ret"]
    df["CumBH"   ] = (1+df["Ret"]).cumprod()
    df["CumStrat"] = (1+df["StratRet"]).cumprod()

    dd     = df["CumStrat"]/df["CumStrat"].cummax() - 1
    max_dd = dd.min()*100
    vol    = df["StratRet"].std()
    sharpe = (df["StratRet"].mean()/vol*np.sqrt(252)) if vol else np.nan
    win    = (df["StratRet"]>0).mean()*100
    return df, max_dd, sharpe, win

# ─── Single Ticker ───────────────────────────────────────────────────────────────
st.subheader("🔎 Single‐Ticker Backtest")
ticker = st.text_input("Ticker (e.g. AAPL)", "AAPL").upper()
if st.button("▶️ Run Composite Backtest"):
    df1 = load_and_compute(ticker,ma_window,rsi_period,macd_fast,macd_slow,macd_signal)
    if df1.empty:
        st.error(f"Insufficient data for {ticker}")
        st.stop()

    df1 = build_signals(df1)
    df1, maxdd, sharpe, win = backtest(df1)
    rec = {1:"🟢 BUY",0:"🟡 HOLD",-1:"🔴 SELL"}[int(df1["Trade"].iloc[-1])]
    st.success(f"**{ticker}** → {rec}")

    bh  = (df1["CumBH"].iloc[-1]-1)*100
    sr  = (df1["CumStrat"].iloc[-1]-1)*100
    st.markdown(f"""
- **Buy & Hold:**   {bh:.2f}%  
- **Strategy:**     {sr:.2f}%  
- **Sharpe:**       {sharpe:.2f}  
- **Max Drawdown:** {maxdd:.2f}%  
- **Win Rate:**     {win:.1f}%  
    """)

    fig, ax = plt.subplots(2,1,figsize=(10,8), sharex=True)
    ax[0].plot(df1["Close"], label="Close"); ax[0].plot(df1["MA"], label=f"MA{ma_window}")
    ax[0].legend(); ax[0].set_title("Price & MA")
    ax[1].plot(df1["CumBH"], ":", label="Buy & Hold")
    ax[1].plot(df1["CumStrat"], "-", label="Strategy")
    ax[1].legend(); ax[1].set_title("Equity Curves")
    plt.xticks(rotation=45); plt.tight_layout()
    st.pyplot(fig)

# ─── Batch Backtest ──────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Batch Backtest")
batch = st.text_area("Enter tickers (comma-separated)", "AAPL, MSFT, TSLA, SPY, QQQ").upper()
if st.button("▶️ Run Batch Backtest"):
    rows=[]
    for t in [x.strip() for x in batch.split(",") if x.strip()]:
        df_b = load_and_compute(t,ma_window,rsi_period,macd_fast,macd_slow,macd_signal)
        if df_b.empty: continue
        df_b = build_signals(df_b)
        df_b, maxdd, sharpe, win = backtest(df_b)
        rows.append({
            "Ticker":      t,
            "Composite":   int(df_b["Composite"].iloc[-1]),
            "Signal":      {1:"BUY",0:"HOLD",-1:"SELL"}[int(df_b["Trade"].iloc[-1])],
            "BuyHold %":   (df_b["CumBH"].iloc[-1]-1)*100,
            "Strat %":     (df_b["CumStrat"].iloc[-1]-1)*100,
            "Sharpe":      sharpe,
            "Max DD %":    maxdd,
            "Win %":       win
        })
    if not rows:
        st.error("No valid tickers/data.")
    else:
        pdf = pd.DataFrame(rows).set_index("Ticker")
        st.dataframe(pdf)
        st.download_button("Download performance CSV", pdf.to_csv(), "batch_perf.csv")

# ─── Hyper-parameter Grid Search ──────────────────────────────────────────────────
st.markdown("---")
st.subheader("🔧 Hyper-parameter Optimization")
gs_ticker = st.text_input("Grid Search Ticker", "AAPL").upper()
if st.button("🏃 Run Grid Search"):
    results=[]
    for mw in grid_ma:
      for rp in grid_rsi:
        for mf in grid_fast:
          for ms in grid_slow:
            for sg in grid_sig:
                dfg = load_and_compute(gs_ticker,mw,rp,mf,ms,sg)
                if dfg.empty: continue
                dfg = build_signals(dfg)
                dfg, maxdd, sharpe, win = backtest(dfg)
                results.append({
                  "MA":mw,"RSI":rp,"MF":mf,"MS":ms,"SG":sg,
                  "Strat %":(dfg["CumStrat"].iloc[-1]-1)*100,
                  "Sharpe":sharpe
                })
    if not results:
        st.error("No combos produced data.")
    else:
        gdf = (pd.DataFrame(results)
                .sort_values("Strat %", ascending=False)
                .reset_index(drop=True))
        st.dataframe(gdf)
        st.download_button("Download grid results CSV", gdf.to_csv(index=False), "grid.csv")
