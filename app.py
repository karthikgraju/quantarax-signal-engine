import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ───────────────────────────── Defaults & Session State ─────────────────────────────
DEFAULTS = {
    "ma_window":   10,
    "rsi_period":  14,
    "macd_fast":   12,
    "macd_slow":   26,
    "macd_signal":  9
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ───────────────────────────── Sidebar Controls ─────────────────────────────
st.sidebar.header("Controls")
if st.sidebar.button("🔄 Reset to defaults"):
    for k, v in DEFAULTS.items():
        st.session_state[k] = v

st.sidebar.header("Indicator Parameters")
ma_window   = st.sidebar.slider("MA window",        5, 50, st.session_state["ma_window"],   key="ma_window")
rsi_period  = st.sidebar.slider("RSI lookback",     5, 30, st.session_state["rsi_period"],  key="rsi_period")
macd_fast   = st.sidebar.slider("MACD fast span",   5, 20, st.session_state["macd_fast"],   key="macd_fast")
macd_slow   = st.sidebar.slider("MACD slow span",  20, 40, st.session_state["macd_slow"],   key="macd_slow")
macd_signal = st.sidebar.slider("MACD signal span", 5, 20, st.session_state["macd_signal"], key="macd_signal")

# ───────────────────────────── Page Setup ─────────────────────────────
st.set_page_config(page_title="QuantaraX Composite Signals", layout="centered")
st.title("🚀 QuantaraX — Composite Signal Engine")
st.write("MA + RSI + MACD Composite Signals & Backtest")

# ───────────────────────────── Load & Compute Indicators ─────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_compute(ticker, ma_w, rsi_p, mf, ms, sig):
    df = yf.download(ticker, period="6mo", progress=False)
    if df.empty or "Close" not in df:
        return pd.DataFrame()

    # GENERIC MOVING AVERAGE
    df["MA"] = df["Close"].rolling(window=ma_w).mean()

    # GENERIC RSI
    delta    = df["Close"].diff()
    up       = delta.clip(lower=0)
    down     = -delta.clip(upper=0)
    ema_up   = up.ewm(com=rsi_p-1, adjust=False).mean()
    ema_down = down.ewm(com=rsi_p-1, adjust=False).mean()
    df["RSI"] = 100 - 100/(1 + ema_up/ema_down)

    # GENERIC MACD
    ema_f = df["Close"].ewm(span=mf, adjust=False).mean()
    ema_s = df["Close"].ewm(span=ms, adjust=False).mean()
    macd  = ema_f - ema_s
    df["MACD"]        = macd
    df["MACD_Signal"] = macd.ewm(span=sig, adjust=False).mean()

    # DROP ANY ROWS MISSING ANY OF OUR FOUR INDICATORS
    required = ["MA","RSI","MACD","MACD_Signal"]
    df = df.dropna(subset=required).reset_index(drop=True)
    return df

# ───────────────────────────── Build Composite Signals ─────────────────────────────
def build_composite(df):
    n         = len(df)
    closes    = df["Close"].to_numpy()
    mas       = df["MA"].to_numpy()
    rsis      = df["RSI"].to_numpy()
    macds     = df["MACD"].to_numpy()
    sigs      = df["MACD_Signal"].to_numpy()

    ma_sig    = np.zeros(n, dtype=int)
    rsi_sig   = np.zeros(n, dtype=int)
    macd_sig2 = np.zeros(n, dtype=int)
    comp      = np.zeros(n, dtype=int)
    trade     = np.zeros(n, dtype=int)

    for i in range(1,n):
        # MA crossover
        if closes[i-1] < mas[i-1] and closes[i] > mas[i]:
            ma_sig[i] =  1
        elif closes[i-1] > mas[i-1] and closes[i] < mas[i]:
            ma_sig[i] = -1

        # RSI thresholds
        if rsis[i] < 30:
            rsi_sig[i] =  1
        elif rsis[i] > 70:
            rsi_sig[i] = -1

        # MACD crossover
        if macds[i-1] < sigs[i-1] and macds[i] > sigs[i]:
            macd_sig2[i] =  1
        elif macds[i-1] > sigs[i-1] and macds[i] < sigs[i]:
            macd_sig2[i] = -1

        # Composite vote and resulting trade
        comp[i]  = ma_sig[i] + rsi_sig[i] + macd_sig2[i]
        trade[i] = np.sign(comp[i])

    df["MA_Signal"]    = ma_sig
    df["RSI_Signal"]   = rsi_sig
    df["MACD_Signal2"] = macd_sig2
    df["Composite"]    = comp
    df["Trade"]        = trade
    return df

# ───────────────────────────── Backtest & Metrics ─────────────────────────────
def backtest(df):
    df = df.copy()
    df["Return"]   = df["Close"].pct_change().fillna(0)
    df["Position"] = df["Trade"].shift(1).fillna(0).clip(0,1)
    df["StratRet"] = df["Position"] * df["Return"]
    df["CumBH"]    = (1 + df["Return"]).cumprod()
    df["CumStrat"] = (1 + df["StratRet"]).cumprod()

    dd      = df["CumStrat"] / df["CumStrat"].cummax() - 1
    max_dd  = dd.min() * 100
    std_dev = df["StratRet"].std()
    sharpe  = (df["StratRet"].mean() / std_dev * np.sqrt(252)) if std_dev else np.nan
    win_rt  = (df["StratRet"] > 0).mean() * 100

    return df, max_dd, sharpe, win_rt

# ───────────────────────────── Single‐Ticker Backtest ─────────────────────────────
st.markdown("## Single‐Ticker Backtest")
ticker = st.text_input("Ticker (e.g. AAPL)", "AAPL").upper()

if st.button("▶️ Run Composite Backtest"):
    df = load_and_compute(ticker, ma_window, rsi_period, macd_fast, macd_slow, macd_signal)
    if df.empty:
        st.error(f"No data for '{ticker}'."); st.stop()

    df = build_composite(df)
    df, max_dd, sharpe, win_rt = backtest(df)

    rec = {1:"🟢 BUY",0:"🟡 HOLD",-1:"🔴 SELL"}[int(df["Trade"].iloc[-1])]
    st.success(f"**{ticker}**: {rec}")

    bh_ret    = (df["CumBH"].iloc[-1] - 1) * 100
    strat_ret = (df["CumStrat"].iloc[-1] - 1) * 100
    st.markdown(f"""
- **Buy & Hold:**    {bh_ret:.2f}%  
- **Strategy:**      {strat_ret:.2f}%  
- **Sharpe:**        {sharpe:.2f}  
- **Max Drawdown:**  {max_dd:.2f}%  
- **Win Rate:**      {win_rt:.1f}%  
    """)

    # Plot 3‐panel chart
    fig, axes = plt.subplots(3,1,figsize=(10,12),sharex=True)
    axes[0].plot(df["Close"], label="Close")
    axes[0].plot(df["MA"],    label=f"MA{ma_window}")
    axes[0].legend(); axes[0].set_title("Price & MA")

    axes[1].bar(df.index, df["Composite"], color="purple")
    axes[1].set_title("Composite Vote")

    axes[2].plot(df["CumBH"],    ":", label="Buy & Hold")
    axes[2].plot(df["CumStrat"], "-", label="Strategy")
    axes[2].legend(); axes[2].set_title("Equity Curves")

    plt.xticks(rotation=45); plt.tight_layout()
    st.pyplot(fig)

# ───────────────────────────── Batch Backtest ─────────────────────────────
st.markdown("---")
st.markdown("## Batch Backtest")
batch = st.text_area("Enter tickers (comma-separated)", "AAPL, MSFT, TSLA, SPY, QQQ").upper()

if st.button("▶️ Run Batch Backtest"):
    perf = []
    for t in [s.strip() for s in batch.split(",") if s.strip()]:
        df_t = load_and_compute(t, ma_window, rsi_period, macd_fast, macd_slow, macd_signal)
        if df_t.empty: continue
        df_t = build_composite(df_t)
        df_t, max_dd, sharpe, win_rt = backtest(df_t)
        perf.append({
            "Ticker":     t,
            "Composite":  int(df_t["Composite"].iloc[-1]),
            "Signal":     {1:"BUY",0:"HOLD",-1:"SELL"}[int(df_t["Trade"].iloc[-1])],
            "BuyHold %":  (df_t["CumBH"].iloc[-1]-1)*100,
            "Strategy %": (df_t["CumStrat"].iloc[-1]-1)*100,
            "Sharpe":     sharpe,
            "Max DD %":   max_dd,
            "Win %":      win_rt
        })
    if not perf:
        st.error("No valid tickers/data.")
    else:
        df_perf = pd.DataFrame(perf).set_index("Ticker")
        st.dataframe(df_perf)
        st.download_button("Download performance CSV", df_perf.to_csv(), "batch_perf.csv")

# ───────────────────────────── Hyperparameter Optimization ─────────────────────────────
st.markdown("---")
st.markdown("## 🛠️ Hyperparameter Optimization")
ma_list   = st.multiselect("MA windows to test",     [5,10,15,20,30], default=[ma_window])
rsi_list  = st.multiselect("RSI lookbacks to test",  [7,14,21,28],     default=[rsi_period])
mf_list   = st.multiselect("MACD fast spans to test",[8,12,16,20],     default=[macd_fast])
ms_list   = st.multiselect("MACD slow spans to test",[20,26,32,40],   default=[macd_slow])
sig_list  = st.multiselect("MACD sig spans to test", [5,9,12,16],      default=[macd_signal])

if st.button("🏃‍♂️ Run Grid Search"):
    df_full = load_and_compute(ticker, ma_window, rsi_period, macd_fast, macd_slow, macd_signal)
    if df_full.empty:
        st.error(f"No data for {ticker}"); st.stop()

    results=[]
    with st.spinner("Testing combos..."):
        for mw in ma_list:
            for rp in rsi_list:
                for mf in mf_list:
                    for ms in ms_list:
                        for sg in sig_list:
                            df_i = load_and_compute(ticker, mw, rp, mf, ms, sg)
                            if df_i.empty: continue
                            df_i = build_composite(df_i)
                            df_i, max_dd, sharpe_i, win_rt_i = backtest(df_i)
                            strat_ret = (df_i["CumStrat"].iloc[-1]-1)*100
                            results.append({
                                "MA": mw, "RSI": rp,
                                "MACD_F": mf, "MACD_S": ms, "MACD_Sig": sg,
                                "Strat %": strat_ret, "Sharpe": sharpe_i,
                                "MaxDD %": max_dd, "Win %": win_rt_i
                            })

    if not results:
        st.error("No valid combos.")
    else:
        df_grid = pd.DataFrame(results).sort_values("Strat %", ascending=False).reset_index(drop=True)
        st.dataframe(df_grid.head(10), use_container_width=True)
        st.download_button("Download full grid CSV", df_grid.to_csv(index=False), "grid.csv")
