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
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ───────────────────────────── Sidebar Controls ─────────────────────────────
st.sidebar.header("Controls")
if st.sidebar.button("🔄 Reset to defaults"):
    for key, val in DEFAULTS.items():
        st.session_state[key] = val

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

    # Moving Average
    ma_col = f"MA{ma_w}"
    df[ma_col] = df["Close"].rolling(ma_w).mean()

    # RSI
    delta    = df["Close"].diff()
    up       = delta.clip(lower=0)
    down     = -delta.clip(upper=0)
    ema_up   = up.ewm(com=rsi_p-1, adjust=False).mean()
    ema_down = down.ewm(com=rsi_p-1, adjust=False).mean()
    rsi_col  = f"RSI{rsi_p}"
    df[rsi_col] = 100 - 100/(1 + ema_up/ema_down)

    # MACD
    ema_f = df["Close"].ewm(span=mf, adjust=False).mean()
    ema_s = df["Close"].ewm(span=ms, adjust=False).mean()
    macd  = ema_f - ema_s
    macd_sig = macd.ewm(span=sig, adjust=False).mean()
    df["MACD"]        = macd
    df["MACD_Signal"] = macd_sig

    # Only dropna if we actually created any of those columns
    wanted = [ma_col, rsi_col, "MACD", "MACD_Signal"]
    present = [c for c in wanted if c in df.columns]
    if present:
        df = df.dropna(subset=present).reset_index(drop=True)

    return df

# ───────────────────────────── Build Composite Signals ─────────────────────────────
def build_composite(df):
    n     = len(df)
    cl    = df["Close"].to_numpy()
    ma    = df[f"MA{ma_window}"].to_numpy()
    rs    = df[f"RSI{rsi_period}"].to_numpy()
    mc    = df["MACD"].to_numpy()
    ms    = df["MACD_Signal"].to_numpy()

    df["MA_Signal"]    = 0
    df["RSI_Signal"]   = 0
    df["MACD_Signal2"] = 0
    df["Composite"]    = 0
    df["Trade"]        = 0

    for i in range(1, n):
        # MA crossover
        if cl[i-1] < ma[i-1] and cl[i] > ma[i]:
            df.at[i, "MA_Signal"] = 1
        elif cl[i-1] > ma[i-1] and cl[i] < ma[i]:
            df.at[i, "MA_Signal"] = -1

        # RSI thresholds
        if rs[i] < 30:
            df.at[i, "RSI_Signal"] = 1
        elif rs[i] > 70:
            df.at[i, "RSI_Signal"] = -1

        # MACD crossover
        if mc[i-1] < ms[i-1] and mc[i] > ms[i]:
            df.at[i, "MACD_Signal2"] = 1
        elif mc[i-1] > ms[i-1] and mc[i] < ms[i]:
            df.at[i, "MACD_Signal2"] = -1

        # Composite
        tot = df.at[i, "MA_Signal"] + df.at[i, "RSI_Signal"] + df.at[i, "MACD_Signal2"]
        df.at[i, "Composite"] = tot
        df.at[i, "Trade"]     = np.sign(tot)

    return df

# ───────────────────────────── Backtest ─────────────────────────────
def backtest(df):
    df = df.copy()
    df["Return"]   = df["Close"].pct_change().fillna(0)
    df["Position"] = df["Trade"].shift(1).fillna(0).clip(0,1)
    df["StratRet"] = df["Position"] * df["Return"]
    df["CumBH"]    = (1 + df["Return"]).cumprod()
    df["CumStrat"] = (1 + df["StratRet"]).cumprod()

    dd      = df["CumStrat"] / df["CumStrat"].cummax() - 1
    max_dd  = dd.min() * 100
    sharpe  = df["StratRet"].mean() / df["StratRet"].std() * np.sqrt(252)
    win_rt  = (df["StratRet"] > 0).mean() * 100

    return df, max_dd, sharpe, win_rt

# ───────────────────────────── Single‐Ticker Backtest ─────────────────────────────
st.markdown("### Single‐Ticker Backtest")
ticker = st.text_input("Ticker (e.g. AAPL)", "AAPL").upper()

if st.button("▶️ Run Composite Backtest"):
    df = load_and_compute(ticker, ma_window, rsi_period, macd_fast, macd_slow, macd_signal)
    if df.empty:
        st.error(f"No data for '{ticker}'.")
        st.stop()

    df = build_composite(df)
    df, max_dd, sharpe, win_rt = backtest(df)

    rec_map = {1: "🟢 BUY", 0: "🟡 HOLD", -1: "🔴 SELL"}
    st.success(f"**{ticker}**: {rec_map[int(df['Trade'].iloc[-1])]}")

    bh_ret    = (df["CumBH"].iloc[-1] - 1) * 100
    strat_ret = (df["CumStrat"].iloc[-1] - 1) * 100
    st.markdown(f"""
**Buy & Hold:** {bh_ret:.2f}%  
**Strategy:**   {strat_ret:.2f}%  
**Sharpe:**     {sharpe:.2f}  
**Max Drawdown:** {max_dd:.2f}%  
**Win Rate:**     {win_rt:.1f}%  
    """)

    # Plot
    fig, axes = plt.subplots(3,1,figsize=(10,12),sharex=True)
    axes[0].plot(df["Close"], label="Close");        axes[0].plot(df[f"MA{ma_window}"], label=f"MA{ma_window}")
    axes[0].legend(); axes[0].set_title("Price & MA")

    axes[1].bar(df.index, df["Composite"], color="purple")
    axes[1].set_title("Composite Vote")

    axes[2].plot(df["CumBH"],":", label="Buy & Hold"); axes[2].plot(df["CumStrat"],"-", label="Strategy")
    axes[2].legend(); axes[2].set_title("Equity Curves")

    plt.xticks(rotation=45); plt.tight_layout()
    st.pyplot(fig)

# ───────────────────────────── Batch Backtest ─────────────────────────────
st.markdown("---")
st.markdown("### Batch Backtest")
batch = st.text_area("Enter tickers (comma-separated)", "AAPL, MSFT, TSLA, SPY, QQQ").upper()

if st.button("▶️ Run Batch Backtest"):
    perf = []
    for t in [x.strip() for x in batch.split(",") if x.strip()]:
        df_t = load_and_compute(t, ma_window, rsi_period, macd_fast, macd_slow, macd_signal)
        if df_t.empty: continue
        df_t = build_composite(df_t)
        df_t, max_dd, sharpe, win_rt = backtest(df_t)
        perf.append({
            "Ticker": t,
            "Composite": df_t["Composite"].iloc[-1],
            "Signal": {1:"BUY",0:"HOLD",-1:"SELL"}[int(df_t["Trade"].iloc[-1])],
            "BuyHold%": (df_t["CumBH"].iloc[-1]-1)*100,
            "Strat%":   (df_t["CumStrat"].iloc[-1]-1)*100,
            "Sharpe": sharpe,
            "MaxDD%": max_dd,
            "Win%": win_rt
        })

    if not perf:
        st.error("No valid tickers/data.")
    else:
        df_perf = pd.DataFrame(perf).set_index("Ticker")
        st.dataframe(df_perf)
        st.download_button("Download performance CSV", df_perf.to_csv(), "batch_perf.csv")
