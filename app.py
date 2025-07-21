import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# — Streamlit page config
st.set_page_config(page_title="QuantaraX Signal Engine", layout="centered")
st.title("🚀 QuantaraX — Smart Signal Engine")
st.subheader("🔍 10-Day MA Crossover Signals & Backtest")

@st.cache_data
def load_data(ticker: str) -> pd.DataFrame:
    # 1) Download 6 months of daily data
    df = yf.download(ticker, period="6mo", progress=False)
    if df.empty or "Close" not in df.columns:
        return pd.DataFrame()
    # 2) Compute 10-day moving average
    df["MA10"] = df["Close"].rolling(window=10).mean()
    # 3) Filter out the initial NaNs and reset to integer index
    df = df[df["MA10"].notna()].reset_index(drop=True)
    return df

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n = len(df)
    signals = [0] * n

    # Column positions for fast scalar access
    close_col = df.columns.get_loc("Close")
    ma_col    = df.columns.get_loc("MA10")

    # 4) Simple loop: compare yesterday's Close/MA vs today's
    for i in range(1, n):
        prev_close = df.iat[i-1, close_col]
        prev_ma    = df.iat[i-1, ma_col]
        curr_close = df.iat[i,   close_col]
        curr_ma    = df.iat[i,   ma_col]

        if prev_close < prev_ma and curr_close > curr_ma:
            signals[i] = 1    # Bullish crossover
        elif prev_close > prev_ma and curr_close < curr_ma:
            signals[i] = -1   # Bearish crossover
        # else 0 = no signal

    df["signal"] = signals
    return df

def backtest(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["return"]    = df["Close"].pct_change().fillna(0)
    # Go long (position=1) if yesterday's signal was bullish
    df["position"]  = df["signal"].shift(1).fillna(0).clip(lower=0)
    df["strat_ret"] = df["position"] * df["return"]
    df["cum_bh"]    = (1 + df["return"]).cumprod()
    df["cum_strat"] = (1 + df["strat_ret"]).cumprod()
    return df

# — Main app
ticker = st.text_input("Ticker (e.g. AAPL)", "AAPL").upper()

if st.button("▶️ Run Signals & Backtest"):
    df = load_data(ticker)
    if df.empty:
        st.error(f"No valid data for '{ticker}'. Please check the symbol.")
        st.stop()

    df = generate_signals(df)
    df = backtest(df)

    # Live signal
    last = df.iloc[-1]
    label = {
        1: "📈 Bullish crossover → BUY",
        0: "⏸️ No crossover → HOLD",
       -1: "📉 Bearish crossover → SELL"
    }[int(last.signal)]
    st.success(f"**{ticker}**: {label}")

    # Performance summary
    bh_pct    = (df["cum_bh"].iloc[-1] - 1) * 100
    strat_pct = (df["cum_strat"].iloc[-1] - 1) * 100
    st.markdown(f"**Buy & Hold Return:** {bh_pct:.2f}%   |   **Strategy Return:** {strat_pct:.2f}%")

    # Plots: Price+MA10 and Cumulative Performance
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(df.index, df["Close"], label="Close", color="blue")
    ax1.plot(df.index, df["MA10"],  label="MA10", linestyle="--", color="orange")
    ax1.set_title("Price & 10-Day MA")
    ax1.legend()

    ax2.plot(df.index, df["cum_bh"],    label="Buy & Hold", linestyle=":", color="gray")
    ax2.plot(df.index, df["cum_strat"], label="MA Strategy", color="green")
    ax2.set_title("Cumulative Performance")
    ax2.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
