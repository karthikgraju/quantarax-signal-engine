import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# ─ Streamlit config ────────────────────────────────────────────────────────
st.set_page_config(page_title="QuantaraX Signal Engine", layout="centered")
st.title("🚀 QuantaraX — Smart Signal Engine")
st.subheader("🔍 10-Day MA Crossover Signals & Backtest")

# ─ Load data & compute MA10 ────────────────────────────────────────────────
@st.cache_data
def load_data(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="6mo", progress=False)
    if df.empty or "Close" not in df.columns:
        return pd.DataFrame()
    df["MA10"] = df["Close"].rolling(window=10).mean()
    # filter out any rows where MA10 is NaN
    return df[df["MA10"].notna()]

# ─ Generate signals (MA crossover) ────────────────────────────────────────
def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    # initialize signal column
    df = df.copy()
    df["signal"] = 0
    # bullish crossover
    bullish = (df["Close"].shift(1) < df["MA10"].shift(1)) & (df["Close"] > df["MA10"])
    # bearish crossover
    bearish = (df["Close"].shift(1) > df["MA10"].shift(1)) & (df["Close"] < df["MA10"])
    df.loc[bullish, "signal"] = 1
    df.loc[bearish, "signal"] = -1
    return df

# ─ Backtest the MA strategy ────────────────────────────────────────────────
def backtest(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["return"]    = df["Close"].pct_change().fillna(0)
    # go long (1) whenever previous signal was bullish
    df["position"]  = df["signal"].shift(1).fillna(0).clip(lower=0)
    df["strat_ret"] = df["position"] * df["return"]
    df["cum_bh"]    = (1 + df["return"]).cumprod()
    df["cum_strat"] = (1 + df["strat_ret"]).cumprod()
    return df

# ─ Main App ───────────────────────────────────────────────────────────────
ticker = st.text_input("Ticker (e.g. AAPL)", "AAPL").upper()

if st.button("▶️ Run Signals & Backtest"):
    df = load_data(ticker)
    if df.empty:
        st.error(f"No data available for '{ticker}'. Check the symbol and try again.")
    else:
        df = generate_signals(df)
        df = backtest(df)

        # Live signal (last row)
        last = df.iloc[-1]
        label = {
            1: "📈 Bullish crossover → BUY",
            0: "⏸️ No crossover → HOLD",
           -1: "📉 Bearish crossover → SELL"
        }[int(last.signal)]
        st.success(f"**{ticker}**: {label}")

        # Performance summary
        bh_pct    = (df.cum_bh.iloc[-1] - 1) * 100
        strat_pct = (df.cum_strat.iloc[-1] - 1) * 100
        st.markdown(f"**Buy & Hold Return:** {bh_pct:.2f}%   |   **Strategy Return:** {strat_pct:.2f}%")

        # Plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Price & MA10
        ax1.plot(df.index, df["Close"], label="Close", color="blue")
        ax1.plot(df.index, df["MA10"], label="MA10", linestyle="--", color="orange")
        ax1.set_title("Price & 10-Day MA Crossover")
        ax1.legend()

        # Cumulative performance
        ax2.plot(df.index, df["cum_bh"],    label="Buy & Hold", linestyle=":", color="gray")
        ax2.plot(df.index, df["cum_strat"], label="MA Strategy", color="green")
        ax2.set_title("Cumulative Performance")
        ax2.legend()

        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
