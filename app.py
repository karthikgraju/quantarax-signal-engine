import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="QuantaraX — Smart Signal Engine", layout="centered")

st.title("🚀 QuantaraX — Smart Signal Engine")
st.subheader("🔍 Generate Today's Signals")

# --- Signal logic ---
def get_signals(ticker: str):
    # 1) Download 3 months of daily data
    df = yf.download(ticker, period="3mo", progress=False)

    # 2) Validate 'Close' exists
    if df.empty or "Close" not in df.columns:
        return df, {"error": "❌ No 'Close' price data available."}

    # 3) Compute 10-day MA
    df["MA"] = df["Close"].rolling(window=10).mean()

    # 4) Filter out rows where MA is NaN
    df_valid = df[df["MA"].notna()]

    # 5) Need at least 2 rows to compare yesterday/today
    if df_valid.shape[0] < 2:
        return df, {"error": "⚠️ Not enough data (after MA) to compute signal."}

    # 6) Grab last two points
    close_yest = df_valid["Close"].iloc[-2]
    close_today = df_valid["Close"].iloc[-1]
    ma_yest = df_valid["MA"].iloc[-2]
    ma_today = df_valid["MA"].iloc[-1]

    # 7) Build signals
    signals = {}
    if (close_yest < ma_yest) and (close_today > ma_today):
        signals["crossover"] = "📈 Bullish crossover"
        signals["recommendation"] = "🟢 BUY"
    elif (close_yest > ma_yest) and (close_today < ma_today):
        signals["crossover"] = "📉 Bearish crossover"
        signals["recommendation"] = "🔴 SELL"
    else:
        signals["crossover"] = "⏸️ No crossover"
        signals["recommendation"] = "🟡 HOLD"

    return df_valid, signals

# --- UI ---
ticker = st.text_input("Enter a stock ticker (e.g., AAPL)", value="AAPL")

if st.button("📊 Generate Today's Signals"):
    df, sig = get_signals(ticker.upper())

    if "error" in sig:
        st.error(f"{ticker.upper()}: {sig['error']}")
    else:
        st.success(f"{ticker.upper()}: {sig['crossover']}")
        st.info(f"Suggestion: {sig['recommendation']}")

        # Plot price + MA
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(df.index, df["Close"], label="Close", color="blue")
        ax.plot(df.index, df["MA"], label="10-day MA", linestyle="--", color="orange")
        ax.set_title("Price vs 10-day Moving Average")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)
