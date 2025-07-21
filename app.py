import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="QuantaraX — Smart Signal Engine", layout="centered")

st.title("🚀 QuantaraX — Smart Signal Engine")
st.subheader("🔍 Generate Today's Signals")

ticker = st.text_input("Enter a stock ticker (e.g., AAPL)", "AAPL")

if st.button("📊 Generate Today's Signals"):

    @st.cache_data
    def get_signals(ticker: str):
        # 1) Download
        df = yf.download(ticker, period="3mo", progress=False)

        # 2) Must have a 'Close' column
        if df.empty or "Close" not in df.columns:
            return None, {"error": "❌ No valid 'Close' price data for this ticker."}

        # 3) Compute MA
        df["MA"] = df["Close"].rolling(window=10).mean()

        # 4) Now ensure 'MA' was created
        if "MA" not in df.columns:
            return df, {"error": "❌ Moving Average (MA) column missing. Check data."}

        # 5) Drop any rows where either is NaN
        df_valid = df.dropna(subset=["Close", "MA"])
        if df_valid.shape[0] < 2:
            return df, {"error": "⚠️ Not enough valid rows after filtering."}

        # 6) Extract the last two points
        close_yest = df_valid["Close"].iloc[-2]
        close_today = df_valid["Close"].iloc[-1]
        ma_yest = df_valid["MA"].iloc[-2]
        ma_today = df_valid["MA"].iloc[-1]

        # 7) Build signals
        signals = {}
        if (close_yest < ma_yest) and (close_today > ma_today):
            signals["ma_crossover"] = "📈 Bullish crossover"
            signals["recommendation"] = "🟢 Suggestion: BUY"
        elif (close_yest > ma_yest) and (close_today < ma_today):
            signals["ma_crossover"] = "📉 Bearish crossover"
            signals["recommendation"] = "🔴 Suggestion: SELL"
        else:
            signals["ma_crossover"] = "⏸️ No crossover"
            signals["recommendation"] = "🟡 Suggestion: HOLD"

        return df_valid, signals

    # call and display
    result = get_signals(ticker.upper())
    df, signals = result

    if df is None:
        st.error(signals["error"])
    elif "error" in signals:
        st.error(f"{ticker.upper()}: {signals['error']}")
    else:
        st.success(f"{ticker.upper()}: {signals['ma_crossover']}")
        st.info(signals["recommendation"])

        # chart
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(df.index, df["Close"], label="Close", color="blue")
        ax.plot(df.index, df["MA"], label="10-day MA", color="orange", linestyle="--")
        ax.set_title("Price & 10-Day Moving Average")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)
