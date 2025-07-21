import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# — Streamlit config
st.set_page_config(page_title="QuantaraX — Smart Signal Engine", layout="centered")
st.title("🚀 QuantaraX — Smart Signal Engine")
st.subheader("🔍 Generate Today's Signals")

# — Core signal logic
def get_signals(ticker: str):
    # 1) Download last 3 months daily data
    df = yf.download(ticker, period="3mo", progress=False)

    # 2) Must have a Close column
    if df.empty or "Close" not in df.columns:
        return None, {"error": "❌ No valid 'Close' price data for this ticker."}

    # 3) Compute 10-day moving average
    df["MA"] = df["Close"].rolling(window=10).mean()

    # 4) Filter out rows where MA is NaN
    df_valid = df[df["MA"].notna()].copy()

    # 5) Need at least two rows to compare today vs yesterday
    if df_valid.shape[0] < 2:
        return None, {"error": "⚠️ Not enough data after MA filter to compute signal."}

    # 6) Extract exact floats
    try:
        close_yest = float(df_valid["Close"].iat[-2])
        close_today = float(df_valid["Close"].iat[-1])
        ma_yest = float(df_valid["MA"].iat[-2])
        ma_today = float(df_valid["MA"].iat[-1])
    except Exception as e:
        return None, {"error": f"❌ Data extraction error: {e}"}

    # 7) Build your crossover signal
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

# — UI
ticker = st.text_input("Enter a stock ticker (e.g., AAPL)", value="AAPL").upper()

if st.button("📊 Generate Today's Signals"):
    df, sig = get_signals(ticker)

    if df is None:
        st.error(f"{ticker}: {sig['error']}")
    else:
        st.success(f"{ticker}: {sig['crossover']}")
        st.info(f"Suggestion: {sig['recommendation']}")

        # Plot price & MA
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(df.index, df["Close"], label="Close", color="blue")
        ax.plot(df.index, df["MA"], label="10-day MA", color="orange", linestyle="--")
        ax.set_title("Price vs 10-Day Moving Average")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)
