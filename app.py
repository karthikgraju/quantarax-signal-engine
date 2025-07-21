import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="QuantaraX — Smart Signal Engine")

# Title
st.markdown("🚀 **QuantaraX — Smart Signal Engine**")
st.markdown("🔍 **Generate Today's Signals**")

# Ticker Input
ticker = st.text_input("Enter a stock ticker (e.g., AAPL)", "AAPL")

# Signal generation logic
def get_signals(ticker):
    df = yf.download(ticker, period="90d", interval="1d")

    if df.empty or len(df) < 20:
        return None, "⚠️ Not enough data to compute signals."

    df["MA_10"] = df["Close"].rolling(window=10).mean()

    # RSI calculation
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # Extract scalar values
    try:
        close_yesterday = df["Close"].iloc[-2]
        close_today = df["Close"].iloc[-1]
        ma_yesterday = df["MA_10"].iloc[-2]
        ma_today = df["MA_10"].iloc[-1]
        rsi_today = df["RSI"].iloc[-1]
    except Exception as e:
        return None, f"⚠️ Failed extracting signals: {e}"

    # NaN safety check
    if pd.isna(close_yesterday) or pd.isna(close_today) or pd.isna(ma_yesterday) or pd.isna(ma_today) or pd.isna(rsi_today):
        return None, "⚠️ Not enough valid data to compute signals."

    # Signal logic
    signal = "📉 No crossover"
    if (close_yesterday < ma_yesterday) and (close_today > ma_today):
        signal = "🔼 Bullish crossover"
    elif (close_yesterday > ma_yesterday) and (close_today < ma_today):
        signal = "🔽 Bearish crossover"
    elif rsi_today < 30:
        signal = "🟢 RSI Oversold → Consider Buy"
    elif rsi_today > 70:
        signal = "🔴 RSI Overbought → Consider Sell"

    return df, signal

# Button and execution
if st.button("📊 Generate Today's Signals"):
    df, signal = get_signals(ticker.upper())

    if df is None:
        st.error(f"{ticker.upper()}: {signal}")
        st.warning("Chart cannot be displayed due to insufficient data.")
    else:
        st.success(f"{ticker.upper()}: Signal → {signal}")

        # Plot
        fig, ax = plt.subplots()
        df["Close"].plot(ax=ax, label="Close Price", color="blue")
        df["MA_10"].plot(ax=ax, label="10-day MA", linestyle="--", color="orange")
        ax.set_title("Price & Moving Average")
        ax.set_ylabel("Price")
        ax.set_xlabel("Date")
        ax.legend()
        st.pyplot(fig)
