import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="QuantaraX — Smart Signal Engine", layout="centered")

st.title("🚀 QuantaraX — Smart Signal Engine")
st.subheader("🔍 Generate Today's Signals")

# Ticker input
ticker = st.text_input("Enter a stock ticker (e.g., AAPL)", "AAPL")

def get_data(ticker):
    try:
        df = yf.download(ticker, period="3mo", interval="1d", progress=False)
        if df.empty or len(df) < 12:
            raise ValueError("Not enough data")
        df["MA_10"] = df["Close"].rolling(window=10).mean()
        return df
    except Exception as e:
        return None

def generate_signal(df):
    try:
        if len(df) < 11:
            return "⚠️ Not enough data to compute signals."

        last_close = df["Close"].iloc[-1]
        prev_close = df["Close"].iloc[-2]
        last_ma = df["MA_10"].iloc[-1]
        prev_ma = df["MA_10"].iloc[-2]

        if prev_close < prev_ma and last_close > last_ma:
            return "✅ Bullish crossover"
        elif prev_close > prev_ma and last_close < last_ma:
            return "❌ Bearish crossover"
        else:
            return "➖ No clear signal"
    except Exception as e:
        return f"⚠️ Error computing signal: {e}"

def plot_chart(df):
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["Close"], label="Close", linewidth=2)
    if "MA_10" in df.columns:
        plt.plot(df.index, df["MA_10"], label="10-day MA", linestyle="--")
    plt.title("Price & Moving Average")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(plt)

if st.button("📊 Generate Today's Signals"):
    df = get_data(ticker)
    if df is not None:
        signal = generate_signal(df)
        st.markdown(f"**{ticker.upper()}:** {signal}")
        if len(df) >= 10:
            plot_chart(df)
        else:
            st.warning("Chart cannot be displayed due to insufficient data.")
    else:
        st.error(f"{ticker.upper()}: ❌ Failed to retrieve data from yfinance or not enough data available.")
