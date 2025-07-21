import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def get_signals(ticker):
    df = yf.download(ticker, period="3mo", interval="1d")
    
    if df.empty or len(df) < 15:
        return None, "⚠️ Not enough data to compute signals."

    df["MA_10"] = df["Close"].rolling(window=10).mean()

    # RSI Calculation
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # Signal Logic
    close_yesterday = df["Close"].iloc[-2]
    close_today = df["Close"].iloc[-1]
    ma_yesterday = df["MA_10"].iloc[-2]
    ma_today = df["MA_10"].iloc[-1]
    rsi_today = df["RSI"].iloc[-1]

    signal = "📉 No crossover"

    if close_yesterday < ma_yesterday and close_today > ma_today:
        signal = "🔼 Bullish crossover"
    elif close_yesterday > ma_yesterday and close_today < ma_today:
        signal = "🔽 Bearish crossover"
    elif rsi_today < 30:
        signal = "🟢 RSI Oversold → Consider Buy"
    elif rsi_today > 70:
        signal = "🔴 RSI Overbought → Consider Sell"

    return df, signal

def plot_chart(df):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df["Close"], label="Close Price", color="blue")
    ax.plot(df.index, df["MA_10"], label="10-day MA", linestyle="--", color="orange")
    ax.set_title("Price & Moving Average")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()

    st.pyplot(fig)

    st.line_chart(df["RSI"], use_container_width=True)

def main():
    st.set_page_config(page_title="QuantaraX — Smart Signal Engine", layout="centered")
    st.title("🚀 QuantaraX — Smart Signal Engine")

    st.subheader("🔍 Generate Today's Signals")
    ticker = st.text_input("Enter a stock ticker (e.g., AAPL)", "AAPL")

    if st.button("📊 Generate Today's Signals"):
        df, signal = get_signals(ticker)

        if df is None:
            st.error(f"{ticker.upper()}: {signal}")
        else:
            st.markdown(f"**{ticker.upper()}:** {signal}")
            plot_chart(df)

if __name__ == "__main__":
    main()
