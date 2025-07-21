import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Signal logic function
def get_signals(ticker):
    df = yf.download(ticker, period="3mo")

    if df.empty or len(df) < 11:
        raise ValueError("Not enough data to compute signals.")

    df["MA_10"] = df["Close"].rolling(window=10).mean()

    if df["MA_10"].isnull().iloc[-1] or df["MA_10"].isnull().iloc[-2]:
        raise ValueError("Insufficient data for moving average.")

    close_today = df["Close"].iloc[-1]
    close_yesterday = df["Close"].iloc[-2]
    ma_today = df["MA_10"].iloc[-1]
    ma_yesterday = df["MA_10"].iloc[-2]

    if (close_yesterday < ma_yesterday) and (close_today > ma_today):
        signal = "✅ Bullish crossover"
    elif (close_yesterday > ma_yesterday) and (close_today < ma_today):
        signal = "❌ Bearish crossover"
    else:
        signal = "➖ No crossover"

    return df, signal

# Plotting function
def plot_chart(df):
    fig, ax = plt.subplots()
    ax.plot(df.index, df["Close"], label="Close", linewidth=2)
    ax.plot(df.index, df["MA_10"], label="10-day MA", linestyle="--")
    ax.set_title("Price & Moving Average")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

# Streamlit UI
st.set_page_config(page_title="QuantaraX — Smart Signal Engine", page_icon="🚀")

st.title("🚀 QuantaraX — Smart Signal Engine")
st.subheader("🔍 Generate Today's Signals")

ticker = st.text_input("Enter a stock ticker (e.g., AAPL)", "AAPL")

if st.button("📊 Generate Today's Signals"):
    try:
        df, signal = get_signals(ticker)
        st.markdown(f"**{ticker.upper()}:** {signal}")
        plot_chart(df)
    except Exception as e:
        st.error(f"{ticker.upper()}: ⚠️ Error retrieving signals: {e}")
        st.warning("Chart cannot be displayed due to insufficient data.")
