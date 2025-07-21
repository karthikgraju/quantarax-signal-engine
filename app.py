import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# ---- Function to get signals ----
def get_signals(ticker):
    df = yf.download(ticker, period="3mo")

    # Ensure we have at least 12 data points
    if df.empty or len(df) < 12:
        raise ValueError("Not enough data to compute signals.")

    # Calculate 10-day moving average
    df["MA_10"] = df["Close"].rolling(window=10).mean()

    # Get last and second-last values
    close_today = df["Close"].iloc[-1]
    close_yesterday = df["Close"].iloc[-2]
    ma_today = df["MA_10"].iloc[-1]
    ma_yesterday = df["MA_10"].iloc[-2]

    # Show debug values for troubleshooting
    st.write("🧪 Debug Values")
    st.write(f"Close Yesterday: {close_yesterday}")
    st.write(f"Close Today: {close_today}")
    st.write(f"MA Yesterday: {ma_yesterday}")
    st.write(f"MA Today: {ma_today}")

    # Check for NaN or Series ambiguity
    if any(pd.isna([close_today, close_yesterday, ma_today, ma_yesterday])):
        raise ValueError("Not enough valid data points to compute signal.")

    # Cast to float for safety
    close_today = float(close_today)
    close_yesterday = float(close_yesterday)
    ma_today = float(ma_today)
    ma_yesterday = float(ma_yesterday)

    # Signal logic
    if (close_yesterday < ma_yesterday) and (close_today > ma_today):
        signal = "✅ Bullish crossover"
    elif (close_yesterday > ma_yesterday) and (close_today < ma_today):
        signal = "❌ Bearish crossover"
    else:
        signal = "➖ No crossover"

    return df, signal

# ---- Chart Plotter ----
def plot_chart(df):
    fig, ax = plt.subplots()
    ax.plot(df.index, df["Close"], label="Close Price", color='blue')
    ax.plot(df.index, df["MA_10"], label="10-day MA", color='orange', linestyle='--')
    ax.set_title("Price & Moving Average")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

# ---- Streamlit App ----
st.set_page_config(page_title="QuantaraX — Smart Signal Engine", page_icon="🚀")
st.title("🚀 QuantaraX — Smart Signal Engine")
st.subheader("🔍 Generate Today's Signals")

ticker = st.text_input("Enter a stock ticker (e.g., AAPL)", value="AAPL")

if st.button("📊 Generate Today's Signals"):
    try:
        df, signal = get_signals(ticker)
        st.markdown(f"**{ticker.upper()}:** {signal}")
        plot_chart(df)
    except Exception as e:
        st.error(f"{ticker.upper()}: ⚠️ Error retrieving signals: {e}")
        st.warning("Chart cannot be displayed due to insufficient data.")
