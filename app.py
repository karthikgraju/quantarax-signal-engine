import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# ========== Signal Logic ==========

def get_signals(ticker):
    df = yf.download(ticker, period='3mo')

    if df.empty or 'Close' not in df.columns:
        return df, {"error": "No price data available for this ticker."}

    # Compute MA before dropping rows
    df['MA'] = df['Close'].rolling(window=10).mean()

    if 'MA' not in df.columns:
        return df, {"error": "MA calculation failed."}

    df_valid = df.dropna(subset=['Close', 'MA'])

    if len(df_valid) < 2:
        return df, {"error": "Not enough clean data to compute signals."}

    close_today = df_valid['Close'].iloc[-1]
    close_yesterday = df_valid['Close'].iloc[-2]
    ma_today = df_valid['MA'].iloc[-1]
    ma_yesterday = df_valid['MA'].iloc[-2]

    signals = {}

    if close_yesterday < ma_yesterday and close_today > ma_today:
        signals['ma_crossover'] = "📈 Bullish crossover"
        signals['recommendation'] = "🟢 Suggestion: BUY"
    elif close_yesterday > ma_yesterday and close_today < ma_today:
        signals['ma_crossover'] = "📉 Bearish crossover"
        signals['recommendation'] = "🔴 Suggestion: SELL"
    else:
        signals['ma_crossover'] = "⏸️ No crossover"
        signals['recommendation'] = "🟡 Suggestion: HOLD"

    return df, signals

# ========== Streamlit UI ==========

st.set_page_config(page_title="QuantaraX Signal Engine", layout="centered")

st.markdown("🚀 **QuantaraX — Smart Signal Engine**")
st.markdown("🔍 **Generate Today's Signals**")

ticker = st.text_input("Enter a stock ticker (e.g., AAPL)", value="AAPL")

if st.button("📊 Generate Today's Signals"):
    with st.spinner(f"Fetching data and generating signal for {ticker.upper()}..."):
        df, signals = get_signals(ticker.upper())

    if "error" in signals:
        st.error(f"{ticker.upper()}: ⚠️ {signals['error']}")
    else:
        st.success(f"{ticker.upper()}: Signal → {signals['ma_crossover']}")
        st.info(signals['recommendation'])

        # Plot chart
        st.subheader("Price & Moving Average")
        fig, ax = plt.subplots()
        ax.plot(df.index, df['Close'], label='Close Price', color='blue')
        ax.plot(df.index, df['MA'], label='10-day MA', color='orange', linestyle='--')
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)
