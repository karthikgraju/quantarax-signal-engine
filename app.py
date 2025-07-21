import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="QuantaraX Signal Engine", layout="centered")

# Title
st.markdown("🚀 **QuantaraX — Smart Signal Engine**")
st.markdown("🔍 **Generate Today's Signals**")

# Input
ticker = st.text_input("Enter a stock ticker (e.g., AAPL)", "AAPL")

# Button
if st.button("📊 Generate Today's Signals"):
    def get_signals(ticker):
        df = yf.download(ticker, period="3mo")

        if df.empty or 'Close' not in df.columns:
            return df, {"error": "❌ No valid 'Close' price data."}

        # Calculate 10-day moving average
        df['MA'] = df['Close'].rolling(window=10).mean()

        # Ensure both columns are present
        required_cols = ['Close', 'MA']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return df, {"error": f"❌ Missing required columns: {missing_cols}"}

        # Drop rows with NA values in these columns
        df_valid = df.dropna(subset=required_cols)

        if df_valid.empty or len(df_valid) < 2:
            return df, {"error": "⚠️ Not enough valid data to calculate signal."}

        # Extract the relevant data points
        close_today = df_valid['Close'].iloc[-1]
        close_yesterday = df_valid['Close'].iloc[-2]
        ma_today = df_valid['MA'].iloc[-1]
        ma_yesterday = df_valid['MA'].iloc[-2]

        # Signal Logic
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

        return df_valid, signals

    # Call logic
    df, signal = get_signals(ticker.upper())

    # Handle errors
    if "error" in signal:
        st.error(f"{ticker.upper()}: {signal['error']}")
    else:
        st.success(f"{ticker.upper()}: Signal → {signal['ma_crossover']}")
        st.info(signal['recommendation'])

        # Plot
        fig, ax = plt.subplots()
        ax.plot(df.index, df['Close'], label=f"{ticker.upper()}", color='blue')
        ax.plot(df.index, df['MA'], label="10-day MA", color='orange', linestyle='--')
        ax.set_title("Price & Moving Average")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)
