import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="QuantaraX — Smart Signal Engine")

st.markdown("🚀 **QuantaraX — Smart Signal Engine**")
st.markdown("🔍 **Generate Today's Signals**")

ticker = st.text_input("Enter a stock ticker (e.g., AAPL)", value="AAPL")

if st.button("📊 Generate Today's Signals"):

    def get_signals(ticker):
        df = yf.download(ticker, period='3mo')

        if df.empty or 'Close' not in df.columns:
            return df, {"error": "❌ No valid price data for this ticker."}

        df['MA'] = df['Close'].rolling(window=10).mean()

        if 'MA' not in df.columns or df['MA'].isnull().all():
            return df, {"error": "⚠️ Moving Average could not be calculated. Not enough data."}

        try:
            df_valid = df.dropna(subset=['Close', 'MA'])
        except KeyError as e:
            return df, {"error": f"❌ Missing required columns: {e}"}

        if len(df_valid) < 2:
            return df, {"error": "⚠️ Not enough data points after cleaning to generate a signal."}

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

        return df_valid, signals

    df, signals = get_signals(ticker.upper())

    if "error" in signals:
        st.error(f"{ticker.upper()}: {signals['error']}")
    else:
        st.success(f"{ticker.upper()}: Signal → {signals['ma_crossover']}")
        st.info(f"{signals['recommendation']}")

        # Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df.index, df['Close'], label='Close Price', color='blue')
        ax.plot(df.index, df['MA'], label='10-day MA', linestyle='--', color='orange')
        ax.set_title("Price & Moving Average")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
