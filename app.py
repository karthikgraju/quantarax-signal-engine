import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

def get_signals(ticker):
    df = yf.download(ticker, period='3mo')
    df['MA'] = df['Close'].rolling(window=10).mean()

    # Ensure we have enough data
    if df.shape[0] < 11:
        return df, {"error": "Not enough data to compute 10-day MA."}

    close_today = df['Close'].iloc[-1]
    close_yesterday = df['Close'].iloc[-2]
    ma_today = df['MA'].iloc[-1]
    ma_yesterday = df['MA'].iloc[-2]

    # NaN guard
    if pd.isna(close_today) or pd.isna(close_yesterday) or pd.isna(ma_today) or pd.isna(ma_yesterday):
        return df, {"error": "Insufficient data to compute signals."}

    signals = {}
    if close_yesterday < ma_yesterday and close_today > ma_today:
        signals['ma_crossover'] = "📈 Bullish crossover"
        signals['suggestion'] = "✅ Buy"
    elif close_yesterday > ma_yesterday and close_today < ma_today:
        signals['ma_crossover'] = "📉 Bearish crossover"
        signals['suggestion'] = "🚫 Sell"
    else:
        signals['ma_crossover'] = "⏸️ No crossover"
        signals['suggestion'] = "🔍 Hold"

    return df, signals

def plot_chart(df, ticker):
    plt.figure(figsize=(10, 5))
    plt.plot(df['Close'], label=ticker.upper(), color='blue')
    plt.plot(df['MA'], label='10-day MA', linestyle='--', color='orange')
    plt.title('Price & Moving Average')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(plt)

def main():
    st.title("🚀 QuantaraX — Smart Signal Engine")
    st.subheader("🔍 Generate Today's Signals")

    ticker = st.text_input("Enter a stock ticker (e.g., AAPL)")
    if st.button("📈 Generate Today's Signals"):
        if not ticker:
            st.warning("Please enter a valid ticker symbol.")
            return

        df, signals = get_signals(ticker.upper())

        if "error" in signals:
            st.error(f"{ticker.upper()}: ⚠️ {signals['error']}")
            st.warning("Chart cannot be displayed due to data error.")
            return

        st.success(f"{ticker.upper()}: Signal ➔ {signals['ma_crossover']}")
        st.info(f"Suggested Action: {signals['suggestion']}")

        plot_chart(df, ticker)

if __name__ == "__main__":
    main()
