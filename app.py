import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Function to generate trading signals
def get_signals(ticker):
    df = yf.download(ticker, period='3mo')
    
    # Validate data
    if df.empty or 'Close' not in df.columns:
        return df, {"error": "❌ No valid 'Close' price data for this ticker."}

    # Calculate 10-day moving average
    df['MA'] = df['Close'].rolling(window=10).mean()

    # Drop rows with missing values
    if 'MA' not in df.columns:
        return df, {"error": "❌ Could not calculate moving average (MA)."}

    df_valid = df.dropna(subset=['Close', 'MA'])
    if len(df_valid) < 2:
        return df, {"error": "⚠️ Not enough valid data after cleaning."}

    # Extract relevant prices
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


# Streamlit App Layout
def main():
    st.title("🚀 QuantaraX — Smart Signal Engine")
    st.subheader("🔍 Generate Today's Signals")

    ticker = st.text_input("Enter a stock ticker (e.g., AAPL)", value="AAPL")

    if st.button("📊 Generate Today's Signals"):
        with st.spinner("Analyzing market data..."):
            df, signals = get_signals(ticker.upper())

        if 'error' in signals:
            st.error(f"{ticker.upper()}: {signals['error']}")
        else:
            st.success(f"{ticker.upper()}: Signal → {signals['ma_crossover']}")
            st.info(signals['recommendation'])

            # Plotting
            st.write("### Price & Moving Average")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df.index, df['Close'], label=ticker.upper(), color='blue')
            ax.plot(df.index, df['MA'], label='10-day MA', linestyle='--', color='orange')
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.set_title("Price & Moving Average")
            ax.legend()
            plt.xticks(rotation=45)
            st.pyplot(fig)


if __name__ == "__main__":
    main()
