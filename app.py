# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt

def get_signals(ticker):
    data = yf.download(ticker, period="6mo")
    if data.empty:
        return None, "No data available."

    data['10_MA'] = data['Close'].rolling(window=10).mean()
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = ema12 - ema26
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # Recent values
    close_today = data['Close'].iloc[-1]
    ma_today = data['10_MA'].iloc[-1]
    rsi_today = data['RSI'].iloc[-1]
    macd_today = data['MACD'].iloc[-1]
    signal_today = data['Signal_Line'].iloc[-1]

    signal_summary = []
    if close_today > ma_today:
        signal_summary.append("Price above 10-MA: Bullish")
    else:
        signal_summary.append("Price below 10-MA: Bearish")

    if rsi_today < 30:
        signal_summary.append("RSI < 30: Oversold")
    elif rsi_today > 70:
        signal_summary.append("RSI > 70: Overbought")
    else:
        signal_summary.append("RSI Neutral")

    if macd_today > signal_today:
        signal_summary.append("MACD Bullish crossover")
    elif macd_today < signal_today:
        signal_summary.append("MACD Bearish crossover")
    else:
        signal_summary.append("MACD Neutral")

    if (close_today > ma_today) and (macd_today > signal_today) and (rsi_today < 70):
        decision = "Buy"
    elif (close_today < ma_today) and (macd_today < signal_today) and (rsi_today > 30):
        decision = "Sell"
    else:
        decision = "Hold"

    signal_summary.append(f"\nRecommendation → {decision}")

    return data, signal_summary

def main():
    st.set_page_config(page_title="QuantaraX — Smart Signal Engine", layout="centered")
    st.title("🚀 QuantaraX — Smart Signal Engine")

    st.subheader("🔍 Generate Today's Signals")
    ticker = st.text_input("Enter a stock ticker (e.g., AAPL)", "AAPL")

    if st.button("📊 Generate Today's Signals"):
        df, signals = get_signals(ticker.upper())

        if df is None:
            st.error(f"{ticker.upper()}: ⚠️ {signals}")
            return

        for s in signals:
            st.success(f"{ticker.upper()}: {s}")

        # Plot chart
        fig, ax = plt.subplots()
        ax.plot(df.index, df['Close'], label=f'{ticker.upper()} Close', color='blue')
        ax.plot(df.index, df['10_MA'], label='10-day MA', color='orange', linestyle='--')
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.set_title("Price & Moving Average")
        ax.legend()
        st.pyplot(fig)

if __name__ == "__main__":
    main()
