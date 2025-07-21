import streamlit as st
import yfinance as yf
import pandas as pd
import datetime

st.title("QuantaraX — Smart Signal Engine")

def get_data(ticker):
    end = datetime.datetime.today()
    start = end - datetime.timedelta(days=30)
    df = yf.download(ticker, start=start, end=end)
    return df

def analyze_ticker(ticker):
    df = get_data(ticker)
    if df.empty or len(df) < 10:
        return {"ticker": ticker, "insight": "⚠️ Not enough data"}
    df["MA_10"] = df["Close"].rolling(10).mean()
    try:
        # Get last closing price and last MA_10 value
        last_close = df["Close"].iloc[-1]
        last_ma = df["MA_10"].iloc[-1]
        if pd.isna(last_ma):
            signal = "⚠️ MA data not ready"
        elif last_close > last_ma:
            signal = "✅ Bullish crossover"
        else:
            signal = "⚠️ Neutral"
    except Exception as e:
        signal = f"❌ Error: {str(e)}"
    return {"ticker": ticker, "insight": signal}

def get_top_signals():
    tickers = ["AAPL", "MSFT", "TSLA", "SPY", "QQQ"]
    return [analyze_ticker(t) for t in tickers]

if st.button("Generate Today's Signals"):
    signals = get_top_signals()
    for sig in signals:
        st.write(f"**{sig['ticker']}**: {sig['insight']}")
