import streamlit as st
import yfinance as yf
import datetime

st.title("QuantaraX — Smart Signal Engine")

def get_data(ticker):
    end = datetime.datetime.today()
    start = end - datetime.timedelta(days=30)
    df = yf.download(ticker, start=start, end=end)
    return df

def analyze_ticker(ticker):
    df = get_data(ticker)
    df["MA_10"] = df["Close"].rolling(10).mean()
    signal = ""
    if df["Close"].iloc[-1] > df["MA_10"].iloc[-1]:
        signal = "Bullish crossover"
    else:
        signal = "Neutral"
    return {"ticker": ticker, "insight": signal}

def get_top_signals():
    tickers = ["AAPL", "MSFT", "TSLA", "SPY", "QQQ"]
    return [analyze_ticker(t) for t in tickers]

if st.button("Generate Today's Signals"):
    signals = get_top_signals()
    for sig in signals:
        st.write(f"**{sig['ticker']}**: {sig['insight']}")
