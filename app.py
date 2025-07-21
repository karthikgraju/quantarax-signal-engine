from openai import OpenAI
import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import os

# Init OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.title("QuantaraX — Smart Signal Engine")

def get_data(ticker):
    end = datetime.datetime.today()
    start = end - datetime.timedelta(days=30)
    df = yf.download(ticker, start=start, end=end)
    return df

def analyze_ticker(ticker):
    df = get_data(ticker)
    if df.empty or len(df) < 10:
        return {"ticker": ticker, "insight": "⚠️ Not enough data", "chart": None}

    df["MA_10"] = df["Close"].rolling(10).mean()

    try:
        last_close = float(df["Close"].iloc[-1])
        last_ma = float(df["MA_10"].iloc[-1])
        if pd.isna(last_ma):
            signal = "⚠️ MA data not ready"
        elif last_close > last_ma:
            signal = "✅ Bullish crossover"
        else:
            signal = "⚠️ Neutral"
    except Exception as e:
        signal = f"❌ Error: {str(e)}"

    # Chart
    fig, ax = plt.subplots()
    df["Close"].plot(ax=ax, label="Close", color="blue")
    df["MA_10"].plot(ax=ax, label="MA 10", color="orange")
    ax.set_title(f"{ticker} Price & MA10")
    ax.legend()

    return {"ticker": ticker, "insight": signal, "chart": fig}

def explain_signal(ticker, insight):
    prompt = f"Explain in one sentence why {ticker} has a {insight.lower()} in stock trading terms using the context of a 10-day moving average."

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"LLM Error: {str(e)}"

def get_top_signals():
    tickers = ["AAPL", "MSFT", "TSLA", "SPY", "QQQ"]
    return [analyze_ticker(t) for t in tickers]

# UI Trigger
if st.button("Generate Today's Signals"):
    signals = get_top_signals()
    for sig in signals:
        st.write(f"**{sig['ticker']}**: {sig['insight']}")
        explanation = explain_signal(sig['ticker'], sig['insight'])
        st.caption(explanation)
        if sig["chart"]:
            st.pyplot(sig["chart"])
