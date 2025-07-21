import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import openai
import os

# Setup for OpenRouter
openai.api_key = os.getenv("OPENROUTER_API_KEY")  # Set in Streamlit secrets
openai.base_url = "https://openrouter.ai/api/v1"

st.title("QuantaraX — Smart Signal Engine")

# -------------------------
# Fetch stock data
# -------------------------
def get_data(ticker):
    end = datetime.datetime.today()
    start = end - datetime.timedelta(days=30)
    df = yf.download(ticker, start=start, end=end)
    return df

# -------------------------
# Use OpenRouter to explain signal
# -------------------------
def get_llm_commentary(ticker, signal):
    try:
        response = openai.chat.completions.create(
            model="openai/gpt-3.5-turbo",  # you can change this to "mistralai/mixtral-8x7b" or others
            messages=[
                {
                    "role": "system",
                    "content": "You are a financial analyst summarizing technical market signals for investors."
                },
                {
                    "role": "user",
                    "content": f"Explain what this means for investors: {ticker} shows a signal of '{signal}'."
                }
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"LLM Error: {str(e)}"

# -------------------------
# Analyze ticker
# -------------------------
def analyze_ticker(ticker):
    df = get_data(ticker)
    if df.empty or len(df) < 10:
        return {"ticker": ticker, "insight": "⚠️ Not enough data", "chart": None, "commentary": None}

    df["MA_10"] = df["Close"].rolling(10).mean()

    try:
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

    # Chart
    fig, ax = plt.subplots()
    df["Close"].plot(ax=ax, label="Close", color="blue")
    df["MA_10"].plot(ax=ax, label="MA 10", color="orange")
    ax.set_title(f"{ticker} Price & MA10")
    ax.legend()

    # Commentary from LLM
    commentary = get_llm_commentary(ticker, signal)

    return {"ticker": ticker, "insight": signal, "chart": fig, "commentary": commentary}

# -------------------------
# Ticker list
# -------------------------
def get_top_signals():
    tickers = ["AAPL", "MSFT", "TSLA", "SPY", "QQQ"]
    return [analyze_ticker(t) for t in tickers]

# -------------------------
# Streamlit UI
# -------------------------
if st.button("Generate Today's Signals"):
    signals = get_top_signals()
    for sig in signals:
        st.subheader(f"{sig['ticker']}: {sig['insight']}")
        if sig["chart"]:
            st.pyplot(sig["chart"])
        if sig["commentary"]:
            st.markdown(f"**LLM Insight:** {sig['commentary']}")
