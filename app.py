import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import requests

# Streamlit setup
st.set_page_config(page_title="QuantaraX Signal Engine", layout="centered")
st.title("🚀 QuantaraX — Smart Signal Engine")

# -------------------------------
# 📈 Fetch 30 days of stock data
# -------------------------------
def get_data(ticker):
    end = datetime.datetime.today()
    start = end - datetime.timedelta(days=30)
    df = yf.download(ticker, start=start, end=end)
    return df

# -------------------------------
# 🤖 Generate commentary via Hugging Face
# -------------------------------
def get_llm_commentary(ticker, signal):
    try:
        hf_api_key = st.secrets["HF_API_KEY"]
        headers = {
            "Authorization": f"Bearer {hf_api_key}"
        }

        prompt = f"Act like a financial analyst. What does it mean when {ticker} shows the signal: '{signal}'?"

        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": 0.7,
                "max_new_tokens": 150
            }
        }

        response = requests.post(
            "https://api-inference.huggingface.co/models/google/flan-t5-xl",
            headers=headers,
            json=payload
        )

        if response.status_code != 200:
            return f"LLM Error: {response.status_code} - {response.text}"

        result = response.json()
        return result[0]['generated_text']
    except Exception as e:
        return f"LLM Error: {str(e)}"

# -------------------------------
# 📊 Analyze a ticker
# -------------------------------
def analyze_ticker(ticker):
    df = get_data(ticker)
    if df.empty or len(df) < 10:
        return {"ticker": ticker, "insight": "⚠️ Not enough data", "chart": None, "commentary": None}

    df["MA_10"] = df["Close"].rolling(window=10).mean()
    last_close = df["Close"].iloc[-1]
    last_ma = df["MA_10"].iloc[-1]

    if pd.isna(last_ma):
        signal = "⚠️ MA data not ready"
    elif last_close > last_ma:
        signal = "✅ Bullish crossover"
    else:
        signal = "📉 Bearish or Neutral"

    fig, ax = plt.subplots()
    df["Close"].plot(ax=ax, label=ticker, color="blue")
    df["MA_10"].plot(ax=ax, label="MA 10", color="orange")
    ax.set_title(f"{ticker} Price & 10-Day Moving Average")
    ax.legend()

    commentary = get_llm_commentary(ticker, signal)

    return {"ticker": ticker, "insight": signal, "chart": fig, "commentary": commentary}

# -------------------------------
# 🔍 Run on a group of tickers
# -------------------------------
def get_top_signals():
    tickers = ["AAPL", "MSFT", "TSLA", "SPY", "QQQ"]
    return [analyze_ticker(t) for t in tickers]

# -------------------------------
# 🖱️ Streamlit UI
# -------------------------------
if st.button("🔍 Generate Today's Signals"):
    signals = get_top_signals()
    for sig in signals:
        st.subheader(f"{sig['ticker']}: {sig['insight']}")
        if sig["chart"]:
            st.pyplot(sig["chart"])
        if sig["commentary"]:
            st.markdown(f"**LLM Insight:** {sig['commentary']}")
