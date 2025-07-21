import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import requests
import json
import pandas as pd

st.set_page_config(page_title="QuantaraX — Smart Signal Engine")

st.title("🚀 QuantaraX — Smart Signal Engine")

st.markdown("### 🔍 Generate Today's Signals")
ticker = st.text_input("Enter a stock ticker (e.g., AAPL)", value="AAPL")
generate = st.button("📊 Generate Today's Signals")

# Load Hugging Face API key from Streamlit secrets
hf_api_key = st.secrets.get("HUGGINGFACE_API_KEY")

def get_signals(ticker):
    df = yf.download(ticker, period="1mo")
    df["MA_10"] = df["Close"].rolling(window=10).mean()

    if len(df) < 11:
        return df, "Not enough data for signal."

    signal = ""
    if df["Close"].iloc[-1] > df["MA_10"].iloc[-1] and df["Close"].iloc[-2] < df["MA_10"].iloc[-2]:
        signal = "✅ Bullish crossover"
    elif df["Close"].iloc[-1] < df["MA_10"].iloc[-1] and df["Close"].iloc[-2] > df["MA_10"].iloc[-2]:
        signal = "❌ Bearish crossover"
    else:
        signal = "No crossover"

    return df, signal

def get_llm_insight(signal_text):
    if not hf_api_key:
        return "Hugging Face API key not set."

    headers = {
        "Authorization": f"Bearer {hf_api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "inputs": f"Explain the trading signal: {signal_text}"
    }

    response = requests.post(
        "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1",
        headers=headers,
        data=json.dumps(payload)
    )

    if response.status_code != 200:
        return f"LLM Error: {response.status_code} - {response.text}"

    try:
        output = response.json()
        return output[0]["generated_text"]
    except Exception as e:
        return f"LLM Error: {str(e)}"

if generate:
    df, signal = get_signals(ticker)

    st.markdown(f"### {ticker}: {signal}")

    if not df.empty and "MA_10" in df.columns:
        fig, ax = plt.subplots()
        ax.plot(df.index, df["Close"], label=ticker, color="blue")
        ax.plot(df.index, df["MA_10"], label="MA 10", color="orange")
        ax.set_title(f"{ticker} Price & 10-Day Moving Average")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)

    st.markdown("**LLM Insight:**")
    with st.spinner("Asking the model..."):
        explanation = get_llm_insight(signal)
        st.markdown(explanation)
