import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import requests
import datetime

st.set_page_config(page_title="QuantaraX — Smart Signal Engine", layout="centered")

st.title("🚀 QuantaraX — Smart Signal Engine")

st.markdown("### 🔍 Generate Today's Signals")

# --------------- CONFIGURATION --------------------
TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL"]
HUGGINGFACE_API_KEY = st.secrets["HUGGINGFACE_API_KEY"]
HF_MODEL_URL = "https://api-inference.huggingface.co/models/google/flan-t5-small"

HEADERS = {
    "Authorization": f"Bearer {HUGGINGFACE_API_KEY}"
}
# --------------------------------------------------

def get_signals(ticker):
    today = datetime.date.today()
    df = yf.download(ticker, period="1mo")
    df["MA_10"] = df["Close"].rolling(window=10).mean()
    
    signal = ""
    if df["Close"].iloc[-1] > df["MA_10"].iloc[-1] and df["Close"].iloc[-2] < df["MA_10"].iloc[-2]:
        signal = "Bullish crossover"
    elif df["Close"].iloc[-1] < df["MA_10"].iloc[-1] and df["Close"].iloc[-2] > df["MA_10"].iloc[-2]:
        signal = "Bearish crossover"
    
    return df, signal

def generate_llm_insight(prompt):
    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": 0.7,
            "max_new_tokens": 150
        }
    }
    response = requests.post(HF_MODEL_URL, headers=HEADERS, json=payload)

    # Handle cold start
    import time
    while response.status_code == 202:
        st.info("Model warming up... please wait.")
        time.sleep(3)
        response = requests.post(HF_MODEL_URL, headers=HEADERS, json=payload)

    if response.status_code == 200:
        result = response.json()
        return result[0]["generated_text"] if isinstance(result, list) else result
    else:
        return f"LLM Error: {response.status_code} - {response.text}"

if st.button("🔎 Generate Today's Signals"):
    for ticker in TICKERS:
        df, signal = get_signals(ticker)
        
        if signal:
            st.markdown(f"### {ticker}: ✅ **{signal}**")
        else:
            st.markdown(f"### {ticker}: ❌ No clear signal")
        
        # Chart
        fig, ax = plt.subplots()
        df["Close"].plot(ax=ax, label=ticker, color="blue")
        df["MA_10"].plot(ax=ax, label="MA 10", color="orange")
        ax.set_title(f"{ticker} Price & 10-Day Moving Average")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)

        # LLM Insight
        st.markdown("**LLM Insight:**")
        try:
            prompt = f"Explain this {signal.lower()} signal in simple terms for {ticker}."
            if not signal:
                prompt = f"Explain why there might be no technical signal for {ticker} today."
            llm_result = generate_llm_insight(prompt)
            st.write(llm_result)
        except Exception as e:
            st.error(f"LLM Error: {str(e)}")
