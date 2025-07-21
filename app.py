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

hf_api_key = st.secrets.get("HUGGINGFACE_API_KEY")

def get_signals(ticker):
    try:
        df = yf.download(ticker, period="1mo")
        if df.empty or df.shape[0] < 11:
            return df, "⚠️ Not enough data to generate signals."

        df["MA_10"] = df["Close"].rolling(window=10).mean()

        latest_close = df["Close"].iloc[-1]
        prev_close = df["Close"].iloc[-2]
        latest_ma = df["MA_10"].iloc[-1]
        prev_ma = df["MA_10"].iloc[-2]

        signal = ""
        if latest_close > latest_ma and prev_close < prev_ma:
            signal = "✅ Bullish crossover"
        elif latest_close < latest_ma and prev_close > prev_ma:
            signal = "❌ Bearish crossover"
        else:
            signal = "No crossover"

        return df, signal

    except Exception as e:
        return pd.DataFrame(), f"❌ Error retrieving signals: {str(e)}"

def get_llm_insight(signal_text):
    if not hf_api_key:
        return "Hugging Face API key not found in secrets."

    headers = {
        "Authorization": f"Bearer {hf_api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "inputs": f"Explain this trading signal: {signal_text}"
    }

    try:
        response = requests.post(
            "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1",
            headers=headers,
            data=json.dumps(payload)
        )

        if response.status_code != 200:
            return f"LLM Error: {response.status_code} - {response.text}"

        output = response.json()
        return output[0]["generated_text"]

    except Exception as e:
        return f"LLM Error: {str(e)}"

if generate:
    df, signal = get_signals(ticker)
    st.markdown(f"### {ticker}: {signal}")

    if not df.empty and "MA_10" in df.columns:
        fig, ax = plt.subplots()
        ax.plot(df.index, df["Close"], label="Close", color="blue")
        ax.plot(df.index, df["MA_10"], label="MA 10", color="orange")
        ax.set_title(f"{ticker} Price & MA 10")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)

        st.markdown("**LLM Insight:**")
        with st.spinner("Thinking..."):
            insight = get_llm_insight(signal)
            st.markdown(insight)
    else:
        st.warning("Chart cannot be displayed due to insufficient data.")
