import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import openai

# For local testing, set the key directly — remove this in prod
openai.api_key = "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"
openai.api_base = "https://openrouter.ai/api/v1"

st.title("QuantaraX — Smart Signal Engine")

def get_data(ticker):
    end = datetime.datetime.today()
    start = end - datetime.timedelta(days=30)
    df = yf.download(ticker, start=start, end=end)
    return df

def get_llm_commentary(ticker, signal):
    try:
        response = openai.chat.completions.create(
            model="openai/gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a financial analyst summarizing market signals."},
                {"role": "user", "content": f"Explain what this means for investors: {ticker} has a signal of '{signal}'."}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"LLM Error: {str(e)}"

def analyze_ticker(ticker):
    df = get_data(ticker)
    if df.empty or len(df) < 10:
        return {"ticker": ticker, "insight": "⚠️ Not enough data", "chart": None, "commentary": None}

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

    fig, ax = plt.subplots()
    df["Close"].plot(ax=ax, label="AAPL", color="blue")
    df["MA_10"].plot(ax=ax, label="MA 10", color="orange")
    ax.set_title(f"{ticker} Price & MA10")
    ax.legend()

    commentary = get_llm_commentary(ticker, signal)

    return {"ticker": ticker, "insight": signal, "chart": fig, "commentary": commentary}

def get_top_signals():
    tickers = ["AAPL", "MSFT", "TSLA", "SPY", "QQQ"]
    return [analyze_ticker(t) for t in tickers]

if st.button("Generate Today's Signals"):
    signals = get_top_signals()
    for sig in signals:
        st.subheader(f"{sig['ticker']}: {sig['insight']}")
        if sig["chart"]:
            st.pyplot(sig["chart"])
        if sig["commentary"]:
            st.markdown(f"**LLM Insight:** {sig['commentary']}")
