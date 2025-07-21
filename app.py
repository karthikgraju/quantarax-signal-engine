import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import openai

# ✅ OpenRouter API key from secrets
openai.api_key = st.secrets["OPENROUTER_API_KEY"]
openai.api_base = "https://openrouter.ai/api/v1"

st.set_page_config(page_title="QuantaraX Signal Engine", layout="centered")
st.title("🚀 QuantaraX — Smart Signal Engine")

# 📈 Fetch 30 days of historical price data
def get_data(ticker):
    end = datetime.datetime.today()
    start = end - datetime.timedelta(days=30)
    df = yf.download(ticker, start=start, end=end)
    return df

# 🤖 Use OpenRouter to generate a market commentary
def get_llm_commentary(ticker, signal):
    try:
        response = openai.chat.completions.create(
            model="openrouter/openai/gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a financial analyst summarizing market signals for traders and investors."},
                {"role": "user", "content": f"What does it mean for investors when {ticker} shows the signal: '{signal}'?"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"LLM Error: {str(e)}"

# 📊 Analyze moving average signal and plot
def analyze_ticker(ticker):
    df = get_data(ticker)
    if df.empty or len(df) < 10:
        return {"ticker": ticker, "insight": "⚠️ Not enough data", "chart": None, "commentary": None}

    df["MA_10"] = df["Close"].rolling(window=10).mean()

    try:
        last_close = float(df["Close"].iloc[-1])
        last_ma = float(df["MA_10"].iloc[-1])

        if pd.isna(last_ma):
            signal = "⚠️ MA data not ready"
        elif last_close > last_ma:
            signal = "✅ Bullish crossover"
        else:
            signal = "📉 Bearish or Neutral"
    except Exception as e:
        signal = f"❌ Error: {str(e)}"

    fig, ax = plt.subplots()
    df["Close"].plot(ax=ax, label=ticker, color="blue")
    df["MA_10"].plot(ax=ax, label="MA 10", color="orange")
    ax.set_title(f"{ticker} Price & 10-Day Moving Average")
    ax.legend()

    commentary = get_llm_commentary(ticker, signal)

    return {"ticker": ticker, "insight": signal, "chart": fig, "commentary": commentary}

# 🔍 Ticker list
def get_top_signals():
    tickers = ["AAPL", "MSFT", "TSLA", "SPY", "QQQ"]
    return [analyze_ticker(t) for t in tickers]

# 🖱️ Streamlit UI
if st.button("🔍 Generate Today's Signals"):
    signals = get_top_signals()
    for sig in signals:
        st.subheader(f"{sig['ticker']}: {sig['insight']}")
        if sig["chart"]:
            st.pyplot(sig["chart"])
        if sig["commentary"]:
            st.markdown(f"**LLM Insight:** {sig['commentary']}")
