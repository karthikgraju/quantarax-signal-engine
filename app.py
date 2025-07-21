import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="QuantaraX — Smart Signal Engine", layout="centered")

st.title("🚀 QuantaraX — Smart Signal Engine")
st.subheader("🔍 Generate Today's Signals")

# ——————————————
# Signal logic
# ——————————————
def get_signals(ticker: str):
    # 1) Load price data
    df = yf.download(ticker, period="3mo", progress=False)

    # 2) Must have 'Close'
    if df.empty or "Close" not in df.columns:
        return None, {"error": "❌ No valid 'Close' price data."}

    # 3) Compute 10-day MA
    df["MA"] = df["Close"].rolling(window=10).mean()

    # 4) Filter out rows where MA is NaN
    #    (this never raises KeyError)
    df_valid = df.loc[df["MA"].notna()]

    # 5) Need at least two rows to compare
    if df_valid.shape[0] < 2:
        return None, {"error": "⚠️ Insufficient data after MA filter."}

    # 6) Grab yesterday's & today's rows
    prev = df_valid.iloc[-2]
    last = df_valid.iloc[-1]

    # 7) Extract as floats
    try:
        close_yest  = float(prev["Close"])
        ma_yest     = float(prev["MA"])
        close_today = float(last["Close"])
        ma_today    = float(last["MA"])
    except Exception as e:
        return None, {"error": f"❌ Data extraction failed: {e}"}

    # 8) Build crossover + recommendation
    signals = {}
    if close_yest < ma_yest and close_today > ma_today:
        signals["crossover"]     = "📈 Bullish crossover"
        signals["recommendation"] = "🟢 BUY"
    elif close_yest > ma_yest and close_today < ma_today:
        signals["crossover"]     = "📉 Bearish crossover"
        signals["recommendation"] = "🔴 SELL"
    else:
        signals["crossover"]     = "⏸️ No crossover"
        signals["recommendation"] = "🟡 HOLD"

    return df_valid, signals

# ——————————————
# Streamlit UI
# ——————————————
ticker = st.text_input("Enter a stock ticker (e.g., AAPL)", "AAPL").upper()

if st.button("📊 Generate Today's Signals"):
    df, sig = get_signals(ticker)

    if df is None:
        st.error(f"{ticker}: {sig['error']}")
    else:
        st.success(f"{ticker}: {sig['crossover']}")
        st.info(f"Suggestion: {sig['recommendation']}")

        # Plot price & MA
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(df.index, df["Close"], label="Close Price", color="blue")
        ax.plot(df.index, df["MA"],    label="10-day MA",   color="orange", linestyle="--")
        ax.set_title("Price vs 10-Day Moving Average")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)
