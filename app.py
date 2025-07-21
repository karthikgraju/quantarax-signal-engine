import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# —————————————————————————————————————————
#  Core Signal Logic (strictly top-level)
# —————————————————————————————————————————
def get_signals(ticker: str):
    # 1) Download the last 3 months of daily data
    df = yf.download(ticker, period="3mo", progress=False)

    # 2) Must have price data
    if df.empty or "Close" not in df.columns:
        return None, {"error": "❌ No price data (Close) available for this ticker."}

    # 3) Compute the 10-day moving average
    df["MA"] = df["Close"].rolling(window=10).mean()

    # 4) Drop any rows where MA is still NaN
    df_valid = df.dropna(subset=["MA"])

    # 5) Need at least two valid rows
    if df_valid.shape[0] < 2:
        return None, {"error": "⚠️ Insufficient data after MA filter to generate signal."}

    # 6) Grab yesterday’s and today’s full rows
    prev_row = df_valid.iloc[-2]
    last_row = df_valid.iloc[-1]

    # 7) Extract floats
    try:
        close_yest  = float(prev_row["Close"])
        ma_yest     = float(prev_row["MA"])
        close_today = float(last_row["Close"])
        ma_today    = float(last_row["MA"])
    except Exception as e:
        return None, {"error": f"❌ Data extraction failed: {e}"}

    # 8) Generate crossover signal + recommendation
    signals = {}
    if (close_yest < ma_yest) and (close_today > ma_today):
        signals["crossover"]     = "📈 Bullish crossover"
        signals["recommendation"] = "🟢 BUY"
    elif (close_yest > ma_yest) and (close_today < ma_today):
        signals["crossover"]     = "📉 Bearish crossover"
        signals["recommendation"] = "🔴 SELL"
    else:
        signals["crossover"]     = "⏸️ No crossover"
        signals["recommendation"] = "🟡 HOLD"

    return df_valid, signals


# —————————————————————————————————————————
#  Streamlit UI
# —————————————————————————————————————————
st.set_page_config(page_title="QuantaraX Signal Engine", layout="centered")
st.title("🚀 QuantaraX — Smart Signal Engine")
st.subheader("🔍 Generate Today's Signals")

ticker = st.text_input("Enter a stock ticker (e.g., AAPL)", "AAPL").upper()

if st.button("📊 Generate Today's Signals"):
    df, sig = get_signals(ticker)

    if df is None:
        st.error(f"{ticker}: {sig['error']}")
    else:
        st.success(f"{ticker}: {sig['crossover']}")
        st.info(f"Suggestion: {sig['recommendation']}")

        # Plot Price vs MA
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(df.index, df["Close"], label="Close Price", color="blue")
        ax.plot(df.index, df["MA"],    label="10-day MA",   color="orange", linestyle="--")
        ax.set_title("Price vs 10-Day Moving Average")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)
