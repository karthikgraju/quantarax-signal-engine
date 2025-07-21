import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="QuantaraX — Smart Signal Engine", layout="centered")
st.title("🚀 QuantaraX — Smart Signal Engine")
st.subheader("🔍 Generate Today's Signals")

def get_signals(ticker: str):
    # Fetch 6 months to give MACD enough history
    df = yf.download(ticker, period="6mo", progress=False)

    if df.empty or "Close" not in df:
        return None, {"error": "❌ No valid price data."}

    # 10-day MA
    df["MA10"] = df["Close"].rolling(10).mean()

    # RSI 14
    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    df["RSI14"] = 100 - (100 / (1 + rs))

    # MACD (12-26) + signal (9)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # Drop initial NaNs
    df = df.dropna(subset=["MA10", "RSI14", "MACD", "MACD_Signal"])

    if len(df) < 2:
        return None, {"error": "⚠️ Not enough data after cleaning."}

    # Grab last two days
    prev, last = df.iloc[-2], df.iloc[-1]
    c_prev, c_last = float(prev["Close"]), float(last["Close"])
    ma_prev, ma_last = float(prev["MA10"]), float(last["MA10"])
    rsi = float(last["RSI14"])
    macd, macd_sig = float(prev["MACD"]), float(last["MACD"])
    sig_line = float(last["MACD_Signal"])

    # Determine signals
    # MA crossover
    if ma_prev < c_prev and c_last > ma_last:
        ma_signal = 1
    elif ma_prev > c_prev and c_last < ma_last:
        ma_signal = -1
    else:
        ma_signal = 0

    # RSI: oversold (<30)=+1, overbought (>70)=-1
    if rsi < 30:
        rsi_signal = 1
    elif rsi > 70:
        rsi_signal = -1
    else:
        rsi_signal = 0

    # MACD crossover: prev MACD < prev signal & MACD > signal => +1, opposite => -1
    if macd < prev["MACD_Signal"] and macd_sig > sig_line:
        macd_signal = 1
    elif macd > prev["MACD_Signal"] and macd_sig < sig_line:
        macd_signal = -1
    else:
        macd_signal = 0

    # Composite score
    score = ma_signal + rsi_signal + macd_signal

    if score > 0:
        reco = "🟢 BUY"
    elif score < 0:
        reco = "🔴 SELL"
    else:
        reco = "🟡 HOLD"

    signals = {
        "ma_crossover": {1: "📈 Bullish", -1: "📉 Bearish", 0: "⏸️ None"}[ma_signal],
        "rsi": f"{rsi:.1f} ({ '🟢 Oversold' if rsi_signal==1 else '🔴 Overbought' if rsi_signal==-1 else '⚪ Neutral'})",
        "macd_crossover": {1:"📈 Bullish", -1:"📉 Bearish",0:"⏸️ None"}[macd_signal],
        "recommendation": reco
    }
    return df, signals

ticker = st.text_input("Enter a stock ticker (e.g., AAPL)", "AAPL").upper()

if st.button("📊 Generate Today's Signals"):
    df, sig = get_signals(ticker)
    if df is None:
        st.error(sig["error"])
    else:
        st.success(f"{ticker}: {sig['ma_crossover']} MA    |    {sig['macd_crossover']} MACD")
        st.info(f"RSI14: {sig['rsi']}    →    Recommendation: {sig['recommendation']}")

        # Plot
        fig, (ax1, ax2, ax3) = plt.subplots(3,1,figsize=(10,8), sharex=True)

        # Price + MA
        ax1.plot(df.index, df["Close"], label="Close", color="blue")
        ax1.plot(df.index, df["MA10"], label="MA10", color="orange", linestyle="--")
        ax1.set_title("Price & 10-Day MA"); ax1.legend()

        # RSI
        ax2.plot(df.index, df["RSI14"], label="RSI14", color="purple")
        ax2.axhline(70, color="red", linestyle="--"); ax2.axhline(30, color="green", linestyle="--")
        ax2.set_title("RSI (14)"); ax2.legend()

        # MACD
        ax3.plot(df.index, df["MACD"], label="MACD", color="black")
        ax3.plot(df.index, df["MACD_Signal"], label="Signal", color="magenta", linestyle="--")
        ax3.bar(df.index, df["MACD_Hist"], label="Hist", color="grey", alpha=0.5)
        ax3.set_title("MACD"); ax3.legend()

        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
