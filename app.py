import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# ——————————————————————————————
# Signal calculation
# ——————————————————————————————
def get_signals(ticker: str):
    # 1) Download 6 months of daily data for MACD lookback
    df = yf.download(ticker, period="6mo", progress=False)
    
    # 2) Must have Close
    if df.empty or "Close" not in df.columns:
        return None, {"error": "❌ No valid 'Close' price data."}

    # 3) Compute 10-day MA
    df["MA10"] = df["Close"].rolling(10).mean()

    # 4) Compute 14-day RSI
    delta = df["Close"].diff()
    up = delta.clip(lower=0); down = -delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    df["RSI14"] = 100 - (100 / (1 + ema_up/ema_down))

    # 5) Compute MACD (12-26) + Signal (9) + Histogram
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # 6) Ensure all columns exist
    required = ["MA10", "RSI14", "MACD", "MACD_Signal"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return df, {"error": f"❌ Missing columns: {missing}"}

    # 7) Filter out rows where any required column is NaN
    mask = df["MA10"].notna() & df["RSI14"].notna() & df["MACD"].notna() & df["MACD_Signal"].notna()
    df_valid = df.loc[mask]
    if len(df_valid) < 2:
        return df, {"error": "⚠️ Not enough data after cleaning."}

    # 8) Grab yesterday & today
    prev, last = df_valid.iloc[-2], df_valid.iloc[-1]
    try:
        c_prev, c_last = float(prev["Close"]), float(last["Close"])
        ma_prev, ma_last = float(prev["MA10"]), float(last["MA10"])
        rsi_last       = float(last["RSI14"])
        macd_prev, macd_last = float(prev["MACD"]), float(last["MACD"])
        sig_prev, sig_last   = float(prev["MACD_Signal"]), float(last["MACD_Signal"])
    except Exception as e:
        return df_valid, {"error": f"❌ Extraction failed: {e}"}

    # 9) Indicator signals
    ma_signal  =  1 if (c_prev < ma_prev and c_last > ma_last) else (-1 if (c_prev > ma_prev and c_last < ma_last) else 0)
    rsi_signal =  1 if rsi_last < 30 else (-1 if rsi_last > 70 else 0)
    macd_signal = 1 if (macd_prev < sig_prev and macd_last > sig_last) else (-1 if (macd_prev > sig_prev and macd_last < sig_last) else 0)

    # 10) Composite recommendation
    score = ma_signal + rsi_signal + macd_signal
    recommendation = "🟢 BUY" if score > 0 else ("🔴 SELL" if score < 0 else "🟡 HOLD")

    signals = {
        "ma":      {1:"📈 Bullish", 0:"⏸️ None", -1:"📉 Bearish"}[ma_signal],
        "rsi":     f"{rsi_last:.1f} ({'Oversold' if rsi_signal==1 else 'Overbought' if rsi_signal==-1 else 'Neutral'})",
        "macd":    {1:"📈 Bullish", 0:"⏸️ None", -1:"📉 Bearish"}[macd_signal],
        "recommendation": recommendation
    }
    return df_valid, signals

# ——————————————————————————————
# Streamlit UI
# ——————————————————————————————
st.set_page_config(page_title="QuantaraX Signal Engine")
st.title("🚀 QuantaraX — Smart Signal Engine")
st.subheader("🔍 Generate Today's Signals")

ticker = st.text_input("Enter a stock ticker (e.g., AAPL)", "AAPL").upper()

if st.button("📊 Generate Today's Signals"):
    df, sig = get_signals(ticker)

    if df is None:
        st.error(sig["error"])
    else:
        # Display signals
        st.success(f"{ticker}: MA→{sig['ma']}   |   RSI→{sig['rsi']}   |   MACD→{sig['macd']}")
        st.info(f"Recommendation: {sig['recommendation']}")

        # Plot panels
        fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(10,8), sharex=True)

        # Price + MA
        ax1.plot(df.index, df["Close"], label="Close", color="blue")
        ax1.plot(df.index, df["MA10"], label="MA10", linestyle="--", color="orange")
        ax1.set_title("Price & 10-Day MA"); ax1.legend()

        # RSI
        ax2.plot(df.index, df["RSI14"], label="RSI(14)", color="purple")
        ax2.axhline(70, color="red", linestyle="--"); ax2.axhline(30, color="green", linestyle="--")
        ax2.set_title("RSI"); ax2.legend()

        # MACD
        ax3.plot(df.index, df["MACD"], label="MACD", color="black")
        ax3.plot(df.index, df["MACD_Signal"], label="Signal", color="magenta", linestyle="--")
        ax3.bar(df.index, df["MACD_Hist"], label="Hist", color="grey", alpha=0.5)
        ax3.set_title("MACD"); ax3.legend()

        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
