import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go

st.set_page_config(page_title="QuantaraX — Signal Engine", page_icon="🚀")

st.title("🚀 QuantaraX — Smart Signal Engine")
st.subheader("🔍 Generate Today's Signals")

ticker = st.text_input("Enter a stock ticker (e.g., AAPL)", "AAPL")

def get_signals(ticker):
    try:
        df = yf.download(ticker, period="1mo")

        if df.empty or len(df) < 12:
            return df, "⚠️ Not enough data to compute signals."

        df["MA_10"] = df["Close"].rolling(window=10).mean()
        df.dropna(inplace=True)

        if "MA_10" not in df.columns or df["MA_10"].isnull().all():
            return df, "⚠️ MA_10 could not be calculated."

        # Make sure we have enough data points
        if len(df) < 2:
            return df, "⚠️ Not enough rows after calculating MA_10."

        # Extract last two values safely
        latest_close = df["Close"].iloc[-1]
        prev_close = df["Close"].iloc[-2]
        latest_ma = df["MA_10"].iloc[-1]
        prev_ma = df["MA_10"].iloc[-2]

        if latest_close > latest_ma and prev_close < prev_ma:
            signal = "✅ Bullish crossover"
        elif latest_close < latest_ma and prev_close > prev_ma:
            signal = "❌ Bearish crossover"
        else:
            signal = "No crossover"

        return df, signal

    except Exception as e:
        return pd.DataFrame(), f"❌ Error retrieving signals: {str(e)}"

if st.button("📊 Generate Today's Signals"):
    df, signal = get_signals(ticker.upper())

    st.markdown(f"### {ticker.upper()}: {signal}")

    if not df.empty and "MA_10" in df.columns and not df["MA_10"].isnull().all():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name='Close', line=dict(color='royalblue')))
        fig.add_trace(go.Scatter(x=df.index, y=df["MA_10"], name='MA_10', line=dict(color='orange')))
        st.plotly_chart(fig)
    else:
        st.warning("Chart cannot be displayed due to insufficient data.")
