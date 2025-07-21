import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# ─ Streamlit config ─────────────────────────────────────────────────────────
st.set_page_config(page_title="QuantaraX Signal Engine", layout="centered")
st.title("🚀 QuantaraX — Smart Signal Engine")
st.subheader("🔍 10-Day MA Crossover Signals & Backtest")

def load_data(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="6mo", progress=False)
    if df.empty or "Close" not in df:
        return pd.DataFrame()
    # Compute 10-day MA
    df["MA10"] = df["Close"].rolling(window=10).mean()
    # Filter out rows where MA10 is still NaN
    df = df[df["MA10"].notna()].copy()
    return df

ticker = st.text_input("Ticker (e.g. AAPL)", "AAPL").upper()

if st.button("▶️ Run Signals & Backtest"):
    df = load_data(ticker)
    if df.empty:
        st.error(f"No data available for '{ticker}'. Check the symbol and try again.")
        st.stop()

    # 1) Generate MA crossover signals via a simple Python loop
    df["signal"] = 0
    closes = df["Close"].tolist()
    ma10s  = df["MA10"].tolist()
    dates  = df.index.to_list()

    for i in range(1, len(dates)):
        # bullish crossover
        if closes[i-1] < ma10s[i-1] and closes[i] > ma10s[i]:
            df.at[dates[i], "signal"] = 1
        # bearish crossover
        elif closes[i-1] > ma10s[i-1] and closes[i] < ma10s[i]:
            df.at[dates[i], "signal"] = -1

    # 2) Backtest: go long (position=1) whenever yesterday’s signal == 1
    df["return"]    = df["Close"].pct_change().fillna(0)
    df["position"]  = df["signal"].shift(1).fillna(0).clip(lower=0)
    df["strat_ret"] = df["position"] * df["return"]
    df["cum_bh"]    = (1 + df["return"]).cumprod()
    df["cum_strat"] = (1 + df["strat_ret"]).cumprod()

    # 3) Live signal display
    last = df.iloc[-1]
    label = {
        1: "📈 Bullish crossover → BUY",
        0: "⏸️ No crossover → HOLD",
       -1: "📉 Bearish crossover → SELL"
    }[int(last.signal)]
    st.success(f"**{ticker}**: {label}")

    # 4) Performance summary
    bh_pct    = (df.cum_bh.iloc[-1] - 1) * 100
    strat_pct = (df.cum_strat.iloc[-1] - 1) * 100
    st.markdown(f"**Buy & Hold Return:** {bh_pct:.2f}%   |   **Strategy Return:** {strat_pct:.2f}%")

    # 5) Plots: Price & MA10, then cumulative P/L
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,8), sharex=True)

    ax1.plot(df.index, df["Close"], label="Close", color="blue")
    ax1.plot(df.index, df["MA10"], label="MA10", linestyle="--", color="orange")
    ax1.set_title("Price & 10-Day MA Crossover"); ax1.legend()

    ax2.plot(df.index, df["cum_bh"],    label="Buy & Hold", linestyle=":", color="gray")
    ax2.plot(df.index, df["cum_strat"], label="MA Strategy", color="green")
    ax2.set_title("Cumulative Performance"); ax2.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
