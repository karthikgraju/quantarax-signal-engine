import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# ─ Streamlit page config ─────────────────────────────────────────────────
st.set_page_config(page_title="QuantaraX Signal Engine", layout="centered")
st.title("🚀 QuantaraX — Smart Signal Engine")
st.subheader("🔍 10-Day MA Crossover Signals & Backtest")

# ─ Load data & compute MA10 ────────────────────────────────────────────────
@st.cache_data
def load_data(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="6mo", progress=False)
    if df.empty or "Close" not in df.columns:
        return pd.DataFrame()
    df["MA10"] = df["Close"].rolling(window=10).mean()
    return df.dropna(subset=["MA10"])

# ─ Generate signals (MA crossover) ────────────────────────────────────────
def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    # 1 = bullish crossover, -1 = bearish crossover, 0 = no signal
    df["signal"] = 0
    up  = (df["Close"].shift(1) < df["MA10"].shift(1)) & (df["Close"] > df["MA10"])
    dn  = (df["Close"].shift(1) > df["MA10"].shift(1)) & (df["Close"] < df["MA10"])
    df.loc[up,  "signal"] = 1
    df.loc[dn,  "signal"] = -1
    return df

# ─ Backtest the MA strategy ────────────────────────────────────────────────
def backtest(df: pd.DataFrame) -> pd.DataFrame:
    df["return"]       = df["Close"].pct_change().fillna(0)
    # We go LONG (1) whenever the signal on the previous day was bullish
    df["position"]     = df["signal"].shift(1).fillna(0).clip(lower=0)
    df["strat_ret"]    = df["position"] * df["return"]
    df["cum_bh"]       = (1 + df["return"]).cumprod()
    df["cum_strat"]    = (1 + df["strat_ret"]).cumprod()
    return df

# ─ UI ─────────────────────────────────────────────────────────────────────
ticker = st.text_input("Ticker (e.g. AAPL)", "AAPL").upper()

if st.button("▶️ Run Signals & Backtest"):
    df = load_data(ticker)
    if df.empty:
        st.error(f"No data for '{ticker}'. Please check the symbol.")
    else:
        df = generate_signals(df)
        df = backtest(df)

        # Live signal (today’s row)
        last = df.iloc[-1]
        live = {1: "📈 Bullish crossover → BUY", 0: "⏸️ No crossover → HOLD", -1: "📉 Bearish crossover → SELL"}[int(last.signal)]
        st.success(f"**{ticker}**: {live}")

        # Performance summary
        bh_ret    = (df.cum_bh.iloc[-1] - 1) * 100
        strat_ret = (df.cum_strat.iloc[-1] - 1) * 100
        st.markdown(f"**Buy & Hold Return:** {bh_ret:.2f}%   |   **Strategy Return:** {strat_ret:.2f}%")

        # Plots
        fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,8), sharex=True)

        # Price & MA10
        ax1.plot(df.index, df["Close"], label="Close", color="blue")
        ax1.plot(df.index, df["MA10"], label="MA10", linestyle="--", color="orange")
        ax1.set_title("Price & 10-Day MA"); ax1.legend()

        # Cumulative returns
        ax2.plot(df.index, df["cum_bh"],    label="Buy & Hold", color="gray", linestyle=":")
        ax2.plot(df.index, df["cum_strat"], label="MA Strategy", color="green")
        ax2.set_title("Cumulative Performance"); ax2.legend()

        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
