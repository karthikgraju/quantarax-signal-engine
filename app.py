import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# ─ Streamlit Setup ─────────────────────────────────────────────────────────
st.set_page_config(page_title="QuantaraX Signal Engine", layout="centered")
st.title("🚀 QuantaraX — Smart Signal Engine")
st.subheader("🔍 10-Day MA Crossover Signals & Backtest")

# ─ Load & Prepare Data ─────────────────────────────────────────────────────
@st.cache_data
def load_prepared_df(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="6mo", progress=False)
    if df.empty or "Close" not in df.columns:
        return pd.DataFrame()
    # Compute 10-day MA and drop the NaN rows
    df["MA10"] = df["Close"].rolling(10).mean()
    df = df[df["MA10"].notna()].copy()
    return df

ticker = st.text_input("Ticker (e.g. AAPL)", "AAPL").upper()

if st.button("▶️ Run Signals & Backtest"):
    df = load_prepared_df(ticker)
    if df.empty:
        st.error(f"No data for '{ticker}'. Check symbol and try again.")
        st.stop()

    # ─ Generate MA crossover signal via pure-Python loop ────────────────
    # We’ll iterate over rows in order, compare prev vs curr, build a list of signals.
    rows = list(df.itertuples())      # namedtuples: (Index, Open, High, Low, Close, Adj Close, Volume, MA10)
    signals = [0]                     # first day has no signal

    for prev, curr in zip(rows, rows[1:]):
        if prev.Close < prev.MA10 and curr.Close > curr.MA10:
            signals.append(1)
        elif prev.Close > prev.MA10 and curr.Close < curr.MA10:
            signals.append(-1)
        else:
            signals.append(0)

    df["signal"] = signals

    # ─ Backtest ──────────────────────────────────────────────────────
    df["return"]    = df["Close"].pct_change().fillna(0)
    # long 1 share if yesterday’s signal==1, else 0
    df["position"]  = df["signal"].shift(1).fillna(0).clip(lower=0)
    df["strat_ret"] = df["position"] * df["return"]
    df["cum_bh"]    = (1 + df["return"]).cumprod()
    df["cum_strat"] = (1 + df["strat_ret"]).cumprod()

    # ─ Live Signal Display ──────────────────────────────────────────
    last = df.iloc[-1]
    label = {
        1: "📈 Bullish crossover → BUY",
        0: "⏸️ No crossover → HOLD",
       -1: "📉 Bearish crossover → SELL"
    }[int(last.signal)]
    st.success(f"**{ticker}**: {label}")

    # ─ Performance Summary ──────────────────────────────────────────
    bh_pct    = (df["cum_bh"].iloc[-1] - 1) * 100
    strat_pct = (df["cum_strat"].iloc[-1] - 1) * 100
    st.markdown(f"**Buy & Hold Return:** {bh_pct:.2f}%   |   **Strategy Return:** {strat_pct:.2f}%")

    # ─ Plots ────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Price & MA10
    ax1.plot(df.index, df["Close"], label="Close", color="blue")
    ax1.plot(df.index, df["MA10"],  label="MA10", linestyle="--", color="orange")
    ax1.set_title("Price & 10-Day Moving Average")
    ax1.legend()

    # Cumulative Performance
    ax2.plot(df.index, df["cum_bh"],    label="Buy & Hold", linestyle=":", color="gray")
    ax2.plot(df.index, df["cum_strat"], label="MA Strategy", color="green")
    ax2.set_title("Cumulative Performance")
    ax2.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
