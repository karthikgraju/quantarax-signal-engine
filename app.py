import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# ── Streamlit page config ────────────────────────────────────────────────
st.set_page_config(page_title="QuantaraX Signal Engine", layout="centered")
st.title("🚀 QuantaraX — Smart Signal Engine")
st.subheader("🔍 10-Day MA Crossover Signals & Backtest")

# ── Load data, compute MA10, filter & reset index ─────────────────────────
@st.cache_data
def load_data(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="6mo", progress=False)
    if df.empty or "Close" not in df.columns:
        return pd.DataFrame()
    df["MA10"] = df["Close"].rolling(window=10).mean()
    df = df[df["MA10"].notna()].copy()
    df.reset_index(drop=True, inplace=True)
    return df

# ── Main app logic ───────────────────────────────────────────────────────
ticker = st.text_input("Ticker (e.g. AAPL)", "AAPL").upper()

if st.button("▶️ Run Signals & Backtest"):
    df = load_data(ticker)
    if df.empty:
        st.error(f"No valid data for '{ticker}'.")
        st.stop()

    # 1) Prepare a 'signal' column (0 = none, 1 = bullish crossover, -1 = bearish)
    df["signal"] = 0
    # locate column positions so we can use .iat
    close_col  = df.columns.get_loc("Close")
    ma_col     = df.columns.get_loc("MA10")
    sig_col    = df.columns.get_loc("signal")

    # 2) Loop by integer position
    for i in range(1, len(df)):
        prev_close = df.iat[i-1, close_col]
        prev_ma    = df.iat[i-1, ma_col]
        curr_close = df.iat[i,   close_col]
        curr_ma    = df.iat[i,   ma_col]

        if prev_close < prev_ma and curr_close > curr_ma:
            df.iat[i, sig_col] = 1
        elif prev_close > prev_ma and curr_close < curr_ma:
            df.iat[i, sig_col] = -1
        # else stays 0

    # 3) Backtest: long (1) if yesterday's signal was bullish
    df["return"]    = df["Close"].pct_change().fillna(0)
    df["position"]  = df["signal"].shift(1).fillna(0).clip(lower=0)
    df["strat_ret"] = df["position"] * df["return"]
    df["cum_bh"]    = (1 + df["return"]).cumprod()
    df["cum_strat"] = (1 + df["strat_ret"]).cumprod()

    # 4) Live signal
    last = df.iloc[-1]
    labels = {
        1: "📈 Bullish crossover → BUY",
        0: "⏸️ No crossover → HOLD",
       -1: "📉 Bearish crossover → SELL"
    }
    st.success(f"**{ticker}**: {labels[int(last.signal)]}")

    # 5) Performance summary
    bh_pct    = (df.cum_bh.iloc[-1] - 1) * 100
    strat_pct = (df.cum_strat.iloc[-1] - 1) * 100
    st.markdown(f"**Buy & Hold Return:** {bh_pct:.2f}%   |   **Strategy Return:** {strat_pct:.2f}%")

    # 6) Plots: Price+MA10 and cumulative P/L
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,8), sharex=True)

    ax1.plot(df["Close"], label="Close", color="blue")
    ax1.plot(df["MA10"],  label="MA10", linestyle="--", color="orange")
    ax1.set_title("Price & 10-Day MA Crossover"); ax1.legend()

    ax2.plot(df["cum_bh"],    label="Buy & Hold", linestyle=":", color="gray")
    ax2.plot(df["cum_strat"], label="MA Strategy", color="green")
    ax2.set_title("Cumulative Performance"); ax2.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
