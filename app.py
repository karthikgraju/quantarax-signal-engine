import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# ── Streamlit Setup ─────────────────────────────────────────────────────────
st.set_page_config(page_title="QuantaraX — Smart Signal Engine", layout="centered")
st.title("🚀 QuantaraX — Smart Signal Engine")
st.subheader("🔍 Generate Today's Signals & Backtest")

# ── Indicator Computation ────────────────────────────────────────────────────
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["MA10"] = df["Close"].rolling(10).mean()

    delta = df["Close"].diff()
    up   = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ema_up   = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    df["RSI14"] = 100 - (100 / (1 + ema_up / ema_down))

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

    return df

# ── Signal Generation ────────────────────────────────────────────────────────
def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    # 1) Align by integer position
    df = df.reset_index(drop=True)

    # 2) MA crossover: up=1, down=-1, else 0
    ma_up = (
        (df["Close"].shift(1) < df["MA10"].shift(1))
        & (df["Close"] > df["MA10"])
    )
    ma_dn = (
        (df["Close"].shift(1) > df["MA10"].shift(1))
        & (df["Close"] < df["MA10"])
    )
    df["ma_signal"] = ma_up.astype(int) - ma_dn.astype(int)

    # 3) RSI oversold/overbought
    rsi_buy  = df["RSI14"] < 30
    rsi_sell = df["RSI14"] > 70
    df["rsi_signal"] = rsi_buy.astype(int) - rsi_sell.astype(int)

    # 4) MACD crossover
    macd_up = (
        (df["MACD"].shift(1) < df["MACD_Signal"].shift(1))
        & (df["MACD"] > df["MACD_Signal"])
    )
    macd_dn = (
        (df["MACD"].shift(1) > df["MACD_Signal"].shift(1))
        & (df["MACD"] < df["MACD_Signal"])
    )
    df["macd_signal"] = macd_up.astype(int) - macd_dn.astype(int)

    # 5) Composite: Go long if any indicator is positive
    df["composite"] = ((df["ma_signal"] + df["rsi_signal"] + df["macd_signal"]) > 0).astype(int)

    return df

# ── Backtester ───────────────────────────────────────────────────────────────
def backtest(df: pd.DataFrame) -> pd.DataFrame:
    df["return"]       = df["Close"].pct_change().fillna(0)
    df["strat_return"] = df["composite"].shift(1).fillna(0) * df["return"]
    df["cum_bh"]       = (1 + df["return"]).cumprod()
    df["cum_strat"]    = (1 + df["strat_return"]).cumprod()
    return df

# ── Main App ────────────────────────────────────────────────────────────────
ticker = st.text_input("Enter a stock ticker (e.g., AAPL)", "AAPL").upper()

if st.button("📊 Generate Today's Signals & Backtest"):
    df = yf.download(ticker, period="6mo", progress=False)

    if df.empty or "Close" not in df.columns:
        st.error("❌ No valid price data for this ticker.")
    else:
        # 1) Compute indicators
        df = compute_indicators(df)

        # 2) Filter out any rows where indicators are NaN
        mask = (
            df["MA10"].notna()
            & df["RSI14"].notna()
            & df["MACD"].notna()
            & df["MACD_Signal"].notna()
        )
        df = df.loc[mask]

        if len(df) < 2:
            st.error("⚠️ Not enough data after filtering indicators.")
        else:
            # 3) Generate signals & backtest
            df = generate_signals(df)
            df = backtest(df)

            # 4) Display latest signals
            last = df.iloc[-1]
            ma_lbl   = {1: "📈", 0: "⏸️", -1: "📉"}[int(last["ma_signal"])]
            rsi_lbl  = f"{last['RSI14']:.1f}"
            macd_lbl = {1: "📈", 0: "⏸️", -1: "📉"}[int(last["macd_signal"])]
            reco     = "🟢 BUY" if last["composite"] == 1 else "🔴 SELL"

            st.success(f"{ticker}: MA→{ma_lbl}   RSI→{rsi_lbl}   MACD→{macd_lbl}")
            st.info(f"Recommendation: {reco}")

            # 5) Backtest performance
            bh_ret    = 100 * (df["cum_bh"].iloc[-1] - 1)
            strat_ret = 100 * (df["cum_strat"].iloc[-1] - 1)
            st.markdown(f"**Buy & Hold Return:** {bh_ret:.2f}%   |   **Strategy Return:** {strat_ret:.2f}%")

            # 6) Plot panels
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

            ax1.plot(df.index, df["Close"], label="Close", color="blue")
            ax1.plot(df.index, df["MA10"], label="MA10", linestyle="--", color="orange")
            ax1.set_title("Price & 10-Day MA"); ax1.legend()

            ax2.plot(df.index, df["RSI14"], label="RSI14", color="purple")
            ax2.axhline(70, color="red", linestyle="--")
            ax2.axhline(30, color="green", linestyle="--")
            ax2.set_title("RSI (14)"); ax2.legend()

            ax3.plot(df.index, df["MACD"], label="MACD", color="black")
            ax3.plot(df.index, df["MACD_Signal"], label="Signal", color="magenta", linestyle="--")
            ax3.bar(df.index, df["MACD_Hist"], label="Hist", color="gray", alpha=0.5)
            ax3.set_title("MACD (12,26,9)"); ax3.legend()

            ax4.plot(df.index, df["cum_bh"], label="Buy & Hold", linestyle=":", color="gray")
            ax4.plot(df.index, df["cum_strat"], label="Strategy", color="green")
            ax4.set_title("Cumulative Performance"); ax4.legend()

            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
