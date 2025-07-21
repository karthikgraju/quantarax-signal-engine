import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ─ Streamlit page config ──────────────────────────────────────────────
st.set_page_config(page_title="QuantaraX Signal Engine", layout="centered")
st.title("🚀 QuantaraX — Smart Signal Engine")
st.subheader("🔍 Generate Today's Signals & Backtest")

# ─ Fetch & compute indicators ──────────────────────────────────────────
def load_and_compute(ticker: str) -> pd.DataFrame:
    # Download 6 months of daily data
    df = yf.download(ticker, period="6mo", progress=False)
    if df.empty or "Close" not in df:
        return pd.DataFrame()

    # Moving average
    df["MA10"] = df["Close"].rolling(10).mean()

    # RSI14
    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    df["RSI14"] = 100 - (100 / (1 + ema_up / ema_down))

    # MACD (12,26) + signal line (9)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

    return df

# ─ Build signals in pure Python lists ──────────────────────────────────
def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    closes    = df["Close"].to_list()
    ma10s     = df["MA10"].to_list()
    rsis      = df["RSI14"].to_list()
    macds     = df["MACD"].to_list()
    macdsigs  = df["MACD_Signal"].to_list()
    n = len(df)

    ma_sig   = [0] * n
    rsi_sig  = [0] * n
    macd_sig = [0] * n
    comp     = [0] * n

    for i in range(1, n):
        # MA crossover
        if closes[i-1] < ma10s[i-1] and closes[i] > ma10s[i]:
            ma_sig[i] = 1
        elif closes[i-1] > ma10s[i-1] and closes[i] < ma10s[i]:
            ma_sig[i] = -1

        # RSI oversold/overbought
        if rsis[i] < 30:
            rsi_sig[i] = 1
        elif rsis[i] > 70:
            rsi_sig[i] = -1

        # MACD crossover
        if macds[i-1] < macdsigs[i-1] and macds[i] > macdsigs[i]:
            macd_sig[i] = 1
        elif macds[i-1] > macdsigs[i-1] and macds[i] < macdsigs[i]:
            macd_sig[i] = -1

        # Composite: buy if any indicator is positive
        comp[i] = 1 if (ma_sig[i] + rsi_sig[i] + macd_sig[i]) > 0 else 0

    df["ma_signal"]   = ma_sig
    df["rsi_signal"]  = rsi_sig
    df["macd_signal"] = macd_sig
    df["composite"]   = comp
    return df

# ─ Simple backtest ─────────────────────────────────────────────────────
def backtest(df: pd.DataFrame) -> pd.DataFrame:
    df["return"]       = df["Close"].pct_change().fillna(0)
    df["strat_return"] = df["composite"].shift(1).fillna(0) * df["return"]
    df["cum_bh"]       = (1 + df["return"]).cumprod()
    df["cum_strat"]    = (1 + df["strat_return"]).cumprod()
    return df

# ── Main App Logic ─────────────────────────────────────────────────────
ticker = st.text_input("Enter a stock ticker (e.g., AAPL)", "AAPL").upper()

if st.button("📊 Generate Today's Signals & Backtest"):
    df = load_and_compute(ticker)
    if df.empty:
        st.error(f"❌ No data for '{ticker}'.")
    else:
        # Drop rows where any indicator is NaN
        df = df[df[["MA10","RSI14","MACD","MACD_Signal"]].notna().all(axis=1)]
        if len(df) < 2:
            st.error("⚠️ Not enough data after cleaning.")
        else:
            # Build signals & backtest
            df = generate_signals(df)
            df = backtest(df)

            # Show latest signals
            last = df.iloc[-1]
            ma_lbl   = {1:"📈",0:"⏸️",-1:"📉"}[last["ma_signal"]]
            rsi_lbl  = f"{last['RSI14']:.1f}"
            macd_lbl = {1:"📈",0:"⏸️",-1:"📉"}[last["macd_signal"]]
            reco     = "🟢 BUY" if last["composite"]==1 else "🔴 SELL"
            st.success(f"{ticker}: MA→{ma_lbl}   RSI→{rsi_lbl}   MACD→{macd_lbl}")
            st.info(f"Recommendation: {reco}")

            # Performance summary
            bh = 100*(df["cum_bh"].iloc[-1]-1)
            st_rt = 100*(df["cum_strat"].iloc[-1]-1)
            st.markdown(f"**Buy & Hold:** {bh:.2f}% &nbsp;|&nbsp; **Strategy:** {st_rt:.2f}%")

            # Plot four panels
            fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1,figsize=(10,12), sharex=True)
            ax1.plot(df.index, df["Close"], label="Close", color="blue")
            ax1.plot(df.index, df["MA10"],  label="MA10", linestyle="--", color="orange")
            ax1.set_title("Price & MA10");    ax1.legend()

            ax2.plot(df.index, df["RSI14"], label="RSI14", color="purple")
            ax2.axhline(70, color="red", linestyle="--")
            ax2.axhline(30, color="green", linestyle="--")
            ax2.set_title("RSI (14)");        ax2.legend()

            ax3.plot(df.index, df["MACD"],        label="MACD", color="black")
            ax3.plot(df.index, df["MACD_Signal"], label="Signal", color="magenta", linestyle="--")
            ax3.bar(df.index, df["MACD_Hist"],    label="Hist", color="gray", alpha=0.5)
            ax3.set_title("MACD (12,26,9)");      ax3.legend()

            ax4.plot(df.index, df["cum_bh"],    label="Buy & Hold", linestyle=":", color="gray")
            ax4.plot(df.index, df["cum_strat"], label="Strategy",    color="green")
            ax4.set_title("Cumulative Performance"); ax4.legend()

            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
