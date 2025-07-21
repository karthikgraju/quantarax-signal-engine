import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="QuantaraX — Smart Signal Engine", layout="centered")
st.title("🚀 QuantaraX — Smart Signal Engine")
st.subheader("🔍 Generate Today's Signals & Backtest")

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # 10-day MA
    df["MA10"] = df["Close"].rolling(10).mean()

    # 14-day RSI
    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    df["RSI14"] = 100 - (100 / (1 + ema_up / ema_down))

    # MACD (12-26) + signal line (9) + histogram
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

    return df

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    # work on the filtered DataFrame only
    n = len(df)
    close     = df["Close"].values
    ma10      = df["MA10"].values
    rsi14     = df["RSI14"].values
    macd      = df["MACD"].values
    macd_sig  = df["MACD_Signal"].values

    # build previous arrays via roll
    prev_close    = np.roll(close,    1); prev_close[0]    = np.nan
    prev_ma10     = np.roll(ma10,     1); prev_ma10[0]     = np.nan
    prev_macd     = np.roll(macd,     1); prev_macd[0]     = np.nan
    prev_macd_sig = np.roll(macd_sig, 1); prev_macd_sig[0] = np.nan

    # MA crossover
    ma_signal = np.zeros(n, dtype=int)
    ma_signal[(prev_close < prev_ma10) & (close > ma10)] = 1
    ma_signal[(prev_close > prev_ma10) & (close < ma10)] = -1

    # RSI signal
    rsi_signal = np.zeros(n, dtype=int)
    rsi_signal[rsi14 < 30] = 1
    rsi_signal[rsi14 > 70] = -1

    # MACD crossover
    macd_signal = np.zeros(n, dtype=int)
    macd_signal[(prev_macd < prev_macd_sig) & (macd > macd_sig)] = 1
    macd_signal[(prev_macd > prev_macd_sig) & (macd < macd_sig)] = -1

    # Composite long if any indicator is positive
    composite = ((ma_signal + rsi_signal + macd_signal) > 0).astype(int)

    # assign back to df
    df["ma_signal"]   = ma_signal
    df["rsi_signal"]  = rsi_signal
    df["macd_signal"] = macd_signal
    df["composite"]   = composite

    return df

def backtest(df: pd.DataFrame) -> pd.DataFrame:
    df["return"]       = df["Close"].pct_change().fillna(0)
    df["strat_return"] = df["composite"].shift(1).fillna(0) * df["return"]
    df["cum_bh"]       = (1 + df["return"]).cumprod()
    df["cum_strat"]    = (1 + df["strat_return"]).cumprod()
    return df

# ─── Streamlit UI ────────────────────────────────────────────────────────────

ticker = st.text_input("Enter a stock ticker (e.g., AAPL)", "AAPL").upper()

if st.button("📊 Generate Today's Signals & Backtest"):
    df = yf.download(ticker, period="6mo", progress=False)

    if df.empty or "Close" not in df.columns:
        st.error("❌ No valid price data for this ticker.")
    else:
        # 1) Compute indicators
        df = compute_indicators(df)

        # 2) Filter out rows with NaN in any indicator
        mask = (
            df["MA10"].notna() &
            df["RSI14"].notna() &
            df["MACD"].notna() &
            df["MACD_Signal"].notna()
        )
        df = df.loc[mask]

        if len(df) < 2:
            st.error("⚠️ Not enough data after filtering indicators.")
        else:
            # 3) Generate signals & backtest
            df = generate_signals(df)
            df = backtest(df)

            # 4) Display today's signals
            last = df.iloc[-1]
            ma_lbl   = {1: "📈", 0: "⏸️", -1: "📉"}[int(last["ma_signal"])]
            rsi_lbl  = f"{last['RSI14']:.1f}"
            macd_lbl = {1: "📈", 0: "⏸️", -1: "📉"}[int(last["macd_signal"])]
            reco     = "🟢 BUY" if last["composite"] == 1 else "🔴 SELL"

            st.success(f"{ticker}: MA→{ma_lbl}   RSI→{rsi_lbl}   MACD→{macd_lbl}")
            st.info(f"Recommendation: {reco}")

            # 5) Backtest performance summary
            bh_ret    = 100 * (df["cum_bh"].iloc[-1] - 1)
            strat_ret = 100 * (df["cum_strat"].iloc[-1] - 1)
            st.markdown(f"**Buy & Hold Return:** {bh_ret:.2f}%   |   **Strategy Return:** {strat_ret:.2f}%")

            # 6) Plot all panels
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
            ax3.set_title("MACD"); ax3.legend()

            ax4.plot(df.index, df["cum_bh"], label="Buy & Hold", linestyle=":", color="gray")
            ax4.plot(df.index, df["cum_strat"], label="Strategy", color="green")
            ax4.set_title("Cumulative Performance"); ax4.legend()

            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
