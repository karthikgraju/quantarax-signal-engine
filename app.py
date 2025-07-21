import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="QuantaraX — Smart Signal Engine", layout="centered")
st.title("🚀 QuantaraX — Smart Signal Engine")
st.subheader("🔍 Generate Today's Signals & Backtest")

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # 10-day moving average
    df["MA10"] = df["Close"].rolling(10).mean()
    # 14-day RSI
    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    df["RSI14"] = 100 - (100 / (1 + ema_up/ema_down))
    # MACD (12,26) & signal (9)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    return df

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    # MA crossover signal
    df["ma_signal"] = 0
    ma_prev = df["MA10"].shift(1)
    close_prev = df["Close"].shift(1)
    df.loc[(close_prev < ma_prev) & (df["Close"] > df["MA10"]),  "ma_signal"] = 1
    df.loc[(close_prev > ma_prev) & (df["Close"] < df["MA10"]),  "ma_signal"] = -1

    # RSI signal
    df["rsi_signal"] = 0
    df.loc[df["RSI14"] < 30, "rsi_signal"] = 1
    df.loc[df["RSI14"] > 70, "rsi_signal"] = -1

    # MACD crossover signal
    df["macd_signal"] = 0
    macd_prev = df["MACD"].shift(1)
    sig_prev  = df["MACD_Signal"].shift(1)
    df.loc[(macd_prev < sig_prev) & (df["MACD"] > df["MACD_Signal"]), "macd_signal"] = 1
    df.loc[(macd_prev > sig_prev) & (df["MACD"] < df["MACD_Signal"]), "macd_signal"] = -1

    # Composite signal: buy=1 if sum>0, sell=0 otherwise
    df["composite"] = (df["ma_signal"] + df["rsi_signal"] + df["macd_signal"] > 0).astype(int)
    return df

def backtest(df: pd.DataFrame) -> pd.DataFrame:
    # daily return
    df["return"] = df["Close"].pct_change().fillna(0)
    # strategy return: position from previous bar * today's return
    df["strategy_ret"] = df["composite"].shift(1).fillna(0) * df["return"]
    # cumulative
    df["cum_bh"] = (1 + df["return"]).cumprod()
    df["cum_strat"] = (1 + df["strategy_ret"]).cumprod()
    return df

ticker = st.text_input("Enter a stock ticker (e.g., AAPL)", "AAPL").upper()

if st.button("📊 Generate Today's Signals & Backtest"):
    # 1) download
    df = yf.download(ticker, period="6mo", progress=False)
    if df.empty or "Close" not in df.columns:
        st.error("❌ No valid price data for this ticker.")
    else:
        # 2) indicators & signals
        df = compute_indicators(df)
        df = df.dropna(subset=["MA10","RSI14","MACD","MACD_Signal"])
        df = generate_signals(df)
        # 3) backtest
        df = backtest(df)

        # Today's signals (last row)
        last = df.iloc[-1]
        st.success(
            f"{ticker}: MA→{['⏸️','📈','📉'][last['ma_signal']+1]}   "
            f"RSI→{last['RSI14']:.1f}   "
            f"MACD→{['⏸️','📈','📉'][last['macd_signal']+1]}"
        )
        rec = "🟢 BUY" if last["composite"]==1 else "🔴 SELL" if last["composite"]==0 else "🟡 HOLD"
        st.info(f"Recommendation: {rec}")

        # 4) performance metrics
        total_bh    = df["cum_bh"].iloc[-1] - 1
        total_strat = df["cum_strat"].iloc[-1] - 1
        st.markdown(f"**Buy & Hold Return**: {total_bh*100:.2f}%  &nbsp; |  &nbsp; **Strategy Return**: {total_strat*100:.2f}%")

        # 5) plots
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1, figsize=(10,12), sharex=True)

        # Price + MA
        ax1.plot(df.index, df["Close"], label="Close", color="blue")
        ax1.plot(df.index, df["MA10"], label="MA10", color="orange", linestyle="--")
        ax1.set_title("Price & 10-Day MA"); ax1.legend()

        # RSI14
        ax2.plot(df.index, df["RSI14"], label="RSI14", color="purple")
        ax2.axhline(70, color="red", linestyle="--"); ax2.axhline(30, color="green", linestyle="--")
        ax2.set_title("RSI (14)"); ax2.legend()

        # MACD
        ax3.plot(df.index, df["MACD"], label="MACD", color="black")
        ax3.plot(df.index, df["MACD_Signal"], label="Signal", color="magenta", linestyle="--")
        ax3.bar(df.index, df["MACD_Hist"], label="Hist", color="gray", alpha=0.5)
        ax3.set_title("MACD (12,26,9)"); ax3.legend()

        # Backtest performance
        ax4.plot(df.index, df["cum_bh"],    label="Buy & Hold", color="gray", linestyle=":")
        ax4.plot(df.index, df["cum_strat"], label="Strategy",    color="green")
        ax4.set_title("Cumulative Performance"); ax4.legend()

        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
