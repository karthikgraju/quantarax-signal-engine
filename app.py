import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="QuantaraX — Smart Signal Engine", layout="centered")
st.title("🚀 QuantaraX — Smart Signal Engine")
st.subheader("🔍 Generate Today's Signals & Backtest")

# ─── Indicator & Signal Functions ─────────────────────────────────────────

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # 10-day MA
    df["MA10"] = df["Close"].rolling(10).mean()
    # 14-day RSI
    delta = df["Close"].diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    ema_up   = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    df["RSI14"] = 100 - (100 / (1 + ema_up / ema_down))
    # MACD (12-26) + signal (9) + hist
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]
    return df

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    # MA crossover
    df["ma_signal"] = 0
    prev_close = df["Close"].shift(1)
    prev_ma    = df["MA10"].shift(1)
    df.loc[(prev_close < prev_ma) & (df["Close"] > df["MA10"]), "ma_signal"] = 1
    df.loc[(prev_close > prev_ma) & (df["Close"] < df["MA10"]), "ma_signal"] = -1

    # RSI signal
    df["rsi_signal"] = 0
    df.loc[df["RSI14"] < 30, "rsi_signal"] = 1
    df.loc[df["RSI14"] > 70, "rsi_signal"] = -1

    # MACD crossover
    df["macd_signal"] = 0
    prev_macd = df["MACD"].shift(1)
    prev_sig  = df["MACD_Signal"].shift(1)
    df.loc[(prev_macd < prev_sig) & (df["MACD"] > df["MACD_Signal"]), "macd_signal"] = 1
    df.loc[(prev_macd > prev_sig) & (df["MACD"] < df["MACD_Signal"]), "macd_signal"] = -1

    # Composite: go long if sum > 0
    df["composite"] = ((df["ma_signal"] + df["rsi_signal"] + df["macd_signal"]) > 0).astype(int)
    return df

def backtest(df: pd.DataFrame) -> pd.DataFrame:
    df["return"]       = df["Close"].pct_change().fillna(0)
    df["strat_return"] = df["composite"].shift(1).fillna(0) * df["return"]
    df["cum_bh"]       = (1 + df["return"]).cumprod()
    df["cum_strat"]    = (1 + df["strat_return"]).cumprod()
    return df

# ─── Main UI & Workflow ────────────────────────────────────────────────────

ticker = st.text_input("Enter a stock ticker (e.g., AAPL)", "AAPL").upper()

if st.button("📊 Generate Today's Signals & Backtest"):

    # 1) Download data
    df = yf.download(ticker, period="6mo", progress=False)
    if df.empty or "Close" not in df:
        st.error("❌ No valid price data for this ticker.")
    else:
        # 2) Compute indicators
        df = compute_indicators(df)

        # 3) Filter out rows where any indicator is NaN
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
            # 4) Generate signals + backtest
            df = generate_signals(df)
            df = backtest(df)

            # 5) Display today's signals
            last = df.iloc[-1]
            ma_lbl   = {1:"📈", 0:"⏸️", -1:"📉"}[int(last["ma_signal"])]
            rsi_lbl  = f"{last['RSI14']:.1f}"
            macd_lbl = {1:"📈", 0:"⏸️", -1:"📉"}[int(last["macd_signal"])]
            rec      = "🟢 BUY" if last["composite"]==1 else "🔴 SELL"

            st.success(f"{ticker}: MA→{ma_lbl}   RSI→{rsi_lbl}   MACD→{macd_lbl}")
            st.info(f"Recommendation: {rec}")

            # 6) Backtest results
            ret_bh    = 100*(df["cum_bh"].iloc[-1] - 1)
            ret_strat = 100*(df["cum_strat"].iloc[-1] - 1)
            st.markdown(f"**Buy & Hold Return:** {ret_bh:.2f}% &nbsp;|&nbsp; **Strategy Return:** {ret_strat:.2f}%")

            # 7) Plot: Price/MA, RSI, MACD, Performance
            fig, axes = plt.subplots(4,1,figsize=(10,12), sharex=True)
            ax1, ax2, ax3, ax4 = axes

            # Price & MA
            ax1.plot(df.index, df["Close"], label="Close", color="blue")
            ax1.plot(df.index, df["MA10"], label="MA10", linestyle="--", color="orange")
            ax1.set_title("Price & 10-Day MA"); ax1.legend()

            # RSI
            ax2.plot(df.index, df["RSI14"], label="RSI(14)", color="purple")
            ax2.axhline(70, color="red", linestyle="--")
            ax2.axhline(30, color="green", linestyle="--")
            ax2.set_title("RSI"); ax2.legend()

            # MACD
            ax3.plot(df.index, df["MACD"], label="MACD", color="black")
            ax3.plot(df.index, df["MACD_Signal"], label="Signal", color="magenta", linestyle="--")
            ax3.bar(df.index, df["MACD_Hist"], label="Hist", color="gray", alpha=0.5)
            ax3.set_title("MACD"); ax3.legend()

            # Strategy vs BH
            ax4.plot(df.index, df["cum_bh"],    label="Buy & Hold", color="gray", linestyle=":")
            ax4.plot(df.index, df["cum_strat"], label="Strategy",    color="green")
            ax4.set_title("Cumulative Performance"); ax4.legend()

            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
