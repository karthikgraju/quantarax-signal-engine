import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# ─ Streamlit Setup ─────────────────────────────────────────────────────────
st.set_page_config(page_title="QuantaraX Signal Engine", layout="centered")
st.title("🚀 QuantaraX — Smart Signal Engine")
st.subheader("🔍 Generate Today's Signals & Backtest")

# ─ Fetch & compute indicators ───────────────────────────────────────────────
def load_and_compute(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="6mo", progress=False)
    if df.empty or "Close" not in df:
        return pd.DataFrame()

    # 10-day MA
    df["MA10"] = df["Close"].rolling(10).mean()

    # RSI14
    delta = df["Close"].diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    ema_up   = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    df["RSI14"] = 100 - (100 / (1 + ema_up / ema_down))

    # MACD (12-26) + signal (9)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

    return df

# ─ Build signals as pure Python lists ───────────────────────────────────────
def build_signal_lists(df: pd.DataFrame):
    closes    = df["Close"].tolist()
    ma10s     = df["MA10"].tolist()
    rsis      = df["RSI14"].tolist()
    macds     = df["MACD"].tolist()
    macdsigs  = df["MACD_Signal"].tolist()

    n = len(closes)
    ma_signal   = [0]*n
    rsi_signal  = [0]*n
    macd_signal = [0]*n
    composite   = [0]*n

    for i in range(1, n):
        # MA crossover
        if closes[i-1] < ma10s[i-1] and closes[i] > ma10s[i]:
            ma_signal[i] = 1
        elif closes[i-1] > ma10s[i-1] and closes[i] < ma10s[i]:
            ma_signal[i] = -1

        # RSI oversold/overbought
        if rsis[i] < 30:
            rsi_signal[i] = 1
        elif rsis[i] > 70:
            rsi_signal[i] = -1

        # MACD crossover
        if macds[i-1] < macdsigs[i-1] and macds[i] > macdsigs[i]:
            macd_signal[i] = 1
        elif macds[i-1] > macdsigs[i-1] and macds[i] < macdsigs[i]:
            macd_signal[i] = -1

        # Composite: go long if any indicator is positive
        if (ma_signal[i] + rsi_signal[i] + macd_signal[i]) > 0:
            composite[i] = 1

    return ma_signal, rsi_signal, macd_signal, composite

# ─ Simple backtest ─────────────────────────────────────────────────────────
def backtest(df: pd.DataFrame) -> pd.DataFrame:
    df["return"]       = df["Close"].pct_change().fillna(0)
    df["strat_return"] = df["composite"].shift(1).fillna(0) * df["return"]
    df["cum_bh"]       = (1 + df["return"]).cumprod()
    df["cum_strat"]    = (1 + df["strat_return"]).cumprod()
    return df

# ── Main App Logic ─────────────────────────────────────────────────────────
ticker = st.text_input("Enter a stock ticker (e.g., AAPL)", "AAPL").upper()

if st.button("📊 Generate Today's Signals & Backtest"):
    df = load_and_compute(ticker)
    if df.empty:
        st.error(f"❌ No data for '{ticker}'.")
        st.stop()

    # Filter out any rows where indicators are NaN
    mask = df[["MA10","RSI14","MACD","MACD_Signal"]].notna().all(axis=1)
    df = df.loc[mask].copy()
    if len(df) < 2:
        st.error("⚠️ Not enough data after cleaning.")
        st.stop()

    # Build signal lists and assign
    ma_sig, rsi_sig, macd_sig, comp = build_signal_lists(df)
    df["ma_signal"]   = ma_sig
    df["rsi_signal"]  = rsi_sig
    df["macd_signal"] = macd_sig
    df["composite"]   = comp

    # Backtest
    df = backtest(df)

    # Display latest signals
    last = df.iloc[-1]
    st.success(
        f"{ticker}: "
        f"MA→{ {1:'📈',0:'⏸️',-1:'📉'}[last.ma_signal] }   "
        f"RSI→{last.RSI14:.1f}   "
        f"MACD→{ {1:'📈',0:'⏸️',-1:'📉'}[last.macd_signal] }"
    )
    st.info("Recommendation: " + ("🟢 BUY" if last.composite==1 else "🔴 SELL"))

    # Performance summary
    bh_ret    = 100*(df.cum_bh.iloc[-1]-1)
    strat_ret = 100*(df.cum_strat.iloc[-1]-1)
    st.markdown(f"**Buy & Hold:** {bh_ret:.2f}%   |   **Strategy:** {strat_ret:.2f}%")

    # Plot panels
    fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1, figsize=(10,12), sharex=True)
    ax1.plot(df.index, df["Close"], label="Close", color="blue")
    ax1.plot(df.index, df["MA10"],  label="MA10", linestyle="--", color="orange")
    ax1.set_title("Price & 10-Day MA"); ax1.legend()

    ax2.plot(df.index, df["RSI14"], label="RSI14", color="purple")
    ax2.axhline(70, color="red", linestyle="--"); ax2.axhline(30, color="green", linestyle="--")
    ax2.set_title("RSI (14)"); ax2.legend()

    ax3.plot(df.index, df["MACD"],        label="MACD", color="black")
    ax3.plot(df.index, df["MACD_Signal"], label="Signal", color="magenta", linestyle="--")
    ax3.bar(df.index, df["MACD_Hist"],    label="Hist", color="gray", alpha=0.5)
    ax3.set_title("MACD (12,26,9)"); ax3.legend()

    ax4.plot(df.index, df["cum_bh"],    label="Buy & Hold", linestyle=":", color="gray")
    ax4.plot(df.index, df["cum_strat"], label="Strategy",    color="green")
    ax4.set_title("Cumulative Performance"); ax4.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
