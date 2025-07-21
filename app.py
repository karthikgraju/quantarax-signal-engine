import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── Page config ─────────────────────────────────────────────────────────
st.set_page_config(page_title="QuantaraX Composite Signals", layout="centered")
st.title("🚀 QuantaraX — Composite Signal Engine")

# ── Sidebar: parameter controls ──────────────────────────────────────────
st.sidebar.header("Indicator Parameters")
ma_window   = st.sidebar.slider("MA window",   5, 50, 10)
rsi_period  = st.sidebar.slider("RSI lookback",5, 30, 14)
macd_fast   = st.sidebar.slider("MACD fast span",   5, 20, 12)
macd_slow   = st.sidebar.slider("MACD slow span",  20, 40, 26)
macd_signal = st.sidebar.slider("MACD signal span", 5, 20, 9)

# ── Load & compute indicators ────────────────────────────────────────────
@st.cache_data
def load_and_compute(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="6mo", progress=False)
    if df.empty or "Close" not in df:
        return pd.DataFrame()
    # MA
    df[f"MA{ma_window}"] = df["Close"].rolling(ma_window).mean()
    # RSI
    delta = df["Close"].diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    ema_up   = up.ewm(com=rsi_period-1, adjust=False).mean()
    ema_down = down.ewm(com=rsi_period-1, adjust=False).mean()
    df[f"RSI{rsi_period}"] = 100 - (100 / (1 + ema_up/ema_down))
    # MACD
    ema_f = df["Close"].ewm(span=macd_fast, adjust=False).mean()
    ema_s = df["Close"].ewm(span=macd_slow, adjust=False).mean()
    macd  = ema_f - ema_s
    sig   = macd.ewm(span=macd_signal, adjust=False).mean()
    df["MACD"], df["MACD_Sig"] = macd, sig
    # Drop any rows where our indicators are NaN, then reset index for easy looping
    df = df.dropna(subset=[f"MA{ma_window}", f"RSI{rsi_period}", "MACD", "MACD_Sig"]).reset_index(drop=True)
    return df

# ── Build composite signals via pure‐Python loop ──────────────────────────
def build_composite_signals(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    ma   = df[f"MA{ma_window}"].values
    close= df["Close"].values
    rsi  = df[f"RSI{rsi_period}"].values
    macd = df["MACD"].values
    sigl = df["MACD_Sig"].values

    ma_sig   = [0]*n
    rsi_sig  = [0]*n
    macd_sig = [0]*n
    comp_sig = [0]*n

    for i in range(1, n):
        # MA crossover
        if close[i-1] < ma[i-1] and close[i] > ma[i]:
            ma_sig[i] = 1
        elif close[i-1] > ma[i-1] and close[i] < ma[i]:
            ma_sig[i] = -1
        # RSI
        if rsi[i] < 30:
            rsi_sig[i] = 1
        elif rsi[i] > 70:
            rsi_sig[i] = -1
        # MACD crossover
        if macd[i-1] < sigl[i-1] and macd[i] > sigl[i]:
            macd_sig[i] = 1
        elif macd[i-1] > sigl[i-1] and macd[i] < sigl[i]:
            macd_sig[i] = -1
        # Composite vote
        comp_sig[i] = ma_sig[i] + rsi_sig[i] + macd_sig[i]

    df["MA_Signal"]   = ma_sig
    df["RSI_Signal"]  = rsi_sig
    df["MACD_Signal"] = macd_sig
    df["Composite"]   = comp_sig
    # Final trade: +1 if vote>0, -1 if vote<0, else 0
    df["Trade"] = df["Composite"].map(lambda x: 1 if x>0 else (-1 if x<0 else 0))
    return df

# ── Backtest & metrics ───────────────────────────────────────────────────
def backtest_and_metrics(df: pd.DataFrame):
    df = df.copy()
    df["Return"]    = df["Close"].pct_change().fillna(0)
    df["Position"]  = df["Trade"].shift(1).fillna(0).clip(lower=0)
    df["StratRet"]  = df["Position"] * df["Return"]
    df["CumBH"]     = (1 + df["Return"]).cumprod()
    df["CumStrat"]  = (1 + df["StratRet"]).cumprod()
    # Drawdown
    dd = df["CumStrat"] / df["CumStrat"].cummax() - 1
    max_dd = dd.min() * 100
    # Sharpe (annualized)
    sharpe = df["StratRet"].mean() / df["StratRet"].std() * np.sqrt(252)
    # Win rate
    win_rate = (df["StratRet"] > 0).mean() * 100
    return df, max_dd, sharpe, win_rate, dd

# ── UI: ticker input & run ─────────────────────────────────────────────────
ticker = st.text_input("Ticker", "AAPL").upper()
if st.button("▶️ Run Composite Backtest"):
    df = load_and_compute = load_and_compute  # avoid name clash
    df = load_and_compute(ticker)
    if df.empty:
        st.error(f"No data for '{ticker}'.")
        st.stop()

    df = build_composite_signals(df)
    df, max_dd, sharpe, win_rate, dd = backtest_and_metrics(df)

    # Live recommendation
    rec_map = {1:"🟢 BUY", 0:"🟡 HOLD", -1:"🔴 SELL"}
    rec = rec_map[int(df["Trade"].iloc[-1])]
    st.success(f"**{ticker}**: {rec}")

    # Metrics display
    bh_ret = (df["CumBH"].iloc[-1] - 1)*100
    strat_ret = (df["CumStrat"].iloc[-1] - 1)*100
    st.markdown(f"""
    **Buy & Hold Return:** {bh_ret:.2f}%  
    **Strategy Return:** {strat_ret:.2f}%  
    **Sharpe Ratio:** {sharpe:.2f}  
    **Max Drawdown:** {max_dd:.2f}%  
    **Win Rate:** {win_rate:.1f}%  
    """)

    # Download
    st.download_button(
        "Download full signals CSV",
        df.to_csv(index=False),
        f"{ticker}_composite_signals.csv"
    )

    # Plots
    fig, axes = plt.subplots(4, 1, figsize=(10, 14), sharex=True)

    axes[0].plot(df["Close"], label="Close")
    axes[0].plot(df[f"MA{ma_window}"], label=f"MA{ma_window}", linestyle="--")
    axes[0].set_title("Price & Moving Average"); axes[0].legend()

    axes[1].bar(df.index, df["Composite"], color="purple")
    axes[1].set_title("Composite Vote (MA + RSI + MACD)")

    axes[2].plot(df["CumBH"],    linestyle=":",  label="Buy & Hold")
    axes[2].plot(df["CumStrat"], label="Strategy")
    axes[2].set_title("Equity Curves"); axes[2].legend()

    axes[3].plot(dd, color="red")
    axes[3].axhline(max_dd/100, color="black", linestyle="--")
    axes[3].set_title("Drawdown")

    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
