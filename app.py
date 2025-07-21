import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ───────── Streamlit page config ───────────────
st.set_page_config(page_title="QuantaraX Composite Signals", layout="centered")
st.title("🚀 QuantaraX — Composite Signal Engine")
st.subheader("MA + RSI + MACD Composite Signals & Backtest")

# ───────── Sidebar params ───────────────────────
st.sidebar.header("Indicator Parameters")
ma_window   = st.sidebar.slider("MA window",       5, 50, 10)
rsi_period  = st.sidebar.slider("RSI lookback",    5, 30, 14)
macd_fast   = st.sidebar.slider("MACD fast span",  5, 20, 12)
macd_slow   = st.sidebar.slider("MACD slow span", 20, 40, 26)
macd_signal = st.sidebar.slider("MACD signal span",5, 20, 9)

# ───────── Load & compute indicators ──────────
@st.cache_data
def load_and_compute(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="6mo", progress=False)
    if df.empty or "Close" not in df.columns:
        return pd.DataFrame()
    # MA
    df[f"MA{ma_window}"] = df["Close"].rolling(ma_window).mean()
    # RSI
    delta = df["Close"].diff()
    up, down = delta.clip(0), -delta.clip(upper=0)
    ema_up   = up.ewm(com=rsi_period-1, adjust=False).mean()
    ema_down = down.ewm(com=rsi_period-1, adjust=False).mean()
    df[f"RSI{rsi_period}"] = 100 - 100/(1 + ema_up/ema_down)
    # MACD
    ema_f = df["Close"].ewm(span=macd_fast, adjust=False).mean()
    ema_s = df["Close"].ewm(span=macd_slow, adjust=False).mean()
    macd  = ema_f - ema_s
    sig   = macd.ewm(span=macd_signal, adjust=False).mean()
    df["MACD"], df["MACD_Signal"] = macd, sig
    # drop rows missing any of our four key columns
    required = [f"MA{ma_window}", f"RSI{rsi_period}", "MACD", "MACD_Signal"]
    df = df.dropna(subset=required).reset_index(drop=True)
    return df

# ───────── Build composite signals ─────────────
def build_composite(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    close = df["Close"].values
    ma    = df[f"MA{ma_window}"].values
    rsi   = df[f"RSI{rsi_period}"].values
    macd  = df["MACD"].values
    sigl  = df["MACD_Signal"].values

    ma_sig   = np.zeros(n, dtype=int)
    rsi_sig  = np.zeros(n, dtype=int)
    macd_sig = np.zeros(n, dtype=int)
    comp     = np.zeros(n, dtype=int)

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
        # composite vote
        comp[i] = ma_sig[i] + rsi_sig[i] + macd_sig[i]

    df["MA_Signal"]   = ma_sig
    df["RSI_Signal"]  = rsi_sig
    df["MACD_Signal"] = macd_sig
    df["Composite"]   = comp
    df["Trade"]       = np.sign(comp)  # +1 buy, -1 sell, 0 hold
    return df

# ───────── Backtest & metrics ───────────────────
def backtest(df: pd.DataFrame):
    df = df.copy()
    df["Return"]   = df["Close"].pct_change().fillna(0)
    df["Position"] = df["Trade"].shift(1).fillna(0).clip(lower=0)
    df["StratRet"] = df["Position"] * df["Return"]
    df["CumBH"]    = (1 + df["Return"]).cumprod()
    df["CumStrat"] = (1 + df["StratRet"]).cumprod()

    # drawdown
    dd = df["CumStrat"] / df["CumStrat"].cummax() - 1
    max_dd = dd.min() * 100
    # sharpe
    sharpe = df["StratRet"].mean() / df["StratRet"].std() * np.sqrt(252)
    # win rate
    win_rate = (df["StratRet"] > 0).mean() * 100
    return df, max_dd, sharpe, win_rate, dd

# ───────── Main ───────────────────────────────────
ticker = st.text_input("Ticker (e.g. AAPL)", "AAPL").upper()
if st.button("▶️ Run Composite Backtest"):
    df = load_and_compute(ticker)       # ← correct call, no extra assignment!
    if df.empty:
        st.error(f"No data for '{ticker}'.")
        st.stop()

    df = build_composite(df)
    df, max_dd, sharpe, win_rate, dd = backtest(df)

    # Live recommendation
    rec_map = {1:"🟢 BUY", 0:"🟡 HOLD", -1:"🔴 SELL"}
    st.success(f"**{ticker}**: {rec_map[int(df['Trade'].iloc[-1])]}")

    # Metrics
    bh_ret    = (df["CumBH"].iloc[-1] - 1) * 100
    strat_ret = (df["CumStrat"].iloc[-1] - 1) * 100
    st.markdown(f"""
    **Buy & Hold:** {bh_ret:.2f}%  
    **Strategy:** {strat_ret:.2f}%  
    **Sharpe:** {sharpe:.2f}  
    **Max Drawdown:** {max_dd:.2f}%  
    **Win Rate:** {win_rate:.1f}%  
    """)

    # Download CSV
    st.download_button(
        "Download full signals",
        df.to_csv(index=False),
        f"{ticker}_signals.csv"
    )

    # Plots
    fig, ax = plt.subplots(4,1, figsize=(10,14), sharex=True)
    ax[0].plot(df["Close"], label="Close");    ax[0].plot(df[f"MA{ma_window}"], "--", label=f"MA{ma_window}")
    ax[0].set_title("Price & MA"); ax[0].legend()

    ax[1].bar(df.index, df["Composite"], color="purple")
    ax[1].set_title("Composite Vote")

    ax[2].plot(df["CumBH"],    linestyle=":",  label="Buy & Hold")
    ax[2].plot(df["CumStrat"], label="Strategy")
    ax[2].set_title("Equity Curves"); ax[2].legend()

    ax[3].plot(dd, color="red"); ax[3].axhline(max_dd/100, linestyle="--", color="black")
    ax[3].set_title("Drawdown")

    plt.xticks(rotation=45); plt.tight_layout()
    st.pyplot(fig)
