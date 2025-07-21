import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ──────────────────────────── Default Values Setup ────────────────────────────
DEFAULTS = {
    "ma_window":   10,
    "rsi_period":  14,
    "macd_fast":   12,
    "macd_slow":   26,
    "macd_signal":  9
}

# Initialize session_state with defaults on first load
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ──────────────────────────── Sidebar Controls ────────────────────────────────
st.sidebar.header("Controls")
if st.sidebar.button("🔄 Reset to defaults"):
    for key, val in DEFAULTS.items():
        st.session_state[key] = val

st.sidebar.header("Indicator Parameters")
ma_window   = st.sidebar.slider(
    "MA window", 5, 50, st.session_state["ma_window"], key="ma_window"
)
rsi_period  = st.sidebar.slider(
    "RSI lookback", 5, 30, st.session_state["rsi_period"], key="rsi_period"
)
macd_fast   = st.sidebar.slider(
    "MACD fast span", 5, 20, st.session_state["macd_fast"], key="macd_fast"
)
macd_slow   = st.sidebar.slider(
    "MACD slow span", 20, 40, st.session_state["macd_slow"], key="macd_slow"
)
macd_signal = st.sidebar.slider(
    "MACD signal span", 5, 20, st.session_state["macd_signal"], key="macd_signal"
)

# ──────────────────────────── Page Config & Title ─────────────────────────────
st.set_page_config(page_title="QuantaraX Composite Signals", layout="centered")
st.title("🚀 QuantaraX — Composite Signal Engine")
st.write("MA + RSI + MACD Composite Signals & Backtest")

# ────────────────────────── Load & Compute Indicators ─────────────────────────
@st.cache_data
def load_and_compute(
    ticker: str,
    ma_window: int,
    rsi_period: int,
    macd_fast: int,
    macd_slow: int,
    macd_signal: int
) -> pd.DataFrame:
    df = yf.download(ticker, period="6mo", progress=False)
    if df.empty or "Close" not in df.columns:
        return pd.DataFrame()

    # 1) MA
    df[f"MA{ma_window}"] = df["Close"].rolling(ma_window).mean()

    # 2) RSI
    delta    = df["Close"].diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    ema_up   = up.ewm(com=rsi_period-1, adjust=False).mean()
    ema_down = down.ewm(com=rsi_period-1, adjust=False).mean()
    df[f"RSI{rsi_period}"] = 100 - 100/(1 + ema_up/ema_down)

    # 3) MACD
    ema_f = df["Close"].ewm(span=macd_fast, adjust=False).mean()
    ema_s = df["Close"].ewm(span=macd_slow, adjust=False).mean()
    macd  = ema_f - ema_s
    sig   = macd.ewm(span=macd_signal, adjust=False).mean()
    df["MACD"], df["MACD_Signal"] = macd, sig

    # 4) Drop rows with NaN in any key column
    cols = [f"MA{ma_window}", f"RSI{rsi_period}", "MACD", "MACD_Signal"]
    mask = pd.Series(True, index=df.index)
    for c in cols:
        mask &= df[c].notna()
    df = df.loc[mask].reset_index(drop=True)
    return df

# ────────────────────────── Build Composite Signals ──────────────────────────
def build_composite(df: pd.DataFrame) -> pd.DataFrame:
    n     = len(df)
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
        # RSI thresholds
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
    df["Trade"]       = np.sign(comp)
    return df

# ───────────────────────────── Backtest & Metrics ────────────────────────────
def backtest(df: pd.DataFrame):
    df = df.copy()
    df["Return"]   = df["Close"].pct_change().fillna(0)
    df["Position"] = df["Trade"].shift(1).fillna(0).clip(lower=0)
    df["StratRet"] = df["Position"] * df["Return"]
    df["CumBH"]    = (1 + df["Return"]).cumprod()
    df["CumStrat"] = (1 + df["StratRet"]).cumprod()

    dd     = df["CumStrat"] / df["CumStrat"].cummax() - 1
    max_dd = dd.min() * 100
    sharpe = df["StratRet"].mean() / df["StratRet"].std() * np.sqrt(252)
    win_rt = (df["StratRet"] > 0).mean() * 100

    return df, max_dd, sharpe, win_rt, dd

# ───────────────────────────── Single‐Ticker Mode ────────────────────────────
st.markdown("### Single‐Ticker Backtest")
ticker = st.text_input("Ticker (e.g. AAPL)", "AAPL").upper()
if st.button("▶️ Run Composite Backtest"):
    df = load_and_compute(
        ticker,
        ma_window,
        rsi_period,
        macd_fast,
        macd_slow,
        macd_signal
    )
    if df.empty:
        st.error(f"No data for '{ticker}'."); st.stop()

    df = build_composite(df)
    df, max_dd, sharpe, win_rt, dd = backtest(df)

    rec_map = {1:"🟢 BUY", 0:"🟡 HOLD", -1:"🔴 SELL"}
    st.success(f"**{ticker}**: {rec_map[int(df['Trade'].iloc[-1])]}")

    bh_ret    = (df["CumBH"].iloc[-1] - 1) * 100
    strat_ret = (df["CumStrat"].iloc[-1] - 1) * 100
    st.markdown(f"""
    **Buy & Hold:** {bh_ret:.2f}%  
    **Strategy:** {strat_ret:.2f}%  
    **Sharpe:** {sharpe:.2f}  
    **Max Drawdown:** {max_dd:.2f}%  
    **Win Rate:** {win_rt:.1f}%  
    """)

    st.pyplot(plt.figure(figsize=(8,4)))  # you can re-plot if you like…

# ───────────────────────────── Batch‐Ticker Mode ─────────────────────────────
st.markdown("---")
st.markdown("### Batch Backtest")
batch_input = st.text_area(
    "Enter tickers (comma-separated)", 
    "AAPL, MSFT, TSLA, SPY, QQQ"
).upper()

if st.button("▶️ Run Batch Backtest"):
    tickers = [t.strip() for t in batch_input.split(",") if t.strip()]
    results = []

    rec_map = {1:"BUY", 0:"HOLD", -1:"SELL"}

    for t in tickers:
        df_t = load_and_compute(
            t, ma_window, rsi_period, macd_fast, macd_slow, macd_signal
        )
        if df_t.empty:
            continue

        df_t = build_composite(df_t)
        df_t, max_dd, sharpe, win_rt, dd = backtest(df_t)

        rec       = rec_map[int(df_t["Trade"].iloc[-1])]
        bh_return = (df_t["CumBH"].iloc[-1] - 1) * 100
        st_return = (df_t["CumStrat"].iloc[-1] - 1) * 100

        results.append({
            "Ticker":         t,
            "Composite vote": df_t["Composite"].iloc[-1],
            "Signal":         rec,
            "BuyHold %":      bh_return,
            "Strat %":        st_return,
            "Sharpe":         sharpe,
            "Max Drawdown %": max_dd,
            "Win Rate %":     win_rt
        })

    if not results:
        st.error("No valid tickers/data.")
    else:
        perf_df = pd.DataFrame(results).set_index("Ticker")
        st.dataframe(perf_df)
        st.download_button(
            "Download performance CSV",
            perf_df.to_csv(),
            "batch_performance.csv"
        )
