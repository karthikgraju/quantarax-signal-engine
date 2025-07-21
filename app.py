# app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

st.set_page_config(page_title="QuantaraX — Composite Signal Engine", layout="wide")

# --- Sidebar / defaults ---
if "ma_window" not in st.session_state:
    st.session_state.ma_window = 10
    st.session_state.rsi_period = 14
    st.session_state.macd_fast = 12
    st.session_state.macd_slow = 26
    st.session_state.macd_sig = 9

with st.sidebar:
    st.header("Controls")
    if st.button("🔄 Reset to defaults"):
        st.session_state.ma_window = 10
        st.session_state.rsi_period = 14
        st.session_state.macd_fast = 12
        st.session_state.macd_slow = 26
        st.session_state.macd_sig = 9
    st.markdown("### Indicator Parameters")
    st.session_state.ma_window = st.slider("MA window", 5, 50, st.session_state.ma_window)
    st.session_state.rsi_period = st.slider("RSI lookback", 5, 30, st.session_state.rsi_period)
    st.session_state.macd_fast = st.slider("MACD fast span", 5, 20, st.session_state.macd_fast)
    st.session_state.macd_slow = st.slider("MACD slow span", 20, 40, st.session_state.macd_slow)
    st.session_state.macd_sig = st.slider("MACD signal span", 5, 20, st.session_state.macd_sig)


def fetch_data(ticker: str):
    """Download last 6 months of daily OHLC data from Yahoo."""
    end = datetime.datetime.today()
    start = end - datetime.timedelta(days=180)
    df = yf.download(ticker, start=start, end=end, progress=False)
    return df


def compute_indicators(df: pd.DataFrame, ma_w, rsi_p, mf, ms, msig):
    """Compute MA, RSI(14), MACD and MACD signal line."""
    df = df.copy()
    df["MA"] = df["Close"].rolling(window=ma_w).mean()

    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=rsi_p).mean()
    avg_loss = loss.rolling(window=rsi_p).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df["Close"].ewm(span=mf, adjust=False).mean()
    exp2 = df["Close"].ewm(span=ms, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["MACD_Signal"] = df["MACD"].ewm(span=msig, adjust=False).mean()

    return df


@st.cache_data(show_spinner=False)
def load_and_compute(ticker, ma_w, rsi_p, mf, ms, msig):
    """Fetch, compute indicators, drop any rows missing all four required columns."""
    df = fetch_data(ticker)
    if df.empty:
        return pd.DataFrame()

    df = compute_indicators(df, ma_w, rsi_p, mf, ms, msig)

    # guard: drop any rows without all four
    required = ["MA", "RSI", "MACD", "MACD_Signal"]
    try:
        df = df.dropna(subset=required).reset_index(drop=True)
    except KeyError:
        # if any column is missing entirely, bail out
        return pd.DataFrame()

    return df


def run_backtest(df):
    """Simple MA crossover backtest: go long when close crosses above MA, exit when below."""
    df = df.copy()
    df["position"] = 0
    df.loc[df["Close"] > df["MA"], "position"] = 1
    df.loc[df["Close"] < df["MA"], "position"] = 0
    df["returns"] = df["Close"].pct_change().fillna(0)
    df["strat_ret"] = df["returns"] * df["position"].shift(1).fillna(0)

    buy_hold = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
    strat = (df["strat_ret"] + 1).cumprod().iloc[-1] - 1
    strat *= 100

    return buy_hold, strat, df


def sharpe_ratio(returns):
    return returns.mean() / returns.std() * np.sqrt(252)


def max_drawdown(equity):
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    return drawdown.min() * 100


def single_ticker_section():
    st.header("Single-Ticker Backtest")
    ticker = st.text_input("Ticker (e.g. AAPL)", "AAPL").upper()
    if st.button("▶️ Run Composite Backtest"):
        df = load_and_compute(
            ticker,
            st.session_state.ma_window,
            st.session_state.rsi_period,
            st.session_state.macd_fast,
            st.session_state.macd_slow,
            st.session_state.macd_sig,
        )
        if df.empty:
            st.error("⚠️ Not enough data to compute signals for that ticker/parameters.")
            return

        # build simple 1-indicator MA strategy for now
        buy_hold, strat, df_bt = run_backtest(df)
        sharpe = sharpe_ratio(df_bt["strat_ret"])
        mdd = max_drawdown((df_bt["strat_ret"] + 1).cumprod())

        latest_signal = (
            "BUY" if df_bt["Close"].iloc[-1] > df_bt["MA"].iloc[-1]
            else ("SELL" if df_bt["Close"].iloc[-1] < df_bt["MA"].iloc[-1] else "HOLD")
        )

        st.success(f"{ticker}: {latest_signal}")
        st.markdown(f"**Buy & Hold:** {buy_hold:.2f}%   •   **Strategy:** {strat:.2f}%   •   **Sharpe:** {sharpe:.2f}   •   **Max DD:** {mdd:.2f}%")

        # price vs MA plot
        fig, ax = plt.subplots(figsize=(8,3))
        ax.plot(df_bt.index, df_bt["Close"], label="Close")
        ax.plot(df_bt.index, df_bt["MA"], label=f"MA {st.session_state.ma_window}")
        ax.set_title(f"{ticker} Price & MA")
        ax.legend()
        st.pyplot(fig, use_container_width=True)


def batch_backtest_section():
    st.header("Batch Backtest")
    tickers = st.text_area("Enter tickers (comma-separated)", "AAPL, MSFT, TSLA, SPY, QQQ")
    if st.button("▶️ Run Batch Backtest"):
        out = []
        for t in [t.strip().upper() for t in tickers.split(",") if t.strip()]:
            df = load_and_compute(
                t,
                st.session_state.ma_window,
                st.session_state.rsi_period,
                st.session_state.macd_fast,
                st.session_state.macd_slow,
                st.session_state.macd_sig,
            )
            if df.empty:
                continue
            bh, strat, df_bt = run_backtest(df)
            sr = sharpe_ratio(df_bt["strat_ret"])
            mdd = max_drawdown((df_bt["strat_ret"]+1).cumprod())
            latest = "BUY" if df_bt["Close"].iloc[-1] > df_bt["MA"].iloc[-1] else "SELL"
            out.append({
                "Ticker": t,
                "Signal": latest,
                "BuyHold%": bh,
                "Strat%": strat,
                "Sharpe": sr,
                "MaxDD%": mdd
            })
        if not out:
            st.warning("No valid tickers/data.")
        else:
            df_out = pd.DataFrame(out)
            st.dataframe(df_out, use_container_width=True)
            csv = df_out.to_csv(index=False)
            st.download_button("Download performance CSV", data=csv, file_name="batch_backtest.csv")


def grid_search_section():
    st.header("🔧 Hyper-Parameter Optimization")
    gs_ticker = st.text_input("Grid Search Ticker", "AAPL").upper()
    ma_tests = st.multiselect("MA windows to test", list(range(5,51)), default=[5,10,15])
    rsi_tests = st.multiselect("RSI lookbacks to test", list(range(5,31)), default=[14])
    mf_tests  = st.multiselect("MACD fast spans", list(range(5,21)), default=[12])
    ms_tests  = st.multiselect("MACD slow spans", list(range(20,41)), default=[26])
    msig_tests= st.multiselect("MACD sig spans", list(range(5,21)), default=[9])

    if st.button("🏃 Run Grid Search"):
        results = []
        for ma_w in ma_tests:
            for rsi_p in rsi_tests:
                for mf in mf_tests:
                    for ms in ms_tests:
                        for msig in msig_tests:
                            df = load_and_compute(gs_ticker, ma_w, rsi_p, mf, ms, msig)
                            if df.empty:
                                continue
                            bh, strat, df_bt = run_backtest(df)
                            results.append({
                                "MA": ma_w, "RSI": rsi_p,
                                "MF": mf, "MS": ms, "MSig": msig,
                                "Strat%": strat
                            })
        if not results:
            st.warning("No valid runs.")
        else:
            dfg = pd.DataFrame(results)
            dfg = dfg.sort_values("Strat%", ascending=False).reset_index(drop=True)
            st.dataframe(dfg.head(20), use_container_width=True)
            csv = dfg.to_csv(index=False)
            st.download_button("Download grid results CSV", data=csv, file_name="grid_search.csv")


st.title("🚀 QuantaraX — Composite Signal Engine")
single_ticker_section()
batch_backtest_section()
grid_search_section()
