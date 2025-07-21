import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ───────────────────────────── Defaults & Session State ─────────────────────────────
DEFAULTS = {
    "ma_window":   10,
    "rsi_period":  14,
    "macd_fast":   12,
    "macd_slow":   26,
    "macd_signal":  9
}
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ───────────────────────────── Sidebar Controls ─────────────────────────────
st.sidebar.header("Controls")
if st.sidebar.button("🔄 Reset to defaults"):
    for key, val in DEFAULTS.items():
        st.session_state[key] = val

st.sidebar.header("Indicator Parameters")
ma_window   = st.sidebar.slider("MA window",       5, 50, st.session_state["ma_window"],   key="ma_window")
rsi_period  = st.sidebar.slider("RSI lookback",    5, 30, st.session_state["rsi_period"],  key="rsi_period")
macd_fast   = st.sidebar.slider("MACD fast span",  5, 20, st.session_state["macd_fast"],   key="macd_fast")
macd_slow   = st.sidebar.slider("MACD slow span", 20, 40, st.session_state["macd_slow"],   key="macd_slow")
macd_signal = st.sidebar.slider("MACD signal span",5, 20, st.session_state["macd_signal"], key="macd_signal")

# ───────────────────────────── Page Setup ─────────────────────────────
st.set_page_config(page_title="QuantaraX Composite Signals", layout="centered")
st.title("🚀 QuantaraX — Composite Signal Engine")
st.write("MA + RSI + MACD Composite Signals & Backtest")

# ───────────────────────────── Load & Compute Indicators ─────────────────────────────
@st.cache_data
def load_and_compute(ticker, ma_window, rsi_period, macd_fast, macd_slow, macd_signal):
    df = yf.download(ticker, period="6mo", progress=False)
    if df.empty or "Close" not in df.columns:
        return pd.DataFrame()

    # Moving Average
    df[f"MA{ma_window}"] = df["Close"].rolling(ma_window).mean()

    # RSI
    delta    = df["Close"].diff()
    up       = delta.clip(lower=0)
    down     = -delta.clip(upper=0)
    ema_up   = up.ewm(com=rsi_period-1, adjust=False).mean()
    ema_down = down.ewm(com=rsi_period-1, adjust=False).mean()
    df[f"RSI{rsi_period}"] = 100 - 100/(1 + ema_up/ema_down)

    # MACD
    ema_f = df["Close"].ewm(span=macd_fast, adjust=False).mean()
    ema_s = df["Close"].ewm(span=macd_slow, adjust=False).mean()
    macd  = ema_f - ema_s
    sig   = macd.ewm(span=macd_signal, adjust=False).mean()
    df["MACD"], df["MACD_Signal"] = macd, sig

    # Drop any row missing a key column
    cols = [f"MA{ma_window}", f"RSI{rsi_period}", "MACD", "MACD_Signal"]
    df = df.dropna(subset=cols).reset_index(drop=True)
    return df

# ───────────────────────────── Build Composite Signals ─────────────────────────────
def build_composite(df):
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
        # MA crossover signal
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
        # Composite vote sum
        comp[i] = ma_sig[i] + rsi_sig[i] + macd_sig[i]

    df["MA_Signal"]   = ma_sig
    df["RSI_Signal"]  = rsi_sig
    df["MACD_Signal"] = macd_sig
    df["Composite"]   = comp
    df["Trade"]       = np.sign(comp)
    return df

# ───────────────────────────── Backtest & Metrics ─────────────────────────────
def backtest(df):
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

# ───────────────────────────── Single‐Ticker Backtest ─────────────────────────────
st.markdown("### Single‐Ticker Backtest")
ticker = st.text_input("Ticker (e.g. AAPL)", "AAPL").upper()
if st.button("▶️ Run Composite Backtest"):
    df = load_and_compute(ticker, ma_window, rsi_period, macd_fast, macd_slow, macd_signal)
    if df.empty:
        st.error(f"No data for '{ticker}'.")
        st.stop()

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

    # Plot 3-panel chart
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    axes[0].plot(df["Close"], label="Close", color="blue")
    axes[0].plot(df[f"MA{ma_window}"], label=f"MA{ma_window}", color="orange")
    axes[0].set_title("Price & Moving Average")
    axes[0].legend()

    axes[1].bar(df.index, df["Composite"], color="purple")
    axes[1].set_title("Composite Vote (MA+RSI+MACD)")

    axes[2].plot(df["CumBH"], ":",  label="Buy & Hold")
    axes[2].plot(df["CumStrat"], "-", label="Strategy")
    axes[2].set_title("Equity Curves")
    axes[2].legend()

    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

# ───────────────────────────── Batch Backtest ─────────────────────────────
st.markdown("---")
st.markdown("### Batch Backtest")
batch_input = st.text_area("Enter tickers (comma-separated)", "AAPL, MSFT, TSLA, SPY, QQQ").upper()

if st.button("▶️ Run Batch Backtest"):
    tickers = [t.strip() for t in batch_input.split(",") if t.strip()]
    results = []
    rec_map = {1:"BUY", 0:"HOLD", -1:"SELL"}

    for t in tickers:
        df_t = load_and_compute(t, ma_window, rsi_period, macd_fast, macd_slow, macd_signal)
        if df_t.empty:
            continue

        df_t = build_composite(df_t)
        df_t, max_dd, sharpe, win_rt, dd = backtest(df_t)

        results.append({
            "Ticker":         t,
            "Composite vote": int(df_t["Composite"].iloc[-1]),
            "Signal":         rec_map[int(df_t["Trade"].iloc[-1])],
            "BuyHold %":      (df_t["CumBH"].iloc[-1] - 1)*100,
            "Strat %":        (df_t["CumStrat"].iloc[-1] - 1)*100,
            "Sharpe":         sharpe,
            "Max Drawdown %": max_dd,
            "Win Rate %":     win_rt
        })

    if not results:
        st.error("No valid tickers/data.")
    else:
        perf_df = pd.DataFrame(results).set_index("Ticker")
        st.dataframe(perf_df)
        st.download_button("Download performance CSV", perf_df.to_csv(), "batch_performance.csv")

# ───────────────────────────── Hyperparameter Grid Search ─────────────────────────────
st.markdown("---")
st.header("🔧 Hyperparameter Optimization")
ma_grid  = st.multiselect("MA windows to test",    [5,10,15,20,25], default=[5,10,15])
rsi_grid = st.multiselect("RSI lookbacks to test", [7,14,21],     default=[14])

if st.button("🏃 Run Grid Search"):
    df_full = load_and_compute(
        ticker, ma_window, rsi_period, macd_fast, macd_slow, macd_signal
    )
    if df_full.empty:
        st.error(f"No data for {ticker}")
        st.stop()

    # split into first half (in-sample) and second half (out-of-sample)
    split = len(df_full) // 2
    df_train = df_full.iloc[:split].reset_index(drop=True)
    df_test  = df_full.iloc[split:].reset_index(drop=True)

    results = []
    for ma in ma_grid:
        for rsi in rsi_grid:
            # in-sample build
            df1 = df_train.copy()
            df1[f"MA{ma}"] = df1["Close"].rolling(ma).mean()
            delta = df1["Close"].diff()
            up    = delta.clip(lower=0)
            down  = -delta.clip(upper=0)
            e_up  = up.ewm(com=rsi-1, adjust=False).mean()
            e_dn  = down.ewm(com=rsi-1, adjust=False).mean()
            df1[f"RSI{rsi}"] = 100 - 100/(1 + e_up/e_dn)
            df1 = df1.dropna(subset=[f"MA{ma}", f"RSI{rsi}"]).reset_index(drop=True)

            df1["SigMA"]  = np.where(
                (df1["Close"].shift(1) < df1[f"MA{ma}"].shift(1)) &
                (df1["Close"] > df1[f"MA{ma}"]), 1,
                np.where(
                  (df1["Close"].shift(1) > df1[f"MA{ma}"].shift(1)) &
                  (df1["Close"] < df1[f"MA{ma}"]), -1, 0
                )
            )
            df1["SigRSI"] = np.where(df1[f"RSI{rsi}"] < 30, 1,
                              np.where(df1[f"RSI{rsi}"] > 70, -1, 0))
            df1["Trade"]  = np.sign(df1["SigMA"] + df1["SigRSI"])
            df1["Ret"]    = df1["Close"].pct_change().fillna(0)
            df1["Pos"]    = df1["Trade"].shift(1).fillna(0).clip(lower=0)
            df1["Sret"]   = df1["Pos"] * df1["Ret"]
            sharpe_in    = df1["Sret"].mean() / df1["Sret"].std() * np.sqrt(252)

            # out-of-sample build
            df2 = df_test.copy()
            df2[f"MA{ma}"] = df2["Close"].rolling(ma).mean()
            delta = df2["Close"].diff()
            up    = delta.clip(lower=0)
            down  = -delta.clip(upper=0)
            e_up  = up.ewm(com=rsi-1, adjust=False).mean()
            e_dn  = down.ewm(com=rsi-1, adjust=False).mean()
            df2[f"RSI{rsi}"] = 100 - 100/(1 + e_up/e_dn)
            df2 = df2.dropna(subset=[f"MA{ma}", f"RSI{rsi}"]).reset_index(drop=True)

            df2["SigMA"]  = np.where(
                (df2["Close"].shift(1) < df2[f"MA{ma}"].shift(1)) &
                (df2["Close"] > df2[f"MA{ma}"]), 1,
                np.where(
                  (df2["Close"].shift(1) > df2[f"MA{ma}"].shift(1)) &
                  (df2["Close"] < df2[f"MA{ma}"]), -1, 0
                )
            )
            df2["SigRSI"] = np.where(df2[f"RSI{rsi}"] < 30, 1,
                              np.where(df2[f"RSI{rsi}"] > 70, -1, 0))
            df2["Trade"] = np.sign(df2["SigMA"] + df2["SigRSI"])
            df2["Ret"]   = df2["Close"].pct_change().fillna(0)
            df2["Pos"]   = df2["Trade"].shift(1).fillna(0).clip(lower=0)
            df2["Sret"]  = df2["Pos"] * df2["Ret"]
            sharpe_out  = df2["Sret"].mean() / df2["Sret"].std() * np.sqrt(252)

            results.append({
                "MA": ma, "RSI": rsi,
                "Sharpe IN":  sharpe_in,
                "Sharpe OOS": sharpe_out
            })

    df_grid = pd.DataFrame(results).sort_values("Sharpe OOS", ascending=False)
    st.dataframe(df_grid, use_container_width=True)
