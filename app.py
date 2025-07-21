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
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ───────────────────────────── Sidebar Controls ─────────────────────────────
st.sidebar.header("Controls")
if st.sidebar.button("🔄 Reset to defaults"):
    for k, v in DEFAULTS.items():
        st.session_state[k] = v

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
@st.cache_data(show_spinner=False)
def load_and_compute(ticker, ma_window, rsi_period, macd_fast, macd_slow, macd_signal):
    df = yf.download(ticker, period="6mo", progress=False)
    if df.empty or "Close" not in df.columns:
        return pd.DataFrame()

    # --- Moving Average
    ma_col = f"MA{ma_window}"
    df[ma_col] = df["Close"].rolling(ma_window).mean()

    # --- RSI
    delta    = df["Close"].diff()
    up       = delta.clip(lower=0)
    down     = -delta.clip(upper=0)
    ema_up   = up.ewm(com=rsi_period-1, adjust=False).mean()
    ema_down = down.ewm(com=rsi_period-1, adjust=False).mean()
    rsi_col  = f"RSI{rsi_period}"
    df[rsi_col] = 100 - 100/(1 + ema_up/ema_down)

    # --- MACD
    ema_f = df["Close"].ewm(span=macd_fast, adjust=False).mean()
    ema_s = df["Close"].ewm(span=macd_slow, adjust=False).mean()
    macd  = ema_f - ema_s
    sig   = macd.ewm(span=macd_signal, adjust=False).mean()
    df["MACD"]        = macd
    df["MACD_Signal"] = sig

    # --- drop any rows missing *any* of the columns we actually created
    wanted = [ma_col, rsi_col, "MACD", "MACD_Signal"]
    present = [c for c in wanted if c in df.columns]
    if present:
        df = df.dropna(subset=present).reset_index(drop=True)

    return df

# ───────────────────────────── Build Composite Signals ─────────────────────────────
def build_composite(df):
    n     = len(df)
    close = df["Close"].to_numpy()
    ma    = df[f"MA{ma_window}"].to_numpy()
    rsi   = df[f"RSI{rsi_period}"].to_numpy()
    macd  = df["MACD"].to_numpy()
    sigl  = df["MACD_Signal"].to_numpy()

    df["MA_Signal"]   = 0
    df["RSI_Signal"]  = 0
    df["MACD_Signal2"]= 0
    df["Composite"]   = 0
    df["Trade"]       = 0

    for i in range(1, n):
        # MA crossover
        if close[i-1] < ma[i-1] and close[i] > ma[i]:
            df.at[i, "MA_Signal"] =  1
        elif close[i-1] > ma[i-1] and close[i] < ma[i]:
            df.at[i, "MA_Signal"] = -1

        # RSI thresholds
        if rsi[i] < 30:
            df.at[i, "RSI_Signal"] = 1
        elif rsi[i] > 70:
            df.at[i, "RSI_Signal"] = -1

        # MACD crossover
        if macd[i-1] < sigl[i-1] and macd[i] > sigl[i]:
            df.at[i, "MACD_Signal2"] = 1
        elif macd[i-1] > sigl[i-1] and macd[i] < sigl[i]:
            df.at[i, "MACD_Signal2"] = -1

        # Composite vote
        total = (
            df.at[i, "MA_Signal"] +
            df.at[i, "RSI_Signal"] +
            df.at[i, "MACD_Signal2"]
        )
        df.at[i, "Composite"] = total
        df.at[i, "Trade"]     = np.sign(total)

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

    return df, max_dd, sharpe, win_rt

# ───────────────────────────── Single‐Ticker Backtest ─────────────────────────────
st.markdown("### Single‐Ticker Backtest")
ticker = st.text_input("Ticker (e.g. AAPL)", "AAPL").upper()
if st.button("▶️ Run Composite Backtest"):
    df = load_and_compute(ticker, ma_window, rsi_period, macd_fast, macd_slow, macd_signal)
    if df.empty:
        st.error(f"No data for '{ticker}'.")
        st.stop()

    df = build_composite(df)
    df, max_dd, sharpe, win_rt = backtest(df)

    rec = {1:"🟢 BUY", 0:"🟡 HOLD", -1:"🔴 SELL"}[int(df["Trade"].iloc[-1])]
    st.success(f"**{ticker}**: {rec}")

    bh_ret    = (df["CumBH"].iloc[-1] - 1) * 100
    strat_ret = (df["CumStrat"].iloc[-1] - 1) * 100
    st.markdown(f"""
    **Buy & Hold:** {bh_ret:.2f}%  
    **Strategy:** {strat_ret:.2f}%  
    **Sharpe:** {sharpe:.2f}  
    **Max Drawdown:** {max_dd:.2f}%  
    **Win Rate:** {win_rt:.1f}%  
    """)

    # three‐panel chart
    fig, ax = plt.subplots(3, 1, figsize=(10,12), sharex=True)
    ax[0].plot(df["Close"], label="Close", color="blue")
    ax[0].plot(df[f"MA{ma_window}"], label=f"MA{ma_window}", color="orange")
    ax[0].set_title("Price & MA")
    ax[0].legend()

    ax[1].bar(df.index, df["Composite"], color="purple")
    ax[1].set_title("Composite Vote")

    ax[2].plot(df["CumBH"], ":",  label="Buy & Hold")
    ax[2].plot(df["CumStrat"], "-", label="Strategy")
    ax[2].set_title("Equity Curves")
    ax[2].legend()

    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

# ───────────────────────────── Batch Backtest ─────────────────────────────
st.markdown("---")
st.markdown("### Batch Backtest")
batch = st.text_area("Enter tickers (comma-separated)", "AAPL, MSFT, TSLA, SPY, QQQ").upper()
if st.button("▶️ Run Batch Backtest"):
    out = []
    for t in [x.strip() for x in batch.split(",") if x.strip()]:
        df_t = load_and_compute(t, ma_window, rsi_period, macd_fast, macd_slow, macd_signal)
        if df_t.empty: continue
        df_t = build_composite(df_t)
        df_t, max_dd, sharpe, win_rt = backtest(df_t)
        out.append({
            "Ticker": t,
            "Vote":    df_t["Composite"].iloc[-1],
            "Signal":  {1:"BUY",0:"HOLD",-1:"SELL"}[int(df_t["Trade"].iloc[-1])],
            "BuyHold%": (df_t["CumBH"].iloc[-1]-1)*100,
            "Strat%":   (df_t["CumStrat"].iloc[-1]-1)*100,
            "Sharpe":  sharpe,
            "MaxDD%":  max_dd,
            "Win%":    win_rt
        })
    if not out:
        st.error("No valid tickers/data.")
    else:
        dfp = pd.DataFrame(out).set_index("Ticker")
        st.dataframe(dfp)
        st.download_button("Download performance CSV", dfp.to_csv(), "batch_perf.csv")

# ───────────────────────────── Hyperparameter Grid Search ─────────────────────────────
st.markdown("---")
st.header("🔧 Hyperparameter Optimization")
ma_grid  = st.multiselect("MA windows to test",    [5,10,15,20,25],   default=[5,10,15])
rsi_grid = st.multiselect("RSI lookbacks to test", [7,14,21],        default=[14])

if st.button("🏃 Run Grid Search"):
    df_full = load_and_compute(ticker, ma_window, rsi_period, macd_fast, macd_slow, macd_signal)
    if df_full.empty:
        st.error(f"No data for {ticker}")
        st.stop()

    split    = len(df_full)//2
    df_train = df_full.iloc[:split].reset_index(drop=True)
    df_test  = df_full.iloc[split:].reset_index(drop=True)

    grid_out = []
    for ma in ma_grid:
      for rsi in rsi_grid:
        # compute & backtest in-sample
        df1 = df_train.copy()
        df1[f"MA{ma}"] = df1["Close"].rolling(ma).mean()
        d1 = df1.dropna(subset=[f"MA{ma}"]).reset_index(drop=True)
        d1["Signal"]   = np.sign(np.where(
            (d1["Close"].shift(1)<d1[f"MA{ma}"].shift(1))&(d1["Close"]>d1[f"MA{ma}"]),1,
            np.where((d1["Close"].shift(1)>d1[f"MA{ma}"].shift(1))&(d1["Close"]<d1[f"MA{ma}"]),-1,0)
        ))
        d1["Ret"]      = d1["Close"].pct_change().fillna(0)
        d1["Pos"]      = d1["Signal"].shift(1).fillna(0).clip(0,1)
        d1["StratRet"] = d1["Pos"]*d1["Ret"]
        sharpe_in     = d1["StratRet"].mean()/d1["StratRet"].std()*np.sqrt(252)

        # compute & backtest out-of-sample
        df2 = df_test.copy()
        df2[f"MA{ma}"] = df2["Close"].rolling(ma).mean()
        d2 = df2.dropna(subset=[f"MA{ma}"]).reset_index(drop=True)
        d2["Signal"]   = np.sign(np.where(
            (d2["Close"].shift(1)<d2[f"MA{ma}"].shift(1))&(d2["Close"]>d2[f"MA{ma}"]),1,
            np.where((d2["Close"].shift(1)>d2[f"MA{ma}"].shift(1))&(d2["Close"]<d2[f"MA{ma}"]),-1,0)
        ))
        d2["Ret"]      = d2["Close"].pct_change().fillna(0)
        d2["Pos"]      = d2["Signal"].shift(1).fillna(0).clip(0,1)
        d2["StratRet"] = d2["Pos"]*d2["Ret"]
        sharpe_out    = d2["StratRet"].mean()/d2["StratRet"].std()*np.sqrt(252)

        grid_out.append({
          "MA": ma, "RSI": rsi,
          "Sharpe_IN":  sharpe_in,
          "Sharpe_OOS": sharpe_out
        })

    gf = pd.DataFrame(grid_out).sort_values("Sharpe_OOS", ascending=False)
    st.dataframe(gf, use_container_width=True)
