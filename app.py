import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

# ───────────────────────────── Page Setup ─────────────────────────────
st.set_page_config(page_title="QuantaraX Composite Signals", layout="centered")

# ───────────────────────────── Tabs ─────────────────────────────
tab_engine, tab_help = st.tabs(["🚀 Engine", "❓ How It Works"])

with tab_help:
    st.header("How QuantaraX Works")
    st.markdown("""
**QuantaraX** combines three technical indicators into a single “composite” vote:

1. **Moving Average (MA) Crossover**  
   - Computes a simple MA over the last *N* days.  
   - **Bull** when price crosses *above* the MA, **Bear** when it crosses *below*.

2. **Relative Strength Index (RSI)**  
   - A momentum oscillator over a lookback period.  
   - **Bull** if RSI < 30 (oversold), **Bear** if RSI > 70 (overbought).

3. **MACD Crossover**  
   - Difference of two EMAs (fast & slow spans) plus a signal‐line EMA.  
   - **Bull** when MACD crosses above its signal line, **Bear** on the flip.

Each indicator issues +1 (bull), –1 (bear) or 0 (neutral). We sum these into a  
**Composite Vote** (–3…+3) and go long/flat based on its sign:

- Composite ≥ +1 → **BUY**  
- Composite ≤ –1 → **SELL**  
- Composite = 0  → **HOLD**

Below, under **Engine**, you can backtest a single ticker, batch test many,  
or grid‐search your MA/RSI/MACD parameters for the best simulated returns.
""")

with tab_engine:
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
    ma_window   = st.sidebar.slider("MA window",        5, 50, st.session_state["ma_window"],   key="ma_window")
    rsi_period  = st.sidebar.slider("RSI lookback",     5, 30, st.session_state["rsi_period"],  key="rsi_period")
    macd_fast   = st.sidebar.slider("MACD fast span",   5, 20, st.session_state["macd_fast"],   key="macd_fast")
    macd_slow   = st.sidebar.slider("MACD slow span",  20, 40, st.session_state["macd_slow"],   key="macd_slow")
    macd_signal = st.sidebar.slider("MACD signal span", 5, 20, st.session_state["macd_signal"], key="macd_signal")

    st.sidebar.markdown("---")
    st.sidebar.header("Grid-Search Parameters")
    ma_list   = st.sidebar.multiselect("MA windows",      [5,10,15,20,30], default=[ma_window],  key="grid_ma")
    rsi_list  = st.sidebar.multiselect("RSI lookbacks",   [7,14,21,28],   default=[rsi_period], key="grid_rsi")
    mf_list   = st.sidebar.multiselect("MACD fast spans", [8,12,16,20],   default=[macd_fast],  key="grid_mf")
    ms_list   = st.sidebar.multiselect("MACD slow spans", [20,26,32,40], default=[macd_slow],  key="grid_ms")
    sg_list   = st.sidebar.multiselect("MACD sig spans",  [5,9,12,16],    default=[macd_signal],key="grid_sig")

    st.title("🚀 QuantaraX — Composite Signal Engine")
    st.write("MA + RSI + MACD Composite Signals & Backtest")

    # ───────────────────────────── Load & Compute Indicators ─────────────────────────────
    @st.cache_data(show_spinner=False)
    def load_and_compute(ticker, ma_w, rsi_p, mf, ms, sig):
        df = yf.download(ticker, period="6mo", progress=False)
        if df.empty or "Close" not in df:
            return pd.DataFrame()

        # 1) MA
        ma_col = f"MA{ma_w}"
        df[ma_col] = df["Close"].rolling(ma_w).mean()

        # 2) RSI
        delta    = df["Close"].diff()
        up       = delta.clip(lower=0)
        down     = -delta.clip(upper=0)
        ema_up   = up.ewm(com=rsi_p-1, adjust=False).mean()
        ema_down = down.ewm(com=rsi_p-1, adjust=False).mean()
        rsi_col  = f"RSI{rsi_p}"
        df[rsi_col] = 100 - 100/(1 + ema_up/ema_down)

        # 3) MACD
        ema_f = df["Close"].ewm(span=mf, adjust=False).mean()
        ema_s = df["Close"].ewm(span=ms, adjust=False).mean()
        macd  = ema_f - ema_s
        df["MACD"]        = macd
        df["MACD_Signal"] = macd.ewm(span=sig, adjust=False).mean()

        # 4) Drop only the columns we created
        wanted = [ma_col, rsi_col, "MACD", "MACD_Signal"]
        present = [c for c in wanted if c in df.columns]
        if present:
            df = df.dropna(subset=present).reset_index(drop=True)

        return df

    # ───────────────────────────── Build Composite Signals ─────────────────────────────
    def build_composite(df, ma_w, rsi_p):
        n         = len(df)
        close_arr = df["Close"].to_numpy()
        ma_arr    = df[f"MA{ma_w}"].to_numpy()
        rsi_arr   = df[f"RSI{rsi_p}"].to_numpy()
        macd_arr  = df["MACD"].to_numpy()
        sig_arr   = df["MACD_Signal"].to_numpy()

        ma_sig    = np.zeros(n, dtype=int)
        rsi_sig   = np.zeros(n, dtype=int)
        macd_sig2 = np.zeros(n, dtype=int)
        comp      = np.zeros(n, dtype=int)
        trade     = np.zeros(n, dtype=int)

        for i in range(1, n):
            if close_arr[i-1] < ma_arr[i-1] and close_arr[i] > ma_arr[i]:
                ma_sig[i] =  1
            elif close_arr[i-1] > ma_arr[i-1] and close_arr[i] < ma_arr[i]:
                ma_sig[i] = -1

            if rsi_arr[i] < 30:
                rsi_sig[i] =  1
            elif rsi_arr[i] > 70:
                rsi_sig[i] = -1

            if macd_arr[i-1] < sig_arr[i-1] and macd_arr[i] > sig_arr[i]:
                macd_sig2[i] =  1
            elif macd_arr[i-1] > sig_arr[i-1] and macd_arr[i] < sig_arr[i]:
                macd_sig2[i] = -1

            comp[i]  = ma_sig[i] + rsi_sig[i] + macd_sig2[i]
            trade[i] = np.sign(comp[i])

        df["Composite"]    = comp
        df["Trade"]        = trade
        return df

    # ───────────────────────────── Backtest & Metrics ─────────────────────────────
    def backtest(df):
        df = df.copy()
        df["Return"]   = df["Close"].pct_change().fillna(0)
        df["Position"] = df["Trade"].shift(1).fillna(0).clip(0,1)
        df["StratRet"] = df["Position"] * df["Return"]
        df["CumBH"]    = (1 + df["Return"]).cumprod()
        df["CumStrat"] = (1 + df["StratRet"]).cumprod()

        dd      = df["CumStrat"] / df["CumStrat"].cummax() - 1
        max_dd  = dd.min() * 100
        std_dev = df["StratRet"].std()
        sharpe  = (df["StratRet"].mean() / std_dev * np.sqrt(252)) if std_dev else np.nan
        win_rt  = (df["StratRet"] > 0).mean() * 100

        return df, max_dd, sharpe, win_rt

    # ───────────────────────────── Single‐Ticker Backtest ─────────────────────────────
    st.markdown("## Single‐Ticker Backtest")
    ticker = st.text_input("Ticker (e.g. AAPL)", "AAPL").upper()

    if st.button("▶️ Run Composite Backtest"):
        df = load_and_compute(ticker, ma_window, rsi_period, macd_fast, macd_slow, macd_signal)
        if df.empty:
            st.error(f"No data for '{ticker}'.")
            st.stop()

        df = build_composite(df, ma_window, rsi_period)
        df, max_dd, sharpe, win_rt = backtest(df)

        rec_map = { 1:"🟢 BUY", 0:"🟡 HOLD", -1:"🔴 SELL" }
        st.success(f"**{ticker}**: {rec_map[int(df['Trade'].iloc[-1])]}")
        bh_ret    = (df["CumBH"].iloc[-1] - 1) * 100
        strat_ret = (df["CumStrat"].iloc[-1] - 1) * 100
        st.markdown(f"- **B&H:** {bh_ret:.2f}%   - **Strat:** {strat_ret:.2f}%   - **Sharpe:** {sharpe:.2f}%   - **MaxDD:** {max_dd:.2f}%   - **Win%:** {win_rt:.1f}%")

        fig, axes = plt.subplots(3,1,figsize=(10,12),sharex=True)
        axes[0].plot(df["Close"], label="Close"); axes[0].plot(df[f"MA{ma_window}"], label=f"MA{ma_window}"); axes[0].legend(); axes[0].set_title("Price & MA")
        axes[1].bar(df.index, df["Composite"], color="purple"); axes[1].set_title("Composite Vote")
        axes[2].plot(df["CumBH"], ":", label="B&H"); axes[2].plot(df["CumStrat"], "-", label="Strategy"); axes[2].legend(); axes[2].set_title("Equity Curves")
        plt.xticks(rotation=45); plt.tight_layout()
        st.pyplot(fig)

    # ───────────────────────────── Batch Backtest ─────────────────────────────
    st.markdown("---")
    st.markdown("## Batch Backtest")
    batch = st.text_area("Enter tickers (comma-separated)", "AAPL, MSFT, TSLA, SPY, QQQ").upper()

    if st.button("▶️ Run Batch Backtest"):
        perf = []
        for t in [s.strip() for s in batch.split(",") if s.strip()]:
            df_t = load_and_compute(t, ma_window, rsi_period, macd_fast, macd_slow, macd_signal)
            if df_t.empty:
                continue
            df_t = build_composite(df_t, ma_window, rsi_period)
            df_t, max_dd, sharpe, win_rt = backtest(df_t)
            perf.append({
                "Ticker":     t,
                "Composite":  int(df_t["Composite"].iloc[-1]),
                "Signal":     {1:"BUY",0:"HOLD",-1:"SELL"}[int(df_t["Trade"].iloc[-1])],
                "BuyHold %":  (df_t["CumBH"].iloc[-1]-1)*100,
                "Strategy %": (df_t["CumStrat"].iloc[-1]-1)*100,
                "Sharpe":     sharpe,
                "MaxDD %":    max_dd,
                "Win Rate %": win_rt
            })
        if not perf:
            st.error("No valid tickers/data.")
        else:
            df_perf = pd.DataFrame(perf).set_index("Ticker")
            st.dataframe(df_perf)
            st.download_button("Download CSV", df_perf.to_csv(), "batch_perf.csv")

    # ───────────────────────────── Hyperparameter Optimization ─────────────────────────────
    st.markdown("---")
    st.markdown("## Hyperparameter Optimization")
    if st.button("🏃‍♂️ Run Grid Search"):
        if not ticker:
            st.error("Enter a ticker above.")
            st.stop()
        df_base = load_and_compute(ticker, ma_window, rsi_period, macd_fast, macd_slow, macd_signal)
        if df_base.empty:
            st.error(f"No data for {ticker}")
            st.stop()
        results = []
        with st.spinner("Testing combos…"):
            for ma_w in ma_list:
                for rsi_p in rsi_list:
                    for mf in mf_list:
                        for ms in ms_list:
                            for sg in sg_list:
                                df_i = load_and_compute(ticker, ma_w, rsi_p, mf, ms, sg)
                                if df_i.empty:
                                    continue
                                df_i = build_composite(df_i, ma_w, rsi_p)
                                df_i, max_dd, sharpe_i, win_rt_i = backtest(df_i)
                                strat_pct = (df_i["CumStrat"].iloc[-1] - 1) * 100
                                results.append({
                                    "MA": ma_w, "RSI": rsi_p,
                                    "MF": mf, "MS": ms, "SG": sg,
                                    "Strat %": strat_pct,
                                    "Sharpe": sharpe_i,
                                    "MaxDD %": max_dd,
                                    "Win %": win_rt_i
                                })
        if not results:
            st.error("No valid parameter combos.")
        else:
            df_grid = pd.DataFrame(results).sort_values("Strat %", ascending=False).reset_index(drop=True)
            st.dataframe(df_grid.head(10), use_container_width=True)
            st.download_button("Download CSV", df_grid.to_csv(index=False), "grid_search.csv")
