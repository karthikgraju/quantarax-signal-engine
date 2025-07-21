import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ───────────────────────────── Page Configuration ─────────────────────────────
st.set_page_config(page_title="QuantaraX Composite Signals", layout="centered")

# ───────────────────────────── Tabs ─────────────────────────────
tab_engine, tab_help = st.tabs(["🚀 Engine", "❓ How It Works"])

with tab_help:
    st.header("How QuantaraX Works")
    st.markdown("""
**QuantaraX** combines three technical indicators into a single composite vote:

1. **Moving Average Crossover**  
   - Simple N-day MA. Bull when price crosses above, bear when it crosses below.

2. **RSI (Relative Strength Index)**  
   - Momentum oscillator. Bull if RSI < 30 (oversold), bear if RSI > 70 (overbought).

3. **MACD Crossover**  
   - Difference of fast & slow EMA + signal line. Bull on MACD line crossing above its signal, bear on crossing below.

Each indicator outputs +1 (bull), –1 (bear), or 0 (neutral). Summing them gives a **Composite** vote (–3…+3).  
We then take a position via:

- Composite ≥ +1 → **BUY**  
- Composite = 0 → **HOLD**  
- Composite ≤ –1 → **SELL**

Under **Engine** you can:
- Backtest a single ticker  
- Batch-test many tickers  
- Grid-search optimal indicator windows  

Happy trading!  
—KG
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

    # ───────────────────────────── Page Title ─────────────────────────────
    st.title("🚀 QuantaraX — Composite Signal Engine")
    st.write("MA + RSI + MACD Composite Signals & Backtest")

    # ───────────────────────────── Load & Compute Indicators ─────────────────────────────
    @st.cache_data(show_spinner=False)
    def load_and_compute(ticker, ma_w, rsi_p, mf, ms, sig):
        df = yf.download(ticker, period="6mo", progress=False)
        if df.empty or "Close" not in df:
            return pd.DataFrame()

        # 1) Moving Average
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
        ema_f    = df["Close"].ewm(span=mf, adjust=False).mean()
        ema_s    = df["Close"].ewm(span=ms, adjust=False).mean()
        macd     = ema_f - ema_s
        macd_sig = macd.ewm(span=sig, adjust=False).mean()
        df["MACD"]        = macd
        df["MACD_Signal"] = macd_sig

        # 4) Drop rows missing any of our indicator columns
        wanted  = [ma_col, rsi_col, "MACD", "MACD_Signal"]
        present = [c for c in wanted if c in df.columns]
        if present:
            try:
                df = df.dropna(subset=present).reset_index(drop=True)
            except KeyError:
                pass

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
            # MA crossover signal
            if close_arr[i-1] < ma_arr[i-1] and close_arr[i] > ma_arr[i]:
                ma_sig[i] =  1
            elif close_arr[i-1] > ma_arr[i-1] and close_arr[i] < ma_arr[i]:
                ma_sig[i] = -1
            # RSI signal
            if rsi_arr[i] < 30:
                rsi_sig[i] =  1
            elif rsi_arr[i] > 70:
                rsi_sig[i] = -1
            # MACD crossover
            if macd_arr[i-1] < sig_arr[i-1] and macd_arr[i] > sig_arr[i]:
                macd_sig2[i] =  1
            elif macd_arr[i-1] > sig_arr[i-1] and macd_arr[i] < sig_arr[i]:
                macd_sig2[i] = -1
            # Composite vote and trade decision
            comp[i]  = ma_sig[i] + rsi_sig[i] + macd_sig2[i]
            trade[i] = np.sign(comp[i])

        df["MA_Signal"]    = ma_sig
        df["RSI_Signal"]   = rsi_sig
        df["MACD_Signal2"] = macd_sig2
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
        sharpe  = (df["StratRet"].mean() / std_dev * np.sqrt(252)) if std_dev != 0 else np.nan
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

        df, max_dd, sharpe, win_rt = (
            *backtest(build_composite(df, ma_window, rsi_period)), 
            max_dd, sharpe, win_rt
        ) = backtest(build_composite(df, ma_window, rsi_period))

        rec_map    = {1:"🟢 BUY", 0:"🟡 HOLD", -1:"🔴 SELL"}
        last_trade = int(df["Trade"].iloc[-1])
        st.success(f"**{ticker}**: {rec_map[last_trade]}")

        bh_ret    = (df["CumBH"].iloc[-1] - 1) * 100
        strat_ret = (df["CumStrat"].iloc[-1] - 1) * 100
        st.markdown(f"""
- **Buy & Hold:**    {bh_ret:.2f}%  
- **Strategy:**      {strat_ret:.2f}%  
- **Sharpe:**        {sharpe:.2f}  
- **Max Drawdown:**  {max_dd:.2f}%  
- **Win Rate:**      {win_rt:.1f}%  
        """)

        fig, axes = plt.subplots(3,1,figsize=(10,12), sharex=True)
        axes[0].plot(df["Close"], label="Close")
        axes[0].plot(df[f"MA{ma_window}"], label=f"MA{ma_window}")
        axes[0].set_title("Price & MA"); axes[0].legend()
        axes[1].bar(df.index, df["Composite"], color="purple")
        axes[1].set_title("Composite Vote")
        axes[2].plot(df["CumBH"], ":", label="Buy & Hold")
        axes[2].plot(df["CumStrat"], "-", label="Strategy")
        axes[2].set_title("Equity Curves"); axes[2].legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

    # ───────────────────────────── Batch Backtest ─────────────────────────────
    st.markdown("---")
    st.markdown("## Batch Backtest")
    batch = st.text_area(
        "Enter tickers (comma-separated)", 
        "AAPL, MSFT, TSLA, SPY, QQQ"
    ).upper()

    if st.button("▶️ Run Batch Backtest"):
        perf = []
        for t in [s.strip() for s in batch.split(",") if s.strip()]:
            df_t = load_and_compute(t, ma_window, rsi_period, macd_fast, macd_slow, macd_signal)
            if df_t.empty:
                continue
            df_t, max_dd, sharpe, win_rt = backtest(build_composite(df_t, ma_window, rsi_period))
            perf.append({
                "Ticker":        t,
                "Composite":     int(df_t["Composite"].iloc[-1]),
                "Signal":        {1:"BUY",0:"HOLD",-1:"SELL"}[int(df_t["Trade"].iloc[-1])],
                "BuyHold %":     (df_t["CumBH"].iloc[-1] - 1)*100,
                "Strategy %":    (df_t["CumStrat"].iloc[-1] - 1)*100,
                "Sharpe":        sharpe,
                "Max Drawdown %":max_dd,
                "Win Rate %":    win_rt
            })

        if not perf:
            st.error("No valid tickers/data.")
        else:
            df_perf = pd.DataFrame(perf).set_index("Ticker")
            st.dataframe(df_perf)
            st.download_button(
                "Download performance CSV", 
                df_perf.to_csv(), 
                "batch_perf.csv"
            )

    # ───────────────────────────── Hyperparameter Optimization ─────────────────────────────
    st.markdown("---")
    st.markdown("## 🛠️ Hyperparameter Optimization")

    # Candidate lists
    ma_options  = [5, 10, 15, 20, 30]
    rsi_options = [7, 14, 21, 28]
    mf_options  = [8, 12, 16, 20]
    ms_options  = [20, 26, 32, 40]
    sig_options = [5, 9, 12, 16]

    # Ensure slider default is in the multiselect choices
    default_ma  = [ma_window]   if ma_window   in ma_options  else []
    default_rsi = [rsi_period]  if rsi_period  in rsi_options else []
    default_mf  = [macd_fast]   if macd_fast   in mf_options  else []
    default_ms  = [macd_slow]   if macd_slow   in ms_options  else []
    default_sig = [macd_signal] if macd_signal in sig_options else []

    ma_list  = st.sidebar.multiselect(
        "MA windows to test", ma_options, default=default_ma, key="grid_ma"
    )
    rsi_list = st.sidebar.multiselect(
        "RSI lookbacks to test", rsi_options, default=default_rsi, key="grid_rsi"
    )
    mf_list  = st.sidebar.multiselect(
        "MACD fast spans to test", mf_options, default=default_mf, key="grid_mf"
    )
    ms_list  = st.sidebar.multiselect(
        "MACD slow spans to test", ms_options, default=default_ms, key="grid_ms"
    )
    sig_list = st.sidebar.multiselect(
        "MACD sig spans to test", sig_options, default=default_sig, key="grid_sig"
    )

    if st.button("🏃‍♂️ Run Grid Search"):
        if not ticker:
            st.error("Enter a ticker above.")
            st.stop()

        df_full = load_and_compute(
            ticker, ma_window, rsi_period, macd_fast, macd_slow, macd_signal
        )
        if df_full.empty:
            st.error(f"No data for {ticker}")
            st.stop()

        results = []
        with st.spinner("Testing combinations..."):
            for ma_w in ma_list:
                for rsi_p in rsi_list:
                    for mf in mf_list:
                        for ms in ms_list:
                            for s in sig_list:
                                df_i = load_and_compute(ticker, ma_w, rsi_p, mf, ms, s)
                                if df_i.empty:
                                    continue
                                df_i, max_dd, sharpe_i, win_rt_i = backtest(
                                    build_composite(df_i, ma_w, rsi_p)
                                )
                                strat_ret = (df_i["CumStrat"].iloc[-1] - 1) * 100
                                results.append({
                                    "MA": ma_w,
                                    "RSI": rsi_p,
                                    "MACD_Fast": mf,
                                    "MACD_Slow": ms,
                                    "MACD_Sig": s,
                                    "Strategy %": strat_ret,
                                    "Sharpe": sharpe_i,
                                    "MaxDD %": max_dd,
                                    "Win %": win_rt_i
                                })

        if not results:
            st.error("No valid parameter combos.")
        else:
            df_grid = (
                pd.DataFrame(results)
                  .sort_values("Strategy %", ascending=False)
                  .reset_index(drop=True)
            )
            st.dataframe(df_grid.head(10), use_container_width=True)
            csv = df_grid.to_csv(index=False)
            st.download_button("Download full grid CSV", csv, "grid_search.csv")
