import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ───────────────────────────── Page Setup ─────────────────────────────
st.set_page_config(page_title="QuantaraX Composite Signals", layout="centered")
analyzer = SentimentIntensityAnalyzer()

# ───────────────────────────── Tabs ─────────────────────────────
tab_engine, tab_help = st.tabs(["🚀 Engine", "❓ How It Works"])

# ───────────────────────────── Help Tab ─────────────────────────────
with tab_help:
    st.header("How QuantaraX Works")
    st.markdown("""
**QuantaraX** combines three indicators into a single composite vote:

1. **Moving Average Crossover**  
   - Simple MA over *N* days. Bull when price crosses above, bear when below.

2. **RSI**  
   - Momentum oscillator. Bull if RSI < 30 (oversold), bear if RSI > 70 (overbought).

3. **MACD Crossover**  
   - EMA fast vs. slow difference + signal line. Bull on crossover up, bear on crossover down.

Each gives +1 (bull), –1 (bear), or 0 (neutral). Sum (–3…+3) → **Composite**.  
Position = sign(composite):  
• ≥+1 → BUY  
• =0  → HOLD  
• ≤–1 → SELL

Under **Engine** you can:  
• Backtest a single ticker  
• Batch-test many tickers  
• Grid-search your parameters  
• View a watchlist summary with reasoning  
• See recent news & sentiment overlay  
""")

# ───────────────────────────── Engine Tab ─────────────────────────────
with tab_engine:

    # Defaults & Session State
    DEFAULTS = dict(ma_window=10, rsi_period=14, macd_fast=12, macd_slow=26, macd_signal=9)
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Sidebar controls
    st.sidebar.header("Controls")
    if st.sidebar.button("🔄 Reset to defaults"):
        for k, v in DEFAULTS.items():
            st.session_state[k] = v

    st.sidebar.header("Indicator Parameters")
    ma_window   = st.sidebar.slider("MA window",      5, 50, st.session_state["ma_window"],   key="ma_window")
    rsi_period  = st.sidebar.slider("RSI lookback",   5, 30, st.session_state["rsi_period"],  key="rsi_period")
    macd_fast   = st.sidebar.slider("MACD fast span", 5, 20, st.session_state["macd_fast"],   key="macd_fast")
    macd_slow   = st.sidebar.slider("MACD slow span",20, 40, st.session_state["macd_slow"],   key="macd_slow")
    macd_signal = st.sidebar.slider("MACD sig span",  5, 20, st.session_state["macd_signal"], key="macd_signal")

    st.title("🚀 QuantaraX — Composite Signal Engine")
    st.write("MA + RSI + MACD Composite Signals & Backtest")

    # ─────────── Data Loading ───────────
    @st.cache_data(show_spinner=False)
    def load_and_compute(ticker, ma_w, rsi_p, mf, ms, sig):
        df = yf.download(ticker, period="6mo", progress=False)
        if df.empty or "Close" not in df:
            return pd.DataFrame()

        # Moving Average
        ma_col = f"MA{ma_w}"
        df[ma_col] = df["Close"].rolling(ma_w).mean()

        # RSI
        d       = df["Close"].diff()
        up      = d.clip(lower=0)
        dn      = -d.clip(upper=0)
        ema_up  = up.ewm(com=rsi_p-1, adjust=False).mean()
        ema_dn  = dn.ewm(com=rsi_p-1, adjust=False).mean()
        rsi_col = f"RSI{rsi_p}"
        df[rsi_col] = 100 - 100/(1 + ema_up/ema_dn)

        # MACD
        ema_f    = df["Close"].ewm(span=mf, adjust=False).mean()
        ema_s    = df["Close"].ewm(span=ms, adjust=False).mean()
        macd     = ema_f - ema_s
        macd_sig = macd.ewm(span=sig, adjust=False).mean()
        df["MACD"]        = macd
        df["MACD_Signal"] = macd_sig

        # Safely drop any NA rows in our new columns
        intended = [ma_col, rsi_col, "MACD", "MACD_Signal"]
        present  = [c for c in intended if c in df.columns]
        if present:
            df = df.dropna(subset=present).reset_index(drop=True)

        return df

    # ─────────── Composite Signals ───────────
    def build_composite(df, ma_w, rsi_p):
        n         = len(df)
        close     = df["Close"].to_numpy()
        ma_arr    = df[f"MA{ma_w}"].to_numpy()
        rsi_arr   = df[f"RSI{rsi_p}"].to_numpy()
        macd_arr  = df["MACD"].to_numpy()
        sig_arr   = df["MACD_Signal"].to_numpy()

        ma_sig    = np.zeros(n, int)
        rsi_sig   = np.zeros(n, int)
        macd_sig2 = np.zeros(n, int)
        comp      = np.zeros(n, int)
        trade     = np.zeros(n, int)

        for i in range(1, n):
            # MA Crossover
            if close[i-1] < ma_arr[i-1] and close[i] > ma_arr[i]:
                ma_sig[i] = 1
            elif close[i-1] > ma_arr[i-1] and close[i] < ma_arr[i]:
                ma_sig[i] = -1

            # RSI thresholds
            if rsi_arr[i] < 30:
                rsi_sig[i] = 1
            elif rsi_arr[i] > 70:
                rsi_sig[i] = -1

            # MACD crossover
            if macd_arr[i-1] < sig_arr[i-1] and macd_arr[i] > sig_arr[i]:
                macd_sig2[i] = 1
            elif macd_arr[i-1] > sig_arr[i-1] and macd_arr[i] < sig_arr[i]:
                macd_sig2[i] = -1

            comp[i]  = ma_sig[i] + rsi_sig[i] + macd_sig2[i]
            trade[i] = np.sign(comp[i])

        df["MA_Signal"]    = ma_sig
        df["RSI_Signal"]   = rsi_sig
        df["MACD_Signal2"] = macd_sig2
        df["Composite"]    = comp
        df["Trade"]        = trade
        return df

    # ─────────── Backtest ───────────
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

    # ─────────────────── Single‐Ticker Backtest ───────────────────
    st.markdown("## Single‐Ticker Backtest")
    ticker = st.text_input("Ticker (e.g. AAPL)", "AAPL").upper()

    # Live Price & News
    if ticker:
        info  = yf.Ticker(ticker).info
        live  = info.get("regularMarketPrice")
        if live is not None:
            st.subheader(f"💲 Live Price: ${live:.2f}")

        news_items = getattr(yf.Ticker(ticker), "news", []) or []
        if news_items:
            st.markdown("### 📰 Recent News & Sentiment")
            shown = 0
            for art in news_items:
                title = art.get("title", "").strip()
                link  = art.get("link", "").strip()
                if not title or not link:
                    continue
                text  = art.get("summary", title)
                score = analyzer.polarity_scores(text)["compound"]
                emoji = "🔺" if score > 0.1 else ("🔻" if score < -0.1 else "➖")
                st.markdown(f"- [{title}]({link}) {emoji}")
                shown += 1
                if shown >= 5:
                    break
            if shown == 0:
                st.info("No recent news found.")
        else:
            st.info("No recent news found.")

    if st.button("▶️ Run Composite Backtest"):
        df_raw = load_and_compute(ticker, ma_window, rsi_period, macd_fast, macd_slow, macd_signal)
        if df_raw.empty:
            st.error(f"No data for '{ticker}'")
            st.stop()

        df_c, max_dd, sharpe, win_rt = backtest(build_composite(df_raw, ma_window, rsi_period))

        last = int(df_c["Trade"].iloc[-1])
        rec  = {1:"🟢 BUY", 0:"🟡 HOLD", -1:"🔴 SELL"}[last]
        st.success(f"**{ticker}**: {rec}")

        # Explain
        ma_s   = int(df_c["MA_Signal"].iloc[-1])
        rsi_s  = int(df_c["RSI_Signal"].iloc[-1])
        macd_s = int(df_c["MACD_Signal2"].iloc[-1])
        rsi_v  = df_c[f"RSI{rsi_period}"].iloc[-1]

        ma_texts = {
            1: f"Price crossed **above** its {ma_window}-day MA.",
            0: "No MA crossover.",
           -1: f"Price crossed **below** its {ma_window}-day MA."
        }
        rsi_texts = {
            1: f"RSI ({rsi_v:.1f}) < 30 → oversold.",
            0: f"RSI ({rsi_v:.1f}) neutral.",
           -1: f"RSI ({rsi_v:.1f}) > 70 → overbought."
        }
        macd_texts = {
            1: "MACD line crossed **above** its signal line.",
            0: "No MACD crossover.",
           -1: "MACD line crossed **below** its signal line."
        }

        with st.expander("🔎 Why This Signal?"):
            st.write(f"- **MA Signal:**    {ma_texts[ma_s]}")
            st.write(f"- **RSI Signal:**   {rsi_texts[rsi_s]}")
            st.write(f"- **MACD Signal:**  {macd_texts[macd_s]}")
            st.write(f"- **Composite:**    {df_c['Composite'].iloc[-1]}")

        bh_ret = (df_c["CumBH"].iloc[-1] - 1) * 100
        st_ret = (df_c["CumStrat"].iloc[-1] - 1) * 100
        st.markdown(f"""
- **Buy & Hold:**    {bh_ret:.2f}%  
- **Strategy:**      {st_ret:.2f}%  
- **Sharpe:**        {sharpe:.2f}  
- **Max Drawdown:**  {max_dd:.2f}%  
- **Win Rate:**      {win_rt:.1f}%  
""")

        fig, axs = plt.subplots(3,1,figsize=(10,12), sharex=True)
        axs[0].plot(df_c["Close"], label="Close")
        axs[0].plot(df_c[f"MA{ma_window}"], label=f"MA{ma_window}")
        axs[0].set_title("Price & MA");       axs[0].legend()
        axs[1].bar(df_c.index, df_c["Composite"], color="purple"); axs[1].set_title("Composite Vote")
        axs[2].plot(df_c["CumBH"], ":", label="Buy & Hold")
        axs[2].plot(df_c["CumStrat"], "-", label="Strategy")
        axs[2].set_title("Equity Curves");    axs[2].legend()
        plt.xticks(rotation=45); plt.tight_layout(); st.pyplot(fig)

    # ───────────────────────────── Batch Backtest ─────────────────────────────
    st.markdown("---")
    st.markdown("## Batch Backtest")
    batch = st.text_area("Tickers (comma-separated)", "AAPL, MSFT, TSLA, SPY, QQQ").upper()

    if st.button("▶️ Run Batch Backtest"):
        perf = []
        for t in [x.strip() for x in batch.split(",") if x.strip()]:
            df_t = load_and_compute(t, ma_window, rsi_period, macd_fast, macd_slow, macd_signal)
            if df_t.empty:
                continue
            df_c, md, sh, wr = backtest(build_composite(df_t, ma_window, rsi_period))
            perf.append({
                "Ticker":      t,
                "Composite":   int(df_c["Composite"].iloc[-1]),
                "Signal":      {1:"BUY",0:"HOLD",-1:"SELL"}[int(df_c["Trade"].iloc[-1])],
                "BuyHold %":   (df_c["CumBH"].iloc[-1] - 1) * 100,
                "Strategy %":  (df_c["CumStrat"].iloc[-1] - 1) * 100,
                "Sharpe":      sh,
                "MaxDD %":     md,
                "Win Rate %":  wr
            })

        if perf:
            df_perf = pd.DataFrame(perf).set_index("Ticker")
            st.dataframe(df_perf, use_container_width=True)
            st.download_button("Download CSV", df_perf.to_csv(), "batch.csv")
        else:
            st.error("No valid data.")

    # ───────────────────────────── Hyperparameter Optimization ─────────────────────────────
    st.markdown("---")
    st.markdown("## 🛠️ Hyperparameter Optimization")
    ma_list  = st.sidebar.multiselect("MA windows",     [5,10,15,20,30], default=[ma_window], key="grid_ma")
    rsi_list = st.sidebar.multiselect("RSI lookbacks",  [7,14,21,28],   default=[rsi_period], key="grid_rsi")
    mf_list  = st.sidebar.multiselect("MACD fast spans",[8,12,16,20],  default=[macd_fast],   key="grid_mf")
    ms_list  = st.sidebar.multiselect("MACD slow spans",[20,26,32,40],default=[macd_slow],  key="grid_ms")
    sig_list = st.sidebar.multiselect("MACD sig spans", [5,9,12,16],   default=[macd_signal],key="grid_sig")

    if st.button("🏃‍♂️ Run Grid Search"):
        if not ticker:
            st.error("Enter a ticker."); st.stop()

        df_full = load_and_compute(ticker, ma_window, rsi_period, macd_fast, macd_slow, macd_signal)
        if df_full.empty:
            st.error(f"No data for '{ticker}'"); st.stop()

        results = []
        with st.spinner("Testing parameter combos…"):
            for mw in ma_list:
                for rp in rsi_list:
                    for mf in mf_list:
                        for ms in ms_list:
                            for s in sig_list:
                                df_i = load_and_compute(ticker, mw, rp, mf, ms, s)
                                if df_i.empty:
                                    continue
                                df_c_i, md_i, sh_i, wr_i = backtest(build_composite(df_i, mw, rp))
                                rtn = (df_c_i["CumStrat"].iloc[-1] - 1) * 100
                                results.append({
                                    "MA": mw, "RSI": rp,
                                    "MACD_Fast": mf, "MACD_Slow": ms, "MACD_Sig": s,
                                    "Return %": rtn, "Sharpe": sh_i,
                                    "MaxDD %": md_i, "Win %": wr_i
                                })

        if results:
            df_grid = pd.DataFrame(results) \
                        .sort_values("Return %", ascending=False) \
                        .head(10)
            st.dataframe(df_grid, use_container_width=True)
            st.download_button("Download full CSV", df_grid.to_csv(index=False), "grid.csv")
        else:
            st.error("No valid parameter combinations.")

    # ───────────────────────────── Watchlist Summary ─────────────────────────────
    st.markdown("---")
    st.markdown("## ⏰ Watchlist Summary")
    watch = st.text_area("Enter tickers (comma-separated)", "AAPL, MSFT, TSLA, SPY, QQQ").upper()

    if st.button("📬 Generate Watchlist Summary"):
        table = []
        for t in [x.strip() for x in watch.split(",") if x.strip()]:
            df_t = load_and_compute(t, ma_window, rsi_period, macd_fast, macd_slow, macd_signal)
            if df_t.empty:
                table.append({"Ticker": t, "Composite": None, "Signal": "N/A"})
            else:
                df_c, *_ = backtest(build_composite(df_t, ma_window, rsi_period))
                comp     = int(df_c["Composite"].iloc[-1])
                sig      = {1:"BUY",0:"HOLD",-1:"SELL"}[int(df_c["Trade"].iloc[-1])]
                table.append({"Ticker": t, "Composite": comp, "Signal": sig})

        df_watch = pd.DataFrame(table).set_index("Ticker")
        st.dataframe(df_watch, use_container_width=True)

        for t in df_watch.index:
            df_t = load_and_compute(t, ma_window, rsi_period, macd_fast, macd_slow, macd_signal)
            if df_t.empty:
                continue
            df_c = build_composite(df_t, ma_window, rsi_period)
            last = df_c.iloc[-1]

            ma_s   = int(last["MA_Signal"])
            rsi_s  = int(last["RSI_Signal"])
            macd_s = int(last["MACD_Signal2"])

            rsi_v  = last.get(f"RSI{rsi_period}", np.nan)
            try:
                rsi_val = float(rsi_v)
            except:
                rsi_val = None

            texts = {
                "MA":   {1:"crossed ↑ MA", 0:"no crossover", -1:"crossed ↓ MA"},
                "RSI":  {1:"<30 (oversold)", 0:"30–70", -1:">70 (overbought)"},
                "MACD": {1:"crossed ↑ signal", 0:"no crossover", -1:"crossed ↓ signal"}
            }

            with st.expander(f"🔎 {t} Reasoning ({df_watch.loc[t,'Signal']})"):
                st.write(f"- **MA:**   Price {texts['MA'][ma_s]}.")
                if rsi_val is not None:
                    st.write(f"- **RSI:**  {rsi_val:.1f} {texts['RSI'][rsi_s]}.")
                else:
                    st.write(f"- **RSI:**  Data unavailable.")
                st.write(f"- **MACD:** {texts['MACD'][macd_s]}.")
                st.write(f"- **Composite:** {df_watch.loc[t,'Composite']}")
