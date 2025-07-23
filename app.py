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
2. **RSI**  
3. **MACD Crossover**  

Each gives +1/0/–1 → **Composite** → BUY/HOLD/SELL.

Under **Engine** you can:
- Backtest a single ticker  
- Batch-test many tickers  
- Grid-search parameters  
- Watchlist summary  
- Recent news & sentiment overlay  
- **Portfolio Simulator**  
""")

# ───────────────────────────── Engine Tab ─────────────────────────────
with tab_engine:

    # Defaults & Session State
    DEFAULTS = dict(ma_window=10, rsi_period=14, macd_fast=12, macd_slow=26, macd_signal=9)
    for k,v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Sidebar controls
    st.sidebar.header("Controls")
    if st.sidebar.button("🔄 Reset to defaults"):
        for k,v in DEFAULTS.items():
            st.session_state[k] = v

    st.sidebar.header("Indicator Parameters")
    ma_window   = st.sidebar.slider("MA window",      5,50, st.session_state["ma_window"],   key="ma_window")
    rsi_period  = st.sidebar.slider("RSI lookback",   5,30, st.session_state["rsi_period"],  key="rsi_period")
    macd_fast   = st.sidebar.slider("MACD fast span", 5,20, st.session_state["macd_fast"],   key="macd_fast")
    macd_slow   = st.sidebar.slider("MACD slow span",20,40, st.session_state["macd_slow"],   key="macd_slow")
    macd_signal = st.sidebar.slider("MACD sig span",  5,20, st.session_state["macd_signal"], key="macd_signal")

    st.title("🚀 QuantaraX — Composite Signal Engine")
    st.write("MA + RSI + MACD Composite Signals & Backtest")

    # ─────────── Data Loading ───────────
    @st.cache_data(show_spinner=False)
    def load_and_compute(ticker, ma_w, rsi_p, mf, ms, sig):
        df = yf.download(ticker, period="6mo", progress=False)
        if df.empty or "Close" not in df:
            return pd.DataFrame()
        # MA
        df[f"MA{ma_w}"] = df["Close"].rolling(ma_w).mean()
        # RSI
        delta = df["Close"].diff()
        up    = delta.clip(lower=0)
        dn    = -delta.clip(upper=0)
        ema_up   = up.ewm(com=rsi_p-1, adjust=False).mean()
        ema_down = dn.ewm(com=rsi_p-1, adjust=False).mean()
        df[f"RSI{rsi_p}"] = 100 - 100/(1 + ema_up/ema_down)
        # MACD
        ema_f = df["Close"].ewm(span=mf, adjust=False).mean()
        ema_s = df["Close"].ewm(span=ms, adjust=False).mean()
        macd  = ema_f - ema_s
        df["MACD"]        = macd
        df["MACD_Signal"] = macd.ewm(span=sig, adjust=False).mean()
        # Drop NAs
        cols = [f"MA{ma_w}", f"RSI{rsi_p}", "MACD", "MACD_Signal"]
        df = df.dropna(subset=[c for c in cols if c in df], how="any")
        return df.reset_index(drop=True)

    # ─────────── Composite Signals ───────────
    def build_composite(df, ma_w, rsi_p):
        n = len(df)
        ma_sig    = np.zeros(n, int)
        rsi_sig   = np.zeros(n, int)
        macd_sig2 = np.zeros(n, int)
        for i in range(1, n):
            # MA
            if df["Close"].iat[i-1] < df[f"MA{ma_w}"].iat[i-1] and df["Close"].iat[i] > df[f"MA{ma_w}"].iat[i]:
                ma_sig[i] = 1
            elif df["Close"].iat[i-1] > df[f"MA{ma_w}"].iat[i-1] and df["Close"].iat[i] < df[f"MA{ma_w}"].iat[i]:
                ma_sig[i] = -1
            # RSI
            r = df[f"RSI{rsi_p}"].iat[i]
            if r < 30:    rsi_sig[i] = 1
            elif r > 70:  rsi_sig[i] = -1
            # MACD
            if df["MACD"].iat[i-1] < df["MACD_Signal"].iat[i-1] and df["MACD"].iat[i] > df["MACD_Signal"].iat[i]:
                macd_sig2[i] = 1
            elif df["MACD"].iat[i-1] > df["MACD_Signal"].iat[i-1] and df["MACD"].iat[i] < df["MACD_Signal"].iat[i]:
                macd_sig2[i] = -1

        comp = ma_sig + rsi_sig + macd_sig2
        df["MA_Signal"]     = ma_sig
        df["RSI_Signal"]    = rsi_sig
        df["MACD_Signal2"]  = macd_sig2
        df["Composite"]     = comp
        df["Trade"]         = np.sign(comp)
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

    if ticker:
        info  = yf.Ticker(ticker).info
        price = info.get("regularMarketPrice")
        if price is not None:
            st.subheader(f"💲 Live Price: ${price:.2f}")
        news = getattr(yf.Ticker(ticker), "news", []) or []
        if news:
            st.markdown("### 📰 Recent News & Sentiment")
            cnt = 0
            for art in news:
                title, link = art.get("title",""), art.get("link","")
                if not title or not link: continue
                txt   = art.get("summary", title)
                score = analyzer.polarity_scores(txt)["compound"]
                emoji = "🔺" if score>0.1 else ("🔻" if score<-0.1 else "➖")
                st.markdown(f"- [{title}]({link}) {emoji}")
                cnt += 1
                if cnt >= 5: break
        else:
            st.info("No recent news found.")

    if st.button("▶️ Run Composite Backtest"):
        df_raw = load_and_compute(ticker, ma_window, rsi_period, macd_fast, macd_slow, macd_signal)
        if df_raw.empty:
            st.error(f"No data for '{ticker}'.")
            st.stop()
        df_c, max_dd, sharpe, win_rt = backtest(build_composite(df_raw, ma_window, rsi_period))

        rec_map = {1:"🟢 BUY", 0:"🟡 HOLD", -1:"🔴 SELL"}
        last_trade = int(df_c["Trade"].iloc[-1])
        st.success(f"**{ticker}** → {rec_map[last_trade]}")

        # Why?
        ma_s   = int(df_c["MA_Signal"].iloc[-1])
        rsi_s  = int(df_c["RSI_Signal"].iloc[-1])
        macd_s = int(df_c["MACD_Signal2"].iloc[-1])
        rsi_v  = df_c[f"RSI{rsi_period}"].iloc[-1]
        ma_txt = {1:f"Price crossed above its {ma_window}-day MA.",
                  0:"No MA crossover today.",
                 -1:f"Price crossed below its {ma_window}-day MA."}[ma_s]
        rsi_txt = {1:f"RSI ({rsi_v:.1f}) < 30 → oversold.",
                   0:f"RSI ({rsi_v:.1f}) between 30–70 → neutral.",
                  -1:f"RSI ({rsi_v:.1f}) > 70 → overbought."}[rsi_s]
        macd_txt = {1:"MACD line crossed above signal.",
                    0:"No MACD crossover today.",
                   -1:"MACD line crossed below signal."}[macd_s]

        with st.expander("🔎 Why This Signal?"):
            st.write(f"- **MA:**  {ma_txt}")
            st.write(f>- **RSI:** {rsi_txt}")
            st.write(f>- **MACD:** {macd_txt}")
            st.write(f>- **Composite Score:** {df_c['Composite'].iloc[-1]}")

        # Metrics
        st.markdown(f"""
- **Buy & Hold:**   {(df_c["CumBH"].iloc[-1]-1)*100:.2f}%  
- **Strategy:**     {(df_c["CumStrat"].iloc[-1]-1)*100:.2f}%  
- **Sharpe:**       {sharpe:.2f}  
- **Max Drawdown:** {max_dd:.2f}%  
- **Win Rate:**     {win_rt:.1f}%  
""")
        fig, axs = plt.subplots(3,1,figsize=(10,12), sharex=True)
        axs[0].plot(df_c["Close"], label="Close")
        axs[0].plot(df_c[f"MA{ma_window}"], label=f"MA{ma_window}")
        axs[0].legend(); axs[0].set_title("Price & MA")
        axs[1].bar(df_c.index, df_c["Composite"], color="purple"); axs[1].set_title("Composite Vote")
        axs[2].plot(df_c["CumBH"], ":", label="Buy & Hold")
        axs[2].plot(df_c["CumStrat"], "-", label="Strategy"); axs[2].legend(); axs[2].set_title("Equity Curves")
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
            df_c, _, _, _ = backtest(build_composite(df_t, ma_window, rsi_period))
            perf.append({
                "Ticker":    t,
                "Composite": int(df_c["Composite"].iloc[-1]),
                "Signal":    {1:"BUY",0:"HOLD",-1:"SELL"}[int(df_c["Trade"].iloc[-1])],
                "BuyHold %": (df_c["CumBH"].iloc[-1]-1)*100,
                "Strat %":   (df_c["CumStrat"].iloc[-1]-1)*100
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
    sig_list = st.sidebar.multiselect("MACD sig spans", [5,9,12,16],   default=[macd_signal], key="grid_sig")
    if st.button("🏃‍♂️ Run Grid Search"):
        if not ticker:
            st.error("Enter a ticker."); st.stop()
        df0 = load_and_compute(ticker, ma_window, rsi_period, macd_fast, macd_slow, macd_signal)
        if df0.empty:
            st.error(f"No data for '{ticker}'."); st.stop()
        results = []
        with st.spinner("Testing combos…"):
            for mw in ma_list:
                for rp in rsi_list:
                    for mf in mf_list:
                        for ms in ms_list:
                            for sig in sig_list:
                                df_i = load_and_compute(ticker, mw, rp, mf, ms, sig)
                                if df_i.empty: continue
                                df_c, _, shap, win = backtest(build_composite(df_i, mw, rp))
                                results.append({
                                    "MA":mw, "RSI":rp, "MACD_Fast":mf, "MACD_Slow":ms, "MACD_Sig":sig,
                                    "Strategy %": (df_c["CumStrat"].iloc[-1]-1)*100,
                                    "Sharpe": shap, "Win %": win
                                })
        if results:
            df_grid = pd.DataFrame(results).sort_values("Strategy %", ascending=False).head(10)
            st.dataframe(df_grid, use_container_width=True)
            st.download_button("Download full CSV", df_grid.to_csv(index=False), "grid.csv")
        else:
            st.error("No valid combos.")

    # ───────────────────────────── Watchlist Summary ─────────────────────────────
    st.markdown("---")
    st.markdown("## ⏰ Watchlist Summary")
    watch = st.text_area("Enter tickers", "AAPL, MSFT, TSLA, SPY, QQQ").upper()
    if st.button("📬 Generate Watchlist Summary"):
        tbl = []
        for t in [x.strip() for x in watch.split(",") if x.strip()]:
            df_t = load_and_compute(t, ma_window, rsi_period, macd_fast, macd_slow, macd_signal)
            if df_t.empty:
                tbl.append({"Ticker": t, "Composite": None, "Signal": "N/A"})
                continue
            df_c, _, _, _ = backtest(build_composite(df_t, ma_window, rsi_period))
            tbl.append({
                "Ticker":    t,
                "Composite": int(df_c["Composite"].iloc[-1]),
                "Signal":    {1:"BUY",0:"HOLD",-1:"SELL"}[int(df_c["Trade"].iloc[-1])]
            })
        df_watch = pd.DataFrame(tbl).set_index("Ticker")
        st.dataframe(df_watch, use_container_width=True)
        for t in df_watch.index:
            df_t = load_and_compute(t, ma_window, rsi_period, macd_fast, macd_slow, macd_signal)
            df_c = build_composite(df_t, ma_window, rsi_period)
            last = df_c.iloc[-1]
            ma_s = int(last["MA_Signal"])
            rsi_s = int(last["RSI_Signal"])
            macd_s = int(last["MACD_Signal2"])
            rsi_v = last[f"RSI{rsi_period}"]
            ma_txt = {1:f"Price ↑ above {ma_window}-day MA.", 0:"No crossover.", -1:f"Price ↓ below {ma_window}-day MA."}[ma_s]
            rsi_txt= {1:f"RSI ({rsi_v:.1f}) <30 oversold.",0:f"RSI ({rsi_v:.1f}) neutral.",-1:f"RSI ({rsi_v:.1f}) >70 overbought."}[rsi_s]
            macd_txt = {1:"MACD ↑ signal.",0:"No crossover.",-1:"MACD ↓ signal."}[macd_s]
            with st.expander(f"🔎 {t} Reasoning ({df_watch.loc[t,'Signal']})"):
                st.write(f"- **MA:**  {ma_txt}")
                st.write(f"- **RSI:** {rsi_txt}")
                st.write(f"- **MACD:** {macd_txt}")
                st.write(f"- **Composite Score:** {df_watch.loc[t,'Composite']}")

    # ───────────────────────────── Portfolio Simulator ─────────────────────────────
    st.markdown("---")
    st.markdown("## 📊 Portfolio Simulator")

    # Portfolio inputs
    port_txt = st.text_input("Portfolio tickers (comma-separated)", "AAPL, MSFT, TSLA")
    port = [t.strip().upper() for t in port_txt.split(",") if t.strip()]
    default_w = 1/len(port) if port else 1.0
    cols = st.columns(len(port))
    weights = []
    for i,t in enumerate(port):
        with cols[i]:
            w = st.number_input(f"Wgt {t}", 0.0, 1.0, default_w, 0.01, key=f"w_{t}")
        weights.append(w)
    total = sum(weights)
    weights = [(w/total) if total>0 else default_w for w in weights]

    start = st.date_input("Start date", pd.to_datetime("2023-01-01"))
    end   = st.date_input("End date", pd.Timestamp.today())

    if st.button("▶️ Run Portfolio Simulation"):
        if not port:
            st.error("Add at least one ticker."); st.stop()
        prices = yf.download(port, start=start, end=end)["Close"].dropna()
        rets = prices.pct_change().dropna()
        port_ret = rets.mul(weights, axis=1).sum(axis=1)
        cum = (1+port_ret).cumprod()
        total_ret = cum.iloc[-1] - 1
        cagr = (cum.iloc[-1])**(252/len(cum)) -1
        vol  = port_ret.std() * np.sqrt(252)
        sharpe = cagr/vol if vol else np.nan
        dd = cum/cum.cummax() -1
        maxdd = dd.min()

        st.markdown(f"""
- **Total Return:**  {total_ret*100:.2f}%
- **CAGR:**          {cagr*100:.2f}%
- **Volatility:**    {vol*100:.2f}%
- **Sharpe Ratio:**  {sharpe:.2f}
- **Max Drawdown:**  {maxdd*100:.2f}%
""")
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(cum, label="Portfolio Equity")
        ax.set_title("Portfolio Cumulative Performance")
        ax.set_ylabel("Growth of $1")
        ax.legend()
        st.pyplot(fig)
