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

Each yields +1/–1/0 → sum → **Composite** → sign → BUY/HOLD/SELL.

Under **Engine** you can:
- Backtest a single ticker  
- Batch-test tickers  
- Grid-search parameters  
- Watchlist summary + reasoning  
- Recent news & sentiment overlay  
- **Export signals & key stats as CSV**
""")

# ───────────────────────────── Engine Tab ─────────────────────────────
with tab_engine:

    # ───────────── Defaults & Session State ─────────────
    DEFAULTS = dict(ma_window=10, rsi_period=14, macd_fast=12, macd_slow=26, macd_signal=9)
    for k,v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

    st.sidebar.header("Controls")
    if st.sidebar.button("🔄 Reset to defaults"):
        for k,v in DEFAULTS.items():
            st.session_state[k] = v

    st.sidebar.header("Indicator Parameters")
    ma_window  = st.sidebar.slider("MA window",       5,50,  st.session_state["ma_window"],   key="ma_window")
    rsi_period = st.sidebar.slider("RSI lookback",    5,30,  st.session_state["rsi_period"],  key="rsi_period")
    macd_fast  = st.sidebar.slider("MACD fast span",  5,20,  st.session_state["macd_fast"],   key="macd_fast")
    macd_slow  = st.sidebar.slider("MACD slow span", 20,40,  st.session_state["macd_slow"],   key="macd_slow")
    macd_sig   = st.sidebar.slider("MACD sig span",   5,20,  st.session_state["macd_signal"], key="macd_signal")

    st.title("🚀 QuantaraX — Composite Signal Engine")
    st.write("MA + RSI + MACD Composite Signals & Backtest")

    # ───────────────────────────── Data Loading ─────────────────────────────
    @st.cache_data(show_spinner=False)
    def load_and_compute(ticker, ma_w, rsi_p, mf, ms, sig):
        df = yf.download(ticker, period="6mo", progress=False)
        if df.empty or "Close" not in df:
            return pd.DataFrame()
        # MA
        ma_col = f"MA{ma_w}"
        df[ma_col] = df["Close"].rolling(ma_w).mean()
        # RSI
        d        = df["Close"].diff()
        up, dn   = d.clip(lower=0), -d.clip(upper=0)
        ema_up   = up.ewm(com=rsi_p-1, adjust=False).mean()
        ema_dn   = dn.ewm(com=rsi_p-1, adjust=False).mean()
        rsi_col  = f"RSI{rsi_p}"
        df[rsi_col] = 100 - 100/(1 + ema_up/ema_dn)
        # MACD
        ema_f    = df["Close"].ewm(span=mf, adjust=False).mean()
        ema_s    = df["Close"].ewm(span=ms, adjust=False).mean()
        macd     = ema_f - ema_s
        macd_sig = macd.ewm(span=sig, adjust=False).mean()
        df["MACD"]        = macd
        df["MACD_Signal"] = macd_sig
        # Drop NAs
        cols = [ma_col, rsi_col, "MACD", "MACD_Signal"]
        prs  = [c for c in cols if c in df.columns]
        if prs:
            df = df.dropna(subset=prs).reset_index(drop=True)
        return df

    # ───────────────────────────── Composite Signals ─────────────────────────────
    def build_composite(df, ma_w, rsi_p):
        n         = len(df)
        close_arr = df["Close"].to_numpy()
        ma_arr    = df[f"MA{ma_w}"].to_numpy()
        rsi_arr   = df[f"RSI{rsi_p}"].to_numpy()
        macd_arr  = df["MACD"].to_numpy()
        sig_arr   = df["MACD_Signal"].to_numpy()

        ma_sig    = np.zeros(n,int)
        rsi_sig   = np.zeros(n,int)
        macd_sig2 = np.zeros(n,int)
        comp      = np.zeros(n,int)
        trade     = np.zeros(n,int)

        for i in range(1,n):
            # MA crossover
            if close_arr[i-1]<ma_arr[i-1] and close_arr[i]>ma_arr[i]:
                ma_sig[i]=1
            elif close_arr[i-1]>ma_arr[i-1] and close_arr[i]<ma_arr[i]:
                ma_sig[i]=-1
            # RSI
            if rsi_arr[i]<30:
                rsi_sig[i]=1
            elif rsi_arr[i]>70:
                rsi_sig[i]=-1
            # MACD crossover
            if macd_arr[i-1]<sig_arr[i-1] and macd_arr[i]>sig_arr[i]:
                macd_sig2[i]=1
            elif macd_arr[i-1]>sig_arr[i-1] and macd_arr[i]<sig_arr[i]:
                macd_sig2[i]=-1
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
        df["CumBH"]    = (1+df["Return"]).cumprod()
        df["CumStrat"] = (1+df["StratRet"]).cumprod()

        dd     = df["CumStrat"] / df["CumStrat"].cummax() - 1
        max_dd = dd.min()*100
        std    = df["StratRet"].std()
        sharpe = (df["StratRet"].mean()/std*np.sqrt(252)) if std else np.nan
        win_rt = (df["StratRet"]>0).mean()*100

        return df, max_dd, sharpe, win_rt

    # ───────────────────────────── Export Helper ─────────────────────────────
    def make_export_df(df_raw, df_comp):
        # build signals table
        out = df_raw.copy()
        out["MA_Signal"]    = df_comp["MA_Signal"]
        out["RSI_Signal"]   = df_comp["RSI_Signal"]
        out["MACD_Signal2"] = df_comp["MACD_Signal2"]
        out["Composite"]    = df_comp["Composite"]
        out["Trade"]        = df_comp["Trade"]
        # metrics
        ret    = (df_comp["CumStrat"].iloc[-1]-1)*100
        dd     = (df_comp["CumStrat"]/df_comp["CumStrat"].cummax()-1).min()*100
        std    = df_comp["StratRet"].std()
        sharpe = (df_comp["StratRet"].mean()/std*np.sqrt(252)) if std else np.nan
        win    = (df_comp["StratRet"]>0).mean()*100
        stats = pd.DataFrame({
            "Metric": ["Return %","Max Drawdown %","Sharpe","Win Rate %"],
            "Value":  [ret, dd, sharpe, win]
        })
        return stats, out

    # ───────────────────────────── Single‐Ticker Backtest ─────────────────────────────
    st.markdown("## Single‐Ticker Backtest")
    ticker = st.text_input("Ticker (e.g. AAPL)","AAPL").upper()

    # Live price + news
    if ticker:
        info  = yf.Ticker(ticker).info
        price = info.get("regularMarketPrice")
        if price is not None:
            st.subheader(f"💲 Live Price: ${price:.2f}")

        news = getattr(yf.Ticker(ticker), "news", []) or []
        if news:
            st.markdown("### 📰 Recent News & Sentiment")
            shown = 0
            for art in news:
                title, link = art.get("title",""), art.get("link","")
                if not (title and link): continue
                txt   = art.get("summary", title)
                score = analyzer.polarity_scores(txt)["compound"]
                emoji = "🔺" if score>0.1 else ("🔻" if score<-0.1 else "➖")
                st.markdown(f"- [{title}]({link}) {emoji}")
                shown += 1
                if shown>=5: break
            if shown==0:
                st.info("No recent news.")
        else:
            st.info("No recent news.")

    if st.button("▶️ Run Composite Backtest"):
        df_raw, df_c, max_dd, sharpe, win_rt = None, None, None, None, None
        df0 = load_and_compute(ticker, ma_window, rsi_period, macd_fast, macd_slow, macd_sig)
        if df0.empty:
            st.error(f"No data for '{ticker}'"); st.stop()

        df_c, max_dd, sharpe, win_rt = backtest(build_composite(df0, ma_window, rsi_period))
        df_raw = df0.copy()

        # Top‐line rec
        rec_map = {1:"🟢 BUY",0:"🟡 HOLD",-1:"🔴 SELL"}
        st.success(f"**{ticker}**: {rec_map[int(df_c['Trade'].iloc[-1])]}")

        # Explain
        ma_s, rsi_s, macd_s = (int(df_c[c].iloc[-1]) for c in ["MA_Signal","RSI_Signal","MACD_Signal2"])
        rsi_v = df_c[f"RSI{rsi_period}"].iloc[-1]
        ma_txt   = {1:f"Price ↑ above {ma_window}-day MA.",0:"No crossover.",-1:f"Price ↓ below MA."}[ma_s]
        rsi_txt  = {1:f"RSI ({rsi_v:.1f}) <30 → oversold.",0:f"RSI ({rsi_v:.1f}) neutral.",-1:f"RSI ({rsi_v:.1f})>70 → overbought."}[rsi_s]
        macd_txt = {1:"MACD ↑ signal.",0:"No crossover.",-1:"MACD ↓ signal."}[macd_s]

        with st.expander("🔎 Why This Signal?"):
            st.write(f"- **MA:** {ma_txt}")
            st.write(f"- **RSI:** {rsi_txt}")
            st.write(f"- **MACD:** {macd_txt}")
            st.write(f"- **Composite Score:** {int(df_c['Composite'].iloc[-1])}")

        # Performance
        st.markdown(f"""
- **Buy & Hold:**    {(df_c['CumBH'].iloc[-1]-1)*100:.2f}%  
- **Strategy:**      {(df_c['CumStrat'].iloc[-1]-1)*100:.2f}%  
- **Sharpe:**        {sharpe:.2f}  
- **Max Drawdown:**  {max_dd:.2f}%  
- **Win Rate:**      {win_rt:.1f}%  
""")

        # Charts
        fig, axs = plt.subplots(3,1,figsize=(10,12), sharex=True)
        axs[0].plot(df_c["Close"], label="Close")
        axs[0].plot(df_c[f"MA{ma_window}"], label=f"MA{ma_window}")
        axs[0].legend(); axs[0].set_title("Price & MA")
        axs[1].bar(df_c.index, df_c["Composite"], color="purple"); axs[1].set_title("Composite")
        axs[2].plot(df_c["CumBH"], ":", label="BH")
        axs[2].plot(df_c["CumStrat"], "-", label="Strat")
        axs[2].legend(); axs[2].set_title("Equity Curves")
        plt.xticks(rotation=45); plt.tight_layout()
        st.pyplot(fig)

        # ─── Export signals & stats ───
        stats_df, signals_df = make_export_df(df_raw, df_c)
        st.download_button(
            "📥 Download Metrics (CSV)",
            stats_df.to_csv(index=False),
            f"{ticker}_metrics.csv",
            "text/csv"
        )
        st.download_button(
            "📥 Download Signals (CSV)",
            signals_df.reset_index().to_csv(index=False),
            f"{ticker}_signals.csv",
            "text/csv"
        )

    # ─────────── Batch Backtest ───────────
    st.markdown("---")
    st.markdown("## Batch Backtest")
    batch = st.text_area("Tickers","AAPL, MSFT, TSLA, SPY, QQQ").upper()
    if st.button("▶️ Run Batch Backtest"):
        perf = []
        for s in [x.strip() for x in batch.split(",") if x.strip()]:
            df_t = load_and_compute(s, ma_window, rsi_period, macd_fast, macd_slow, macd_sig)
            if df_t.empty: continue
            df_tc, md, sh, wr = backtest(build_composite(df_t, ma_window, rsi_period))
            perf.append({
                "Ticker": s,
                "Composite": int(df_tc["Composite"].iloc[-1]),
                "Signal":    {1:"BUY",0:"HOLD",-1:"SELL"}[int(df_tc["Trade"].iloc[-1])],
                "BH %":      (df_tc["CumBH"].iloc[-1]-1)*100,
                "Strat %":   (df_tc["CumStrat"].iloc[-1]-1)*100,
                "Sharpe":    sh,
                "MaxDD":     md,
                "Win %":     wr
            })
        if perf:
            df_perf = pd.DataFrame(perf).set_index("Ticker")
            st.dataframe(df_perf, use_container_width=True)
            st.download_button("Download CSV", df_perf.to_csv(), "batch.csv")
        else:
            st.error("No valid data.")

    # ─────────── Hyperparameter Optimization ───────────
    st.markdown("---")
    st.markdown("## 🛠️ Hyperparameter Optimization")
    ma_list  = st.sidebar.multiselect("MA windows",[5,10,15,20,30],default=[ma_window],key="grid_ma")
    rsi_list = st.sidebar.multiselect("RSI lookbacks",[7,14,21,28],default=[rsi_period],key="grid_rsi")
    mf_list  = st.sidebar.multiselect("MACD fast spans",[8,12,16,20],default=[macd_fast],key="grid_mf")
    ms_list  = st.sidebar.multiselect("MACD slow spans",[20,26,32,40],default=[macd_slow],key="grid_ms")
    sig_list = st.sidebar.multiselect("MACD sig spans",[5,9,12,16],default=[macd_sig],key="grid_sig")

    if st.button("🏃‍♂️ Run Grid Search"):
        if not ticker:
            st.error("Enter a ticker."); st.stop()
        df0 = load_and_compute(ticker, ma_window, rsi_period, macd_fast, macd_slow, macd_sig)
        if df0.empty:
            st.error(f"No data for '{ticker}'"); st.stop()
        results = []
        with st.spinner("Testing combos…"):
            for mw in ma_list:
                for rp in rsi_list:
                    for mf_ in mf_list:
                        for ms_ in ms_list:
                            for s_ in sig_list:
                                df_i = load_and_compute(ticker, mw, rp, mf_, ms_, s_)
                                if df_i.empty: continue
                                df_ci, md_i, sh_i, wr_i = backtest(build_composite(df_i, mw, rp))
                                results.append({
                                    "MA":mw, "RSI":rp,
                                    "MACD_Fast":mf_, "MACD_Slow":ms_, "MACD_Sig":s_,
                                    "Strategy %":(df_ci["CumStrat"].iloc[-1]-1)*100,
                                    "Sharpe":sh_i, "MaxDD %":md_i, "Win %":wr_i
                                })
        if results:
            df_grid = (pd.DataFrame(results)
                       .sort_values("Strategy %", ascending=False)
                       .head(10))
            st.dataframe(df_grid, use_container_width=True)
            st.download_button("Download full CSV",
                               df_grid.to_csv(index=False),
                               "grid.csv",
                               "text/csv")
        else:
            st.error("No valid combos.")

    # ─────────── Watchlist Summary ───────────
    st.markdown("---")
    st.markdown("## ⏰ Watchlist Summary")
    watch = st.text_area("Enter tickers","AAPL, MSFT, TSLA, SPY, QQQ").upper()
    if st.button("📬 Generate Watchlist Summary"):
        tbl = []
        for s in [x.strip() for x in watch.split(",") if x.strip()]:
            df_t = load_and_compute(s, ma_window, rsi_period, macd_fast, macd_slow, macd_sig)
            if df_t.empty:
                tbl.append({"Ticker":s,"Composite":None,"Signal":"N/A"})
                continue
            df_w,_,_,_ = backtest(build_composite(df_t, ma_window, rsi_period))
            tbl.append({
                "Ticker":   s,
                "Composite":int(df_w["Composite"].iloc[-1]),
                "Signal":   {1:"BUY",0:"HOLD",-1:"SELL"}[int(df_w["Trade"].iloc[-1])]
            })
        df_watch = pd.DataFrame(tbl).set_index("Ticker")
        st.dataframe(df_watch, use_container_width=True)

        for s in df_watch.index:
            df_t = load_and_compute(s, ma_window, rsi_period, macd_fast, macd_slow, macd_sig)
            df_c = build_composite(df_t, ma_window, rsi_period)
            last = df_c.iloc[-1]
            ma_s, rsi_s, macd_s = (
                int(last["MA_Signal"]),
                int(last["RSI_Signal"]),
                int(last["MACD_Signal2"])
            )
            rsi_v = last.get(f"RSI{rsi_period}", np.nan)
            ma_txt   = {1:"Cross ↑ MA",0:"No crossover", -1:"Cross ↓ MA"}[ma_s]
            rsi_txt  = {1:"RSI oversold",0:"RSI neutral", -1:"RSI overbought"}[rsi_s]
            macd_txt = {1:"MACD ↑ signal",0:"No cross", -1:"MACD ↓ signal"}[macd_s]

            with st.expander(f"🔎 {s} Reasoning ({df_watch.loc[s,'Signal']})"):
                st.write(f"- **MA:**  {ma_txt}")
                st.write(f"- **RSI:** {rsi_txt}")
                st.write(f"- **MACD:** {macd_txt}")
                st.write(f"- **Composite Score:** {df_watch.loc[s,'Composite']}")
