import streamlit as st 
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ───────────────────────────── Page Setup ─────────────────────────────
st.set_page_config(page_title="QuantaraX Composite Signals", layout="centered")
analyzer = SentimentIntensityAnalyzer()

# ───────────────────────────── Mappings ─────────────────────────────
rec_map = {
    1: "🟢 BUY",
    0: "🟡 HOLD",
   -1: "🔴 SELL",
}

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

    @st.cache_data(show_spinner=False)
    def load_and_compute(ticker, ma_w, rsi_p, mf, ms, sig):
        df = yf.download(ticker, period="6mo", progress=False)
        if df.empty or "Close" not in df:
            return pd.DataFrame()
        df[f"MA{ma_w}"] = df["Close"].rolling(ma_w).mean()
        d = df["Close"].diff()
        up, dn = d.clip(lower=0), -d.clip(upper=0)
        ema_up   = up.ewm(com=rsi_p-1, adjust=False).mean()
        ema_down = dn.ewm(com=rsi_p-1, adjust=False).mean()
        df[f"RSI{rsi_p}"] = 100 - 100/(1 + ema_up/ema_down)
        ema_f    = df["Close"].ewm(span=mf, adjust=False).mean()
        ema_s    = df["Close"].ewm(span=ms, adjust=False).mean()
        macd     = ema_f - ema_s
        macd_sig = macd.ewm(span=sig, adjust=False).mean()
        df["MACD"], df["MACD_Signal"] = macd, macd_sig
        cols = [f"MA{ma_w}", f"RSI{rsi_p}", "MACD", "MACD_Signal"]
        try:
            df = df.dropna(subset=cols).reset_index(drop=True)
        except KeyError:
            pass
        return df

    def build_composite(df, ma_w, rsi_p):
        n = len(df)
        close = df["Close"].to_numpy()
        ma    = df[f"MA{ma_w}"].to_numpy()
        rsi   = df[f"RSI{rsi_p}"].to_numpy()
        macd  = df["MACD"].to_numpy()
        sig   = df["MACD_Signal"].to_numpy()
        ma_sig = np.zeros(n,int); rsi_sig = np.zeros(n,int); macd_sig2 = np.zeros(n,int)
        comp   = np.zeros(n,int); trade   = np.zeros(n,int)
        for i in range(1,n):
            if close[i-1]<ma[i-1]<close[i]:
                ma_sig[i]=1
            elif close[i-1]>ma[i-1]>close[i]:
                ma_sig[i]=-1
            if rsi[i]<30:   rsi_sig[i]=1
            elif rsi[i]>70: rsi_sig[i]=-1
            if macd[i-1]<sig[i-1]<macd[i]:
                macd_sig2[i]=1
            elif macd[i-1]>sig[i-1]>macd[i]:
                macd_sig2[i]=-1
            comp[i]  = ma_sig[i]+rsi_sig[i]+macd_sig2[i]
            trade[i] = np.sign(comp[i])
        df["MA_Signal"], df["RSI_Signal"] = ma_sig, rsi_sig
        df["MACD_Signal2"], df["Composite"] = macd_sig2, comp
        df["Trade"] = trade
        return df

    def backtest(df):
        df = df.copy()
        df["Return"]   = df["Close"].pct_change().fillna(0)
        df["Position"] = df["Trade"].shift(1).fillna(0).clip(0,1)
        df["StratRet"] = df["Position"] * df["Return"]
        df["CumBH"]    = (1 + df["Return"]).cumprod()
        df["CumStrat"] = (1 + df["StratRet"]).cumprod()
        dd      = df["CumStrat"]/df["CumStrat"].cummax() - 1
        max_dd  = dd.min()*100
        std_dev = df["StratRet"].std()
        sharpe  = (df["StratRet"].mean()/std_dev*np.sqrt(252)) if std_dev else np.nan
        win_rt  = (df["StratRet"]>0).mean()*100
        return df, max_dd, sharpe, win_rt

    # --- Single-Ticker ---
    st.markdown("## Single-Ticker Backtest")
    ticker = st.text_input("Ticker (e.g. AAPL)","AAPL").upper()

    if ticker:
        info  = yf.Ticker(ticker).info
        price = info.get("regularMarketPrice")
        if price: st.subheader(f"💲 Live Price: ${price:.2f}")
        news  = getattr(yf.Ticker(ticker), "news", []) or []
        if news:
            st.markdown("### 📰 Recent News & Sentiment")
            shown=0
            for art in news:
                title, link = art.get("title",""), art.get("link","")
                if not (title and link): continue
                score = analyzer.polarity_scores(art.get("summary",title))["compound"]
                emoji = "🔺" if score>0.1 else "🔻" if score<-0.1 else "➖"
                st.markdown(f"- [{title}]({link}) {emoji}")
                shown+=1
                if shown>=5: break
            if shown==0: st.info("No recent news found.")
        else:
            st.info("No recent news found.")

    if st.button("▶️ Run Composite Backtest"):
        df_raw = load_and_compute(ticker, ma_window, rsi_period, macd_fast, macd_slow, macd_signal)
        if df_raw.empty:
            st.error(f"No data for '{ticker}'"); st.stop()
        df_c, max_dd, sharpe, win_rt = backtest(build_composite(df_raw, ma_window, rsi_period))

        st.success(f"**{ticker}**: {rec_map[int(df_c['Trade'].iloc[-1])]}")
        ma_s, rsi_s, macd_s = [int(df_c[col].iloc[-1]) for col in ["MA_Signal","RSI_Signal","MACD_Signal2"]]
        rsi_v = df_c[f"RSI{rsi_period}"].iloc[-1]

        ma_txt   = {1:f"Price ↑ above {ma_window}-day MA.",0:"No crossover.",-1:f"Price ↓ below MA."}[ma_s]
        rsi_txt  = {1:f"RSI ({rsi_v:.1f}) <30 oversold.",0:f"RSI ({rsi_v:.1f}) neutral.",-1:f"RSI ({rsi_v:.1f}) >70 overbought."}[rsi_s]
        macd_txt = {1:"MACD ↑ signal.",0:"No crossover.",-1:"MACD ↓ signal."}[macd_s]

        with st.expander("🔎 Why This Signal?"):
            st.write(f"- **MA:**  {ma_txt}")
            st.write(f"- **RSI:** {rsi_txt}")
            st.write(f"- **MACD:** {macd_txt}")
            st.write(f"- **Composite Score:** {int(df_c['Composite'].iloc[-1])}")

        st.markdown(f"""
- **Buy & Hold:**    {(df_c['CumBH'].iloc[-1]-1)*100:.2f}%  
- **Strategy:**      {(df_c['CumStrat'].iloc[-1]-1)*100:.2f}%  
- **Sharpe:**        {sharpe:.2f}  
- **Max Drawdown:**  {max_dd:.2f}%  
- **Win Rate:**      {win_rt:.1f}%  
""")

        fig, axs = plt.subplots(3,1,figsize=(10,12), sharex=True)
        axs[0].plot(df_c["Close"], label="Close")
        axs[0].plot(df_c[f"MA{ma_window}"], label=f"MA{ma_window}")
        axs[0].legend(); axs[0].set_title("Price & MA")
        axs[1].bar(df_c.index, df_c["Composite"], color="purple"); axs[1].set_title("Composite Vote")
        axs[2].plot(df_c["CumBH"], ":", label="Buy & Hold")
        axs[2].plot(df_c["CumStrat"], "-", label="Strategy")
        axs[2].legend(); axs[2].set_title("Equity Curves")
        plt.xticks(rotation=45); plt.tight_layout(); st.pyplot(fig)

    # --- Batch Backtest ---
    st.markdown("---")
    st.markdown("## Batch Backtest")
    batch = st.text_area("Tickers (comma-separated)","AAPL, MSFT, TSLA, SPY, QQQ").upper()
    if st.button("▶️ Run Batch Backtest"):
        perf=[]
        for t in [x.strip() for x in batch.split(",") if x.strip()]:
            df_t = load_and_compute(t, ma_window, rsi_period, macd_fast, macd_slow, macd_signal)
            if df_t.empty: continue
            df_tc, md, sh, wr = backtest(build_composite(df_t, ma_window, rsi_period))
            perf.append({
                "Ticker":    t,
                "Composite": int(df_tc["Composite"].iloc[-1]),
                "Signal":    rec_map[int(df_tc["Trade"].iloc[-1])],
                "BuyHold %": (df_tc["CumBH"].iloc[-1]-1)*100,
                "Strategy %":(df_tc["CumStrat"].iloc[-1]-1)*100,
                "Sharpe":    sh,
                "MaxDD %":   md,
                "Win %":     wr
            })
        if perf:
            df_perf = pd.DataFrame(perf).set_index("Ticker")
            st.dataframe(df_perf, use_container_width=True)
            st.download_button("Download CSV", df_perf.to_csv(), "batch.csv")
        else:
            st.error("No valid data.")

    # --- Hyperparameter Optimization ---
    st.markdown("---")
    st.markdown("## 🛠️ Hyperparameter Optimization")
    ma_list  = st.sidebar.multiselect("MA windows",[5,10,15,20,30],default=[ma_window],key="grid_ma")
    rsi_list = st.sidebar.multiselect("RSI lookbacks",[7,14,21,28],default=[rsi_period],key="grid_rsi")
    mf_list  = st.sidebar.multiselect("MACD fast spans",[8,12,16,20],default=[macd_fast],key="grid_mf")
    ms_list  = st.sidebar.multiselect("MACD slow spans",[20,26,32,40],default=[macd_slow],key="grid_ms")
    sig_list = st.sidebar.multiselect("MACD sig spans",[5,9,12,16],default=[macd_signal],key="grid_sig")

    if st.button("🏃‍♂️ Run Grid Search"):
        if not ticker:
            st.error("Enter a ticker."); st.stop()
        df_full = load_and_compute(ticker, ma_window, rsi_period, macd_fast, macd_slow, macd_signal)
        if df_full.empty:
            st.error(f"No data for '{ticker}'"); st.stop()
        results=[]
        with st.spinner("Testing parameter combos…"):
            for mw in ma_list:
                for rp in rsi_list:
                    for mf_ in mf_list:
                        for ms_ in ms_list:
                            for s_ in sig_list:
                                df_i = load_and_compute(ticker, mw, rp, mf_, ms_, s_)
                                if df_i.empty: continue
                                df_ci, md_i, sh_i, wr_i = backtest(build_composite(df_i, mw, rp))
                                results.append({
                                    "MA":mw,"RSI":rp,"MACD_Fast":mf_,"MACD_Slow":ms_,"MACD_Sig":s_,
                                    "Strategy %":(df_ci["CumStrat"].iloc[-1]-1)*100,
                                    "Sharpe":sh_i,"MaxDD %":md_i,"Win %":wr_i
                                })
        if results:
            df_grid = pd.DataFrame(results).sort_values("Strategy %",ascending=False).head(10)
            st.dataframe(df_grid, use_container_width=True)
            st.download_button("Download CSV", df_grid.to_csv(index=False), "grid.csv")
        else:
            st.error("No valid combos.")

    # --- Watchlist Summary ---
    st.markdown("---")
    st.markdown("## ⏰ Watchlist Summary")
    watch = st.text_area("Enter tickers","AAPL, MSFT, TSLA, SPY, QQQ").upper()
    if st.button("📬 Generate Watchlist Summary"):
        table=[]
        for t in [s.strip() for s in watch.split(",") if s.strip()]:
            df_t = load_and_compute(t, ma_window, rsi_period, macd_fast, macd_slow, macd_signal)
            if df_t.empty:
                table.append({"Ticker":t,"Composite":None,"Signal":"N/A"})
                continue
            df_w,_,_,_ = backtest(build_composite(df_t, ma_window, rsi_period))
            table.append({
                "Ticker":t,
                "Composite":int(df_w["Composite"].iloc[-1]),
                "Signal":rec_map[int(df_w["Trade"].iloc[-1])]
            })
        df_watch = pd.DataFrame(table).set_index("Ticker")
        st.dataframe(df_watch, use_container_width=True)

        for t in df_watch.index:
            df_t = load_and_compute(t, ma_window, rsi_period, macd_fast, macd_slow, macd_signal)
            if df_t.empty: continue
            df_c = build_composite(df_t, ma_window, rsi_period)  # <— fixed here
            last = df_c.iloc[-1]
            ma_s, rsi_s, macd_s = int(last["MA_Signal"]), int(last["RSI_Signal"]), int(last["MACD_Signal2"])
            rsi_v = last[f"RSI{rsi_period}"]
            ma_txt = {1:f"Price ↑ above {ma_window}-day MA.",0:"No crossover.",-1:f"Price ↓ below MA."}[ma_s]
            rsi_txt= {1:f"RSI ({rsi_v:.1f}) <30 oversold.",0:f"RSI ({rsi_v:.1f}) neutral.",-1:f"RSI ({rsi_v:.1f}) >70 overbought."}[rsi_s]
            macd_txt={1:"MACD ↑ signal.",0:"No crossover.",-1:"MACD ↓ signal."}[macd_s]

            with st.expander(f"🔎 {t} Reasoning ({df_watch.loc[t,'Signal']})"):
                st.write(f"- **MA:**  {ma_txt}")
                st.write(f"- **RSI:** {rsi_txt}")
                st.write(f"- **MACD:** {macd_txt}")
                st.write(f"- **Composite Score:** {df_watch.loc[t,'Composite']}")
