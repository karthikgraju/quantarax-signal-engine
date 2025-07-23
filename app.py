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
tab_engine, tab_help, tab_port = st.tabs(["🚀 Engine", "❓ How It Works", "💼 Portfolio Simulator"])

# ───────────────────────────── Help Tab ─────────────────────────────
with tab_help:
    st.header("How QuantaraX Works")
    st.markdown("""
**QuantaraX** combines three indicators into a single composite vote:

1. **Moving Average Crossover**  
2. **RSI**  
3. **MACD Crossover**  

Each gives +1 (bull), –1 (bear), or 0 (neutral). Sum → **Composite** (–3…+3).  
Position = sign(composite):  
• ≥+1 → BUY  
• =0  → HOLD  
• ≤–1 → SELL

Under **Engine** you can backtest, batch-test, grid-search, view a watchlist summary, and see news & sentiment.  
Under **Portfolio Simulator** you can paste your own positions CSV to see live P&L & signals.
""")

# ───────────────────────────── Engine Tab ─────────────────────────────
with tab_engine:

    # Session defaults
    DEFAULTS = dict(ma_window=10, rsi_period=14, macd_fast=12, macd_slow=26, macd_signal=9)
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(k, v)

    # Sidebar controls
    st.sidebar.header("Controls")
    if st.sidebar.button("🔄 Reset to defaults"):
        for k, v in DEFAULTS.items():
            st.session_state[k] = v

    st.sidebar.header("Indicator Parameters")
    ma_window   = st.sidebar.slider("MA window",      5, 50, st.session_state.ma_window,   key="ma_window")
    rsi_period  = st.sidebar.slider("RSI lookback",   5, 30, st.session_state.rsi_period,  key="rsi_period")
    macd_fast   = st.sidebar.slider("MACD fast span", 5, 20, st.session_state.macd_fast,   key="macd_fast")
    macd_slow   = st.sidebar.slider("MACD slow span",20, 40, st.session_state.macd_slow,   key="macd_slow")
    macd_signal = st.sidebar.slider("MACD sig span",  5, 20, st.session_state.macd_signal, key="macd_signal")

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
        ema_dn   = dn.ewm(com=rsi_p-1, adjust=False).mean()
        df[f"RSI{rsi_p}"] = 100 - 100/(1 + ema_up/ema_dn)
        ema_f = df["Close"].ewm(span=mf, adjust=False).mean()
        ema_s = df["Close"].ewm(span=ms, adjust=False).mean()
        macd  = ema_f - ema_s
        df["MACD"] = macd
        df["MACD_Signal"] = macd.ewm(span=sig, adjust=False).mean()
        cols = [f"MA{ma_w}", f"RSI{rsi_p}", "MACD", "MACD_Signal"]
        return df.dropna(subset=cols).reset_index(drop=True)

    def build_composite(df, ma_w, rsi_p):
        n = len(df)
        ma = df[f"MA{ma_w}"].to_numpy()
        rsi = df[f"RSI{rsi_p}"].to_numpy()
        macd = df["MACD"].to_numpy()
        sigl = df["MACD_Signal"].to_numpy()
        ma_sig = np.zeros(n,int)
        rsi_sig = np.zeros(n,int)
        macd_sig2 = np.zeros(n,int)
        comp = np.zeros(n,int)
        trade= np.zeros(n,int)
        for i in range(1,n):
            c, m, r, x, y = df["Close"][i], ma[i], rsi[i], macd[i], sigl[i]
            if df["Close"][i-1]<ma[i-1] and c>m: ma_sig[i]=1
            elif df["Close"][i-1]>ma[i-1] and c<m: ma_sig[i]=-1
            if r<30: rsi_sig[i]=1
            elif r>70: rsi_sig[i]=-1
            if macd[i-1]<sigl[i-1] and x>y: macd_sig2[i]=1
            elif macd[i-1]>sigl[i-1] and x<y: macd_sig2[i]=-1
            comp[i]=ma_sig[i]+rsi_sig[i]+macd_sig2[i]
            trade[i]=np.sign(comp[i])
        df["MA_Signal"]=ma_sig
        df["RSI_Signal"]=rsi_sig
        df["MACD_Signal2"]=macd_sig2
        df["Composite"]=comp
        df["Trade"]=trade
        return df

    def backtest(df):
        df = df.copy()
        df["Return"]   = df["Close"].pct_change().fillna(0)
        df["Position"] = df["Trade"].shift(1).fillna(0).clip(0,1)
        df["StratRet"] = df["Position"] * df["Return"]
        df["CumBH"]    = (1+df["Return"]).cumprod()
        df["CumStrat"] = (1+df["StratRet"]).cumprod()
        dd = df["CumStrat"]/df["CumStrat"].cummax() -1
        max_dd = dd.min()*100
        std = df["StratRet"].std()
        sharpe= (df["StratRet"].mean()/std*np.sqrt(252)) if std else np.nan
        win_rt= (df["StratRet"]>0).mean()*100
        return df, max_dd, sharpe, win_rt

    # Single‐Ticker Backtest
    st.markdown("## Single‐Ticker Backtest")
    ticker = st.text_input("Ticker (e.g. AAPL)", "AAPL").upper()
    if ticker:
        info= yf.Ticker(ticker).info
        price=info.get("regularMarketPrice")
        if price: st.subheader(f"💲 Live Price: ${price:.2f}")
        news=getattr(yf.Ticker(ticker),"news",[]) or []
        if news:
            st.markdown("### 📰 Recent News & Sentiment")
            for art in news[:5]:
                t,l=art.get("title",""),art.get("link","")
                txt=art.get("summary",t)
                s=analyzer.polarity_scores(txt)["compound"]
                emo="🔺" if s>0.1 else ("🔻" if s< -0.1 else "➖")
                st.markdown(f"- [{t}]({l}) {emo}")
        else:
            st.info("No recent news found.")

    if st.button("▶️ Run Composite Backtest"):
        df_raw = load_and_compute(ticker, ma_window, rsi_period, macd_fast, macd_slow, macd_signal)
        if df_raw.empty:
            st.error(f"No data for '{ticker}'"); st.stop()
        df_c, max_dd, sharpe, win_rt = backtest(build_composite(df_raw, ma_window, rsi_period))
        rec = rec_map[int(df_c["Trade"].iloc[-1])]
        st.success(f"**{ticker}**: {rec}")

        ma_s, rsi_s, macd_s = (int(df_c[c].iloc[-1]) for c in ["MA_Signal","RSI_Signal","MACD_Signal2"])
        rsi_v = df_c[f"RSI{rsi_period}"].iloc[-1]
        ma_txt={1:f"Price ↑ above {ma_window}-day MA.",0:"No crossover.",-1:f"Price ↓ below MA."}[ma_s]
        rsi_txt={1:f"RSI ({rsi_v:.1f}) <30 → oversold.",0:f"RSI ({rsi_v:.1f}) neutral.",-1:f"RSI ({rsi_v:.1f}) >70 → overbought."}[rsi_s]
        macd_txt={1:"MACD ↑ signal.",0:"No crossover.",-1:"MACD ↓ signal."}[macd_s]

        with st.expander("🔎 Why This Signal?"):
            st.write(f"- **MA:**  {ma_txt}")
            st.write(f>- **RSI:** {rsi_txt}")
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
        axs[0].plot(df_c["Close"],label="Close"); axs[0].plot(df_c[f"MA{ma_window}"],label=f"MA{ma_window}")
        axs[0].legend(); axs[0].set_title("Price & MA")
        axs[1].bar(df_c.index,df_c["Composite"],color="purple"); axs[1].set_title("Composite")
        axs[2].plot(df_c["CumBH"],":",label="BH"); axs[2].plot(df_c["CumStrat"],"-",label="Strat")
        axs[2].legend(); axs[2].set_title("Equity"); plt.xticks(rotation=45)
        plt.tight_layout(); st.pyplot(fig)

    # Batch Backtest
    st.markdown("---")
    st.markdown("## Batch Backtest")
    batch = st.text_area("Tickers (comma-separated)", "AAPL, MSFT, TSLA, SPY, QQQ").upper()
    if st.button("▶️ Run Batch Backtest"):
        perf=[]
        for t in [x.strip() for x in batch.split(",") if x.strip()]:
            df_t = load_and_compute(t, ma_window, rsi_period, macd_fast, macd_slow, macd_signal)
            if df_t.empty: continue
            df_tc, md, sh, wr = backtest(build_composite(df_t, ma_window, rsi_period))
            perf.append({
                "Ticker":t,
                "Composite":int(df_tc["Composite"].iloc[-1]),
                "Signal":rec_map[int(df_tc["Trade"].iloc[-1])],
                "BuyHold %":(df_tc["CumBH"].iloc[-1]-1)*100,
                "Strat %":  (df_tc["CumStrat"].iloc[-1]-1)*100,
                "Sharpe":sh, "MaxDD %":md, "Win %":wr
            })
        if perf:
            df_perf=pd.DataFrame(perf).set_index("Ticker")
            st.dataframe(df_perf,use_container_width=True)
            st.download_button("Download CSV", df_perf.to_csv(), "batch.csv")
        else:
            st.error("No valid data for batch tickers.")

    # Hyperparameter Optimization
    st.markdown("---")
    st.markdown("## 🛠️ Hyperparameter Optimization")
    ma_list  = st.sidebar.multiselect("MA windows",[5,10,15,20,30],default=[ma_window],key="grid_ma")
    rsi_list = st.sidebar.multiselect("RSI lookbacks",[7,14,21,28],default=[rsi_period],key="grid_rsi")
    mf_list  = st.sidebar.multiselect("MACD fast spans",[8,12,16,20],default=[macd_fast],key="grid_mf")
    ms_list  = st.sidebar.multiselect("MACD slow spans",[20,26,32,40],default=[macd_slow],key="grid_ms")
    sig_list = st.sidebar.multiselect("MACD sig spans",[5,9,12,16],default=[macd_signal],key="grid_sig")
    if st.button("🏃‍♂️ Run Grid Search"):
        if not ticker: st.error("Enter ticker"); st.stop()
        df_full=load_and_compute(ticker,ma_window,rsi_period,macd_fast,macd_slow,macd_signal)
        if df_full.empty:
            st.error(f"No data for '{ticker}'"); st.stop()
        results=[]
        with st.spinner("Testing combos…"):
            for mw in ma_list:
                for rp in rsi_list:
                    for mf_ in mf_list:
                        for ms_ in ms_list:
                            for s_ in sig_list:
                                df_i=load_and_compute(ticker,mw,rp,mf_,ms_,s_)
                                if df_i.empty: continue
                                df_ci, md_i, sh_i, wr_i = backtest(build_composite(df_i,mw,rp))
                                results.append({
                                    "MA":mw,"RSI":rp,
                                    "MF":mf_,"MS":ms_,"SIG":s_,
                                    "Return %":(df_ci["CumStrat"].iloc[-1]-1)*100,
                                    "Sharpe":sh_i,"MaxDD %":md_i,"Win %":wr_i
                                })
        if results:
            df_grid=pd.DataFrame(results).sort_values("Return %",ascending=False).head(10)
            st.dataframe(df_grid,use_container_width=True)
            st.download_button("Download CSV", df_grid.to_csv(index=False), "grid.csv")
        else:
            st.error("No valid combos found.")

    # Watchlist Summary
    st.markdown("---")
    st.markdown("## ⏰ Watchlist Summary")
    watch = st.text_area("Enter tickers", "AAPL, MSFT, TSLA, SPY, QQQ").upper()
    if st.button("📬 Generate Watchlist Summary"):
        tbl=[]
        for t in [x.strip() for x in watch.split(",") if x.strip()]:
            df_t=load_and_compute(t,ma_window,rsi_period,macd_fast,macd_slow,macd_signal)
            if df_t.empty:
                tbl.append({"Ticker":t,"Composite":None,"Signal":"N/A"})
                continue
            df_w,_,_,_=backtest(build_composite(df_t,ma_window,rsi_period))
            comp=int(df_w["Composite"].iloc[-1])
            sig = rec_map[int(df_w["Trade"].iloc[-1])]
            tbl.append({"Ticker":t,"Composite":comp,"Signal":sig})
        df_watch=pd.DataFrame(tbl).set_index("Ticker")
        st.dataframe(df_watch,use_container_width=True)
        for t in df_watch.index:
            df_t=load_and_compute(t,ma_window,rsi_period,macd_fast,macd_slow,macd_signal)
            if df_t.empty: continue
            df_c=build_composite(df_t,ma_window,rsi_period)
            last=df_c.iloc[-1]
            ma_s, rsi_s, macd_s = int(last["MA_Signal"]), int(last["RSI_Signal"]), int(last["MACD_Signal2"])
            try: rsi_v=float(last[f"RSI{rsi_period}"]); valid=True
            except: valid=False
            ma_txt={1:f"Price ↑ above {ma_window}-day MA.",0:"No crossover.",-1:f"Price ↓ below MA."}[ma_s]
            if valid:
                rsi_txt={1:f"RSI ({rsi_v:.1f}) <30 → oversold.",0:f"RSI ({rsi_v:.1f}) neutral.",-1:f"RSI ({rsi_v:.1f}) >70 → overbought."}[rsi_s]
            else:
                rsi_txt="RSI unavailable."
            macd_txt={1:"MACD ↑ signal line.",0:"No crossover.",-1:"MACD ↓ signal line."}[macd_s]
            with st.expander(f"🔎 {t} Reasoning ({df_watch.loc[t,'Signal']})"):
                st.write(f"- **MA:**  {ma_txt}")
                st.write(f"- **RSI:** {rsi_txt}")
                st.write(f"- **MACD:** {macd_txt}")
                st.write(f"- **Composite Score:** {df_watch.loc[t,'Composite']}")

# ───────────────────────────── Portfolio Simulator ─────────────────────────────
with tab_port:
    st.header("💼 Portfolio Simulator")
    st.markdown("Paste your positions CSV (Ticker,Quantity,Buy_Price) and simulate P&L + current signals.")

    example = "Ticker,Quantity,Buy_Price\nAAPL,10,150\nMSFT,5,280"
    raw = st.text_area("Positions CSV", example, height=120)
    if st.button("▶️ Simulate Portfolio"):
        try:
            pos = pd.read_csv(pd.compat.StringIO(raw))
        except:
            st.error("Failed to parse CSV. Ensure columns: Ticker,Quantity,Buy_Price")
            st.stop()

        out=[]
        for _,r in pos.iterrows():
            t=r["Ticker"].upper(); q=float(r["Quantity"]); bp=float(r["Buy_Price"])
            df = load_and_compute(t, ma_window, rsi_period, macd_fast, macd_slow, macd_signal)
            if df.empty: continue
            price = df["Close"].iloc[-1]
            pnl_d = (price - bp)*q
            pnl_p = (price/bp - 1)*100
            sig   = rec_map[int(build_composite(df,ma_window,rsi_period)["Trade"].iloc[-1])]
            out.append({"Ticker":t,"Qty":q,"Buy":bp,"Current":price,"P&L $":pnl_d,"P&L %":pnl_p,"Signal":sig})

        if out:
            df_out = pd.DataFrame(out).set_index("Ticker")
            st.dataframe(df_out,use_container_width=True)
            st.markdown(f"**Total P&L:** ${df_out['P&L $'].sum():.2f} — **Avg %:** {df_out['P&L %'].mean():.2f}%")
        else:
            st.info("No valid positions to simulate.")
