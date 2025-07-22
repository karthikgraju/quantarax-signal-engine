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

Each yields +1/–1/0, summed (–3…+3) → **Composite**, then `sign()` → BUY/HOLD/SELL.

Under **Engine** you can:
- Backtest a single ticker  
- Batch-test many tickers  
- Grid-search parameters  
- View a watchlist with reasoning  
- See recent news & sentiment  
- Export raw signals CSV  
- View key stats (52-week high/low, P/E, dividend yield)
""")

# ───────────────────────────── Engine Tab ─────────────────────────────
with tab_engine:

    # Defaults & Session State
    DEFAULTS = dict(ma_window=10, rsi_period=14, macd_fast=12, macd_slow=26, macd_signal=9)
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(k, v)

    # Sidebar
    st.sidebar.header("Controls")
    if st.sidebar.button("🔄 Reset to defaults"):
        for k, v in DEFAULTS.items():
            st.session_state[k] = v

    st.sidebar.header("Indicator Parameters")
    ma_window   = st.sidebar.slider("MA window", 5, 50, st.session_state["ma_window"],   key="ma_window")
    rsi_period  = st.sidebar.slider("RSI lookback", 5, 30, st.session_state["rsi_period"],  key="rsi_period")
    macd_fast   = st.sidebar.slider("MACD fast span", 5, 20, st.session_state["macd_fast"],   key="macd_fast")
    macd_slow   = st.sidebar.slider("MACD slow span",20, 40, st.session_state["macd_slow"],   key="macd_slow")
    macd_signal = st.sidebar.slider("MACD sig span", 5, 20, st.session_state["macd_signal"], key="macd_signal")

    st.title("🚀 QuantaraX — Composite Signal Engine")
    st.write("MA + RSI + MACD Composite Signals & Backtest")

    # ─────────── Load & Compute ───────────
    @st.cache_data(show_spinner=False)
    def load_and_compute(ticker, ma_w, rsi_p, mf, ms, sig):
        df = yf.download(ticker, period="6mo", progress=False)
        if df.empty or "Close" not in df: return pd.DataFrame()
        df.reset_index(inplace=True)

        # MA
        df[f"MA{ma_w}"] = df["Close"].rolling(ma_w).mean()
        # RSI
        d = df["Close"].diff()
        up = d.clip(lower=0); dn = -d.clip(upper=0)
        ema_up = up.ewm(com=rsi_p-1, adjust=False).mean()
        ema_dn = dn.ewm(com=rsi_p-1, adjust=False).mean()
        df[f"RSI{rsi_p}"] = 100 - 100/(1 + ema_up/ema_dn)
        # MACD
        ema_f = df["Close"].ewm(span=mf, adjust=False).mean()
        ema_s = df["Close"].ewm(span=ms, adjust=False).mean()
        macd  = ema_f - ema_s
        df["MACD"] = macd
        df["MACD_Signal"] = macd.ewm(span=sig, adjust=False).mean()

        df.dropna(subset=[f"MA{ma_w}", f"RSI{rsi_p}", "MACD", "MACD_Signal"], inplace=True)
        return df

    # ─────────── Composite ───────────
    def build_composite(df, ma_w, rsi_p):
        n = len(df)
        close = df["Close"].to_numpy()
        ma    = df[f"MA{ma_w}"].to_numpy()
        rsi   = df[f"RSI{rsi_p}"].to_numpy()
        macd  = df["MACD"].to_numpy()
        sig   = df["MACD_Signal"].to_numpy()
        ma_s, rsi_s, macd_s = [np.zeros(n,int) for _ in range(3)]
        comp = np.zeros(n,int)
        trade= np.zeros(n,int)
        for i in range(1,n):
            if close[i-1]<ma[i-1]<close[i]: ma_s[i]=1
            elif close[i-1]>ma[i-1]>close[i]: ma_s[i]=-1
            if rsi[i]<30: rsi_s[i]=1
            elif rsi[i]>70: rsi_s[i]=-1
            if macd[i-1]<sig[i-1]<macd[i]: macd_s[i]=1
            elif macd[i-1]>sig[i-1]>macd[i]: macd_s[i]=-1
            comp[i]=ma_s[i]+rsi_s[i]+macd_s[i]
            trade[i]=np.sign(comp[i])
        df["MA_Signal"]=ma_s
        df["RSI_Signal"]=rsi_s
        df["MACD_Signal2"]=macd_s
        df["Composite"]=comp
        df["Trade"]=trade
        return df

    # ─────────── Backtest ───────────
    def backtest(df):
        df = df.copy()
        df["Return"]=df["Close"].pct_change().fillna(0)
        df["Position"]=df["Trade"].shift(1).fillna(0).clip(0,1)
        df["StratRet"]=df["Position"]*df["Return"]
        df["CumBH"]=(1+df["Return"]).cumprod()
        df["CumStrat"]=(1+df["StratRet"]).cumprod()
        dd = df["CumStrat"]/df["CumStrat"].cummax()-1
        sharpe = (df["StratRet"].mean()/df["StratRet"].std()*np.sqrt(252)
                  if df["StratRet"].std()>0 else np.nan)
        win_rt = (df["StratRet"]>0).mean()*100
        return df, dd.min()*100, sharpe, win_rt

    # ─────────────────── Single‐Ticker ───────────────────
    st.markdown("## Single‐Ticker Backtest")
    ticker = st.text_input("Ticker", "AAPL").upper()

    if ticker:
        tk   = yf.Ticker(ticker)
        info = tk.info
        price = info.get("regularMarketPrice")
        if price is not None:
            st.subheader(f"💲 {ticker}: ${price:.2f} "
                         f"({info.get('regularMarketChangePercent',0):+.2f}%)")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("52wk High", f"${info.get('fiftyTwoWeekHigh',0):.2f}")
            c2.metric("52wk Low",  f"${info.get('fiftyTwoWeekLow',0):.2f}")
            c3.metric("P/E",       info.get("trailingPE","N/A"))
            c4.metric("Div Yield", info.get("dividendYield","N/A"))

        news = getattr(tk, "news", []) or []
        if news:
            st.markdown("### 📰 Recent News & Sentiment")
            for art in news[:5]:
                t, l = art.get("title",""), art.get("link","")
                if not (t and l): continue
                txt   = art.get("summary", t)
                score = analyzer.polarity_scores(txt)["compound"]
                emoji = "🔺" if score>0.1 else "🔻" if score< -0.1 else "➖"
                st.markdown(f"- [{t}]({l}) {emoji}")

    if st.button("▶️ Run Composite Backtest"):
        df0 = load_and_compute(ticker, ma_window, rsi_period,
                               macd_fast, macd_slow, macd_signal)
        if df0.empty:
            st.error("No data.")
            st.stop()

        dfc, max_dd, sharpe, win_rt = backtest(build_composite(df0, ma_window, rsi_period))

        rec = {1:"🟢 BUY",0:"🟡 HOLD",-1:"🔴 SELL"}[int(dfc["Trade"].iloc[-1])]
        st.success(f"**{ticker}**: {rec}")

        # Reasoning
        i = -1
        texts = {
            "MA":    {1:f"Price ↑ above {ma_window}-day MA",0:"No MA crossover", -1:f"Price ↓ below MA"},
            "RSI":   {1:"RSI <30 oversold",0:"RSI neutral", -1:"RSI >70 overbought"},
            "MACD":  {1:"MACD ↑ signal",0:"No MACD crossover", -1:"MACD ↓ signal"},
        }
        st.markdown("#### 🔎 Why This Signal?")
        st.write(f"- **MA:**   {texts['MA'][int(dfc['MA_Signal'].iloc[i])]]}")
        st.write(f"- **RSI:**  {texts['RSI'][int(dfc['RSI_Signal'].iloc[i])]]}")
        st.write(f"- **MACD:** {texts['MACD'][int(dfc['MACD_Signal2'].iloc[i])]]}")
        st.write(f"- **Composite Score:** {int(dfc['Composite'].iloc[i])}")

        # Performance
        st.markdown(f"""
- **Buy & Hold:**   {(dfc['CumBH'].iloc[i]-1)*100:.2f}%  
- **Strategy:**     {(dfc['CumStrat'].iloc[i]-1)*100:.2f}%  
- **Sharpe:**       {sharpe:.2f}  
- **Max Drawdown:** {max_dd:.2f}%  
- **Win Rate:**     {win_rt:.1f}%  
""")

        # Charts
        fig, axs = plt.subplots(3,1,figsize=(10,12), sharex=True)
        axs[0].plot(dfc["Date"], dfc["Close"], label="Close")
        axs[0].plot(dfc["Date"], dfc[f"MA{ma_window}"], label=f"MA{ma_window}")
        axs[0].set_title("Price & MA"); axs[0].legend()
        axs[1].bar(dfc["Date"], dfc["Composite"], color="purple")
        axs[1].set_title("Composite Vote")
        axs[2].plot(dfc["Date"], dfc["CumBH"], ":", label="BH")
        axs[2].plot(dfc["Date"], dfc["CumStrat"], "-", label="Strat")
        axs[2].set_title("Equity Curves"); axs[2].legend()
        plt.xticks(rotation=45); plt.tight_layout()
        st.pyplot(fig)

        # Export signals
        signals = dfc[["Date","Close",f"MA{ma_window}",f"RSI{rsi_period}",
                       "MACD","MACD_Signal","Composite","Trade"]]
        st.download_button("📥 Download Signals CSV",
                           signals.to_csv(index=False),
                           f"{ticker}_signals.csv")

    # ─────────────────── Batch Backtest ───────────────────
    st.markdown("---")
    st.markdown("## Batch Backtest")
    batch = st.text_area("Tickers (comma-separated)", "AAPL, MSFT, TSLA").upper()
    if st.button("▶️ Run Batch Backtest"):
        perf=[]
        for t in [x.strip() for x in batch.split(",") if x.strip()]:
            dfb = load_and_compute(t, ma_window, rsi_period, macd_fast, macd_slow, macd_signal)
            if dfb.empty: continue
            dfc, md, sh, wr = backtest(build_composite(dfb, ma_window, rsi_period))
            perf.append({
                "Ticker": t,
                "Composite": int(dfc["Composite"].iloc[-1]),
                "Signal": {1:"BUY",0:"HOLD",-1:"SELL"}[int(dfc["Trade"].iloc[-1])],
                "BH %": (dfc["CumBH"].iloc[-1]-1)*100,
                "Strat %": (dfc["CumStrat"].iloc[-1]-1)*100,
                "Sharpe": sh,
                "Max DD": md,
                "Win %": wr
            })
        if perf:
            dfp = pd.DataFrame(perf).set_index("Ticker")
            st.dataframe(dfp, use_container_width=True)
            st.download_button("📥 Download Batch CSV",
                               dfp.to_csv(), "batch.csv")
        else:
            st.error("No valid tickers/data.")

    # ─────────────────── Grid Search ───────────────────
    st.markdown("---")
    st.markdown("## 🛠️ Hyperparameter Optimization")
    ma_list  = st.sidebar.multiselect("MA windows",[5,10,20,50], [ma_window], key="grid_ma")
    rsi_list = st.sidebar.multiselect("RSI periods",[7,14,21], [rsi_period], key="grid_rsi")
    mf_list  = st.sidebar.multiselect("MACD fast",[8,12,16], [macd_fast], key="grid_mf")
    ms_list  = st.sidebar.multiselect("MACD slow",[20,26,34], [macd_slow], key="grid_ms")
    s_list   = st.sidebar.multiselect("MACD sig",[5,9,12], [macd_signal], key="grid_sig")
    if st.button("🏃‍♂️ Run Grid Search"):
        if not ticker:
            st.error("Enter ticker"); st.stop()
        dfb = load_and_compute(ticker,ma_window,rsi_period,macd_fast,macd_slow,macd_signal)
        if dfb.empty:
            st.error("No data"); st.stop()
        rows=[]
        with st.spinner("Testing combos…"):
            for mw in ma_list:
                for rp in rsi_list:
                    for mf in mf_list:
                        for ms in ms_list:
                            for ss in s_list:
                                dfc,_md,_sh,_wr = backtest(build_composite(
                                    load_and_compute(ticker,mw,rp,mf,ms,ss),
                                    mw, rp))
                                rows.append({
                                    "MA":mw,"RSI":rp,"MF":mf,"MS":ms,"SIG":ss,
                                    "Return":(dfc["CumStrat"].iloc[-1]-1)*100,
                                    "Sharpe":_sh,"MaxDD":_md,"Win%":_wr
                                })
        if rows:
            dg = pd.DataFrame(rows).sort_values("Return",ascending=False).head(10)
            st.dataframe(dg, use_container_width=True)
            st.download_button("📥 Download Grid CSV",
                               dg.to_csv(index=False), "grid.csv")
        else:
            st.error("No valid combos.")

    # ─────────────────── Watchlist Summary ───────────────────
    st.markdown("---")
    st.markdown("## ⏰ Watchlist Summary")
    watch = st.text_area("Watchlist tickers", "AAPL, MSFT, TSLA").upper()
    if st.button("📬 Generate Watchlist Summary"):
        tbl = []
        for t in [x.strip() for x in watch.split(",") if x.strip()]:
            dfw = load_and_compute(t, ma_window, rsi_period, macd_fast, macd_slow, macd_signal)
            if dfw.empty:
                tbl.append({"Ticker":t,"Composite":None,"Signal":"N/A"})
                continue
            dfc, *_ = backtest(build_composite(dfw, ma_window, rsi_period))
            tbl.append({
                "Ticker":t,
                "Composite": int(dfc["Composite"].iloc[-1]),
                "Signal": {1:"BUY",0:"HOLD",-1:"SELL"}[int(dfc["Trade"].iloc[-1])]
            })
        dfw = pd.DataFrame(tbl).set_index("Ticker")
        st.dataframe(dfw, use_container_width=True)
        for t in dfw.index:
            dfw0 = load_and_compute(t, ma_window, rsi_period, macd_fast, macd_slow, macd_signal)
            if dfw0.empty: continue
            dfc = build_composite(dfw0, ma_window, rsi_period)
            last = dfc.iloc[-1]
            texts = {
                "MA":    {1:f"Price ↑ above MA",0:"No crossover", -1:"Price ↓ below MA"},
                "RSI":   {1:"RSI <30",0:"RSI neutral", -1:"RSI >70"},
                "MACD":  {1:"MACD ↑",0:"No crossover", -1:"MACD ↓"}
            }
            with st.expander(f"🔎 {t} Reasoning ({dfw.loc[t,'Signal']})"):
                st.write(f"- **MA:**   {texts['MA'][int(last['MA_Signal'])]]}")
                st.write(f"- **RSI:**  {texts['RSI'][int(last['RSI_Signal'])]]}")
                st.write(f"- **MACD:** {texts['MACD'][int(last['MACD_Signal2'])]]}")
                st.write(f"- **Composite Score:** {int(last['Composite'])}")
