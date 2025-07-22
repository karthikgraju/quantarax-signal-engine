import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ───────────────────────────── Page Setup ─────────────────────────────
st.set_page_config(page_title="QuantaraX Composite Signals", layout="centered")
analyzer = SentimentIntensityAnalyzer()

# ───────────────────────────── Mode Toggle ─────────────────────────────
mode = st.sidebar.selectbox("Mode", ["Beginner", "Advanced"])

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

Each gives +1 (bull), –1 (bear), or 0 (neutral). Sum → **Composite**.  
Position = sign(composite): BUY / HOLD / SELL.

In Engine:
- **Signal of the Day** (Beginner only)  
- **Single‐Ticker Backtest**  
- **Watchlist Summary**  
- **Batch Backtest**, **Grid Search**, **Hyperparams** (Advanced only)  
- **News & Sentiment** overlay  
""")

# ───────────────────────────── Engine Tab ─────────────────────────────
with tab_engine:
    # ─────────── Defaults & Session State ───────────
    DEFAULTS = dict(ma_window=10, rsi_period=14, macd_fast=12, macd_slow=26, macd_signal=9)
    for k,v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ─────────── Sidebar Controls ───────────
    st.sidebar.header("Controls")
    if st.sidebar.button("🔄 Reset to defaults"):
        for k in DEFAULTS:
            st.session_state[k] = DEFAULTS[k]

    # ─────────── Hyperparameter sliders (Advanced) ───────────
    if mode == "Advanced":
        st.sidebar.header("Indicator Parameters")
        ma_window   = st.sidebar.slider("MA window",      5, 50, st.session_state["ma_window"],   key="ma_window")
        rsi_period  = st.sidebar.slider("RSI lookback",   5, 30, st.session_state["rsi_period"],  key="rsi_period")
        macd_fast   = st.sidebar.slider("MACD fast span", 5, 20, st.session_state["macd_fast"],   key="macd_fast")
        macd_slow   = st.sidebar.slider("MACD slow span",20, 40, st.session_state["macd_slow"],   key="macd_slow")
        macd_signal = st.sidebar.slider("MACD sig span",  5, 20, st.session_state["macd_signal"], key="macd_signal")
    else:
        ma_window   = st.session_state["ma_window"]
        rsi_period  = st.session_state["rsi_period"]
        macd_fast   = st.session_state["macd_fast"]
        macd_slow   = st.session_state["macd_slow"]
        macd_signal = st.session_state["macd_signal"]

    st.title("🚀 QuantaraX — Composite Signal Engine")

    # ─────────── Data Loading & Indicators ───────────
    @st.cache_data
    def load_and_compute(ticker, ma_w, rsi_p, mf, ms, sig):
        df = yf.download(ticker, period="6mo", progress=False)
        if df.empty or "Close" not in df:
            return pd.DataFrame()
        df[f"MA{ma_w}"] = df["Close"].rolling(ma_w).mean()
        delta = df["Close"].diff()
        up, dn = delta.clip(lower=0), -delta.clip(upper=0)
        df[f"RSI{rsi_p}"] = 100 - 100/(1 + up.ewm(com=rsi_p-1).mean() / dn.ewm(com=rsi_p-1).mean())
        ema_f = df["Close"].ewm(span=mf).mean()
        ema_s = df["Close"].ewm(span=ms).mean()
        macd  = ema_f - ema_s
        df["MACD"] = macd
        df["MACD_Signal"] = macd.ewm(span=sig).mean()
        cols = [f"MA{ma_w}", f"RSI{rsi_p}", "MACD", "MACD_Signal"]
        return df.dropna(subset=cols).reset_index()

    # ─────────── Build Composite Signal ───────────
    def build_composite(df, ma_w, rsi_p):
        n = len(df)
        close = df["Close"].to_numpy()
        ma    = df[f"MA{ma_w}"].to_numpy()
        rsi   = df[f"RSI{rsi_p}"].to_numpy()
        macd  = df["MACD"].to_numpy()
        sig   = df["MACD_Signal"].to_numpy()

        ma_sig = np.zeros(n,int)
        rsi_sig= np.zeros(n,int)
        macd_sig2 = np.zeros(n,int)
        comp   = np.zeros(n,int)
        trade  = np.zeros(n,int)

        for i in range(1,n):
            if close[i-1]<ma[i-1]<close[i]:    ma_sig[i]=1
            elif close[i-1]>ma[i-1]>close[i]:  ma_sig[i]=-1
            if rsi[i]<30:   rsi_sig[i]=1
            elif rsi[i]>70: rsi_sig[i]=-1
            if macd[i-1]<sig[i-1]<macd[i]:      macd_sig2[i]=1
            elif macd[i-1]>sig[i-1]>macd[i]:    macd_sig2[i]=-1
            comp[i] = ma_sig[i] + rsi_sig[i] + macd_sig2[i]
            trade[i] = np.sign(comp[i])

        df["MA_Signal"]    = ma_sig
        df["RSI_Signal"]   = rsi_sig
        df["MACD_Signal2"] = macd_sig2
        df["Composite"]    = comp
        df["Trade"]        = trade
        return df

    # ─────────── Backtest & Metrics ───────────
    def backtest(df):
        df = df.copy()
        df["Return"] = df["Close"].pct_change().fillna(0)
        df["Position"] = df["Trade"].shift().fillna(0).clip(0,1)
        df["StratRet"] = df["Position"] * df["Return"]
        df["CumBH"]    = (1 + df["Return"]).cumprod()
        df["CumStrat"] = (1 + df["StratRet"]).cumprod()
        dd = df["CumStrat"]/df["CumStrat"].cummax() - 1
        sharpe = df["StratRet"].mean()/df["StratRet"].std()*np.sqrt(252) if df["StratRet"].std() else np.nan
        win_rt = (df["StratRet"]>0).mean()*100
        return df, dd.min()*100, sharpe, win_rt

    # ─────────── Signal of the Day (Beginner) ───────────
    if mode=="Beginner":
        st.subheader("🔔 Signal of the Day")
        wl_default = st.text_area("Watchlist (comma-separated)", "AAPL, MSFT, TSLA, SPY, QQQ")
        tickers = [x.strip().upper() for x in wl_default.split(",") if x.strip()]
        scores = []
        for t in tickers:
            df = load_and_compute(t, ma_window, rsi_period, macd_fast, macd_slow, macd_signal)
            if df.empty: continue
            df = build_composite(df, ma_window, rsi_period)
            scores.append((t, int(df["Composite"].iloc[-1])))
        if scores:
            df_sig = pd.DataFrame(scores, columns=["Ticker","Score"]).sort_values("Score", ascending=False)
            buy, sell = df_sig.iloc[0], df_sig.iloc[-1]
            c1, c2 = st.columns(2)
            c1.metric("Top BUY",  buy.Ticker,  f"Score {buy.Score}")
            c2.metric("Top SELL", sell.Ticker, f"Score {sell.Score}")
        else:
            st.info("No valid watchlist data.")

    # ─────────── Single-Ticker Backtest & News ───────────
    st.markdown("## Single‐Ticker Backtest")
    ticker = st.text_input("Ticker", "AAPL").upper()
    if ticker:
        df_raw = load_and_compute(ticker, ma_window, rsi_period, macd_fast, macd_slow, macd_signal)
        if df_raw.empty:
            st.error("No data.")
        else:
            info = yf.Ticker(ticker).info
            price = info.get("regularMarketPrice")
            if price: st.subheader(f"💲 {ticker}: ${price:.2f}")

            news = getattr(yf.Ticker(ticker), "news", []) or []
            if news:
                st.markdown("### 📰 Recent News & Sentiment")
                for art in news[:3]:
                    txt   = art.get("summary") or art.get("title","")
                    score = analyzer.polarity_scores(txt)["compound"]
                    emoji = "🔺" if score>0.1 else ("🔻" if score<-0.1 else "➖")
                    st.markdown(f"- [{art['title']}]({art['link']}) {emoji}")

            df = build_composite(df_raw, ma_window, rsi_period)
            df_bt, max_dd, sharpe, win_rt = backtest(df)

            rec = {1:"BUY",0:"HOLD",-1:"SELL"}[int(df_bt["Trade"].iloc[-1])]
            st.success(f"Signal: **{rec}**")

            st.markdown(f"""
- **BH Return:** {(df_bt["CumBH"].iloc[-1]-1)*100:.2f}%  
- **Strat Return:** {(df_bt["CumStrat"].iloc[-1]-1)*100:.2f}%  
- **Sharpe:** {sharpe:.2f}  
- **Max Drawdown:** {max_dd:.2f}%  
- **Win Rate:** {win_rt:.1f}%  
""")

            fig, axes = plt.subplots(2,1,figsize=(8,6), sharex=True)
            axes[0].plot(df_bt["Close"], label="Close")
            axes[0].plot(df_bt[f"MA{ma_window}"], label=f"MA{ma_window}")
            axes[0].legend(); axes[0].set_title("Price & MA")
            axes[1].plot(df_bt["CumStrat"], label="Strategy")
            axes[1].plot(df_bt["CumBH"],    label="Buy & Hold", linestyle=":")
            axes[1].legend(); axes[1].set_title("Equity Curves")
            plt.xticks(rotation=45); plt.tight_layout()
            st.pyplot(fig)

    # ─────────── Watchlist Summary ───────────
    st.markdown("---")
    st.markdown("## ⏰ Watchlist Summary")
    wl2 = st.text_area("Tickers", "AAPL, MSFT, TSLA").upper()
    if st.button("Generate Summary"):
        tbl = []
        for t in [x.strip() for x in wl2.split(",") if x.strip()]:
            df = load_and_compute(t, ma_window, rsi_period, macd_fast, macd_slow, macd_signal)
            if df.empty:
                tbl.append({"Ticker":t,"Score":None,"Signal":"N/A"})
            else:
                df2 = build_composite(df, ma_window, rsi_period)
                tbl.append({
                    "Ticker":t,
                    "Score": int(df2["Composite"].iloc[-1]),
                    "Signal": {1:"BUY",0:"HOLD",-1:"SELL"}[int(df2["Trade"].iloc[-1])]
                })
        st.table(pd.DataFrame(tbl).set_index("Ticker"))

    # ─────────── Advanced-Only: Batch & Grid ───────────
    if mode=="Advanced":
        st.markdown("---")
        st.markdown("## Batch Backtest")
        batch = st.text_area("Tickers for batch", "AAPL, MSFT, TSLA, SPY, QQQ", key="batch").upper()
        if st.button("▶️ Run Batch"):
            perf = []
            for t in [x.strip() for x in batch.split(",") if x.strip()]:
                df = load_and_compute(t, ma_window, rsi_period, macd_fast, macd_slow, macd_signal)
                if df.empty: continue
                df2, md, sh, wr = backtest(build_composite(df, ma_window, rsi_period))
                perf.append({
                    "Ticker":     t,
                    "Score":      int(df2["Composite"].iloc[-1]),
                    "Signal":     {1:"BUY",0:"HOLD",-1:"SELL"}[int(df2["Trade"].iloc[-1])],
                    "BH %":       (df2["CumBH"].iloc[-1]-1)*100,
                    "Strat %":    (df2["CumStrat"].iloc[-1]-1)*100,
                    "Sharpe":     sh,
                    "Max DD %":   md,
                    "Win Rate %": wr
                })
            if perf:
                df_perf = pd.DataFrame(perf).set_index("Ticker")
                st.dataframe(df_perf, use_container_width=True)
                st.download_button("Download CSV", df_perf.to_csv(), "batch.csv")
            else:
                st.error("No valid data.")

        st.markdown("---")
        st.markdown("## 🛠️ Grid Search")
        ma_list  = st.sidebar.multiselect("MA windows",     [5,10,15,20,30], default=[ma_window], key="grid_ma")
        rsi_list = st.sidebar.multiselect("RSI lookbacks",  [7,14,21,28],   default=[rsi_period], key="grid_rsi")
        mf_list  = st.sidebar.multiselect("MACD fast spans",[5,8,12,16,20], default=[macd_fast], key="grid_mf")
        ms_list  = st.sidebar.multiselect("MACD slow spans",[20,26,32,40], default=[macd_slow], key="grid_ms")
        sig_list = st.sidebar.multiselect("MACD sig spans", [5,9,12,16],   default=[macd_signal], key="grid_sig")

        if st.button("🏃‍♂️ Run Grid Search"):
            if not ticker:
                st.error("Enter a ticker above"); st.stop()
            results = []
            for mw in ma_list:
                for rp in rsi_list:
                    for mf in mf_list:
                        for ms in ms_list:
                            for s in sig_list:
                                df = load_and_compute(ticker, mw, rp, mf, ms, s)
                                if df.empty: continue
                                df2, md, sh, wr = backtest(build_composite(df, mw, rp))
                                results.append({
                                    "MA":mw, "RSI":rp, "MF":mf, "MS":ms, "SIG":s,
                                    "Return %": (df2["CumStrat"].iloc[-1]-1)*100,
                                    "Sharpe": sh,
                                    "Max DD": md,
                                    "Win %": wr
                                })
            if results:
                df_grid = pd.DataFrame(results).sort_values("Return %", ascending=False).head(10)
                st.dataframe(df_grid, use_container_width=True)
                st.download_button("Download CSV", df_grid.to_csv(index=False), "grid.csv")
            else:
                st.error("No valid combos.")
