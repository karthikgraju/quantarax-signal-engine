import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import feedparser  # for RSS fallback

# ───────────────────────────── Page Setup ─────────────────────────────
st.set_page_config(page_title="QuantaraX Composite Signals BETA v2", layout="centered")
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
    st.markdown(r"""
Welcome to **QuantaraX**, the MVP from a hands-on team of quants, data scientists, and former traders on a mission to **democratize** institutional-grade quantitative tools for **every** investor.

---

## 🎯 Our Purpose & Mission

We believe retail investors deserve the same rigor, clarity, and transparency that professional funds enjoy.  
**QuantaraX** exists to:

- **Demystify** technical analysis by **combining** multiple indicators into one clear, composite recommendation.  
- **Reduce emotional bias** by delivering consistent, rules-based signals.  
- **Empower** users through **education**, exposing the “why” behind every BUY, HOLD, or SELL.  
- **Accelerate** decision-making with live prices, sentiment-weighted news, and portfolio simulations.  
- **Scale** from a weekend MVP to a full platform with real-time alerts, multi-asset support, and broker connectivity.

---

## 🔧 Choosing Slider Settings

Every slider trades off **responsiveness** vs. **smoothness**.  
| Slider                 | What it does                                | If you want…                                                              |
|------------------------|---------------------------------------------|---------------------------------------------------------------------------|
| **MA window**          | # of days for moving average               | • **Lower** (5–10) → more responsive • **Higher** (20–50) → smoother      |
| **RSI lookback**       | Period for RSI’s EMA calculation           | • **Short** (5–10) → choppier • **Long** (20–30) → stable                 |
| **MACD spans**         | Fast/slow/signal spans                     | • **Lower** → quicker signals • **Higher** → fewer whipsaws              |

Tip: start with defaults (MA=10, RSI=14, MACD=12/26/9), tweak one at a time.

---

## 🏆 Objectives

1. Polished MVP by week’s end.  
2. Onboard 100+ beta users in 30 days.  
3. Real-time streaming & alerts (Q3).  
4. Expand to crypto, forex, alt data (Q4).  
5. Community-driven features.  

Made in Toronto, Canada by KG
""")

# ───────────────────────────── Engine Tab ─────────────────────────────
with tab_engine:

    # ─────────── Defaults & Session State ───────────
    DEFAULTS = dict(
        ma_window=10,
        rsi_period=14,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9
    )
    for k,v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ─────────── Sidebar Controls ───────────
    st.sidebar.header("Controls")
    if st.sidebar.button("🔄 Reset to defaults"):
        for k,v in DEFAULTS.items():
            st.session_state[k] = v

    st.sidebar.header("Indicator Parameters")
    ma_window   = st.sidebar.slider("MA window",      5, 50, st.session_state["ma_window"],  key="ma_window")
    rsi_period  = st.sidebar.slider("RSI lookback",   5, 30, st.session_state["rsi_period"], key="rsi_period")
    macd_fast   = st.sidebar.slider("MACD fast span", 5, 20, st.session_state["macd_fast"],  key="macd_fast")
    macd_slow   = st.sidebar.slider("MACD slow span",20, 40, st.session_state["macd_slow"],  key="macd_slow")
    macd_signal = st.sidebar.slider("MACD sig span",  5, 20, st.session_state["macd_signal"],key="macd_signal")

    # ─────────── Asset Type Selector ───────────────────
    asset_type = st.sidebar.selectbox(
        "Asset Type",
        ["Stock","Crypto"],
        help="Crypto tickers like BTC/USDT → BTC-USD under the hood"
    )

    # ─────────── Profit/Loss Sliders ───────────
    profit_target = st.sidebar.slider("Profit target (%)",1,100,5)
    loss_limit    = st.sidebar.slider("Loss limit (%)",1,100,5)

    st.title("🚀 QuantaraX — Composite Signal Engine")
    st.write("MA + RSI + MACD Composite Signals & Backtest")

    # ─────────── Data Loading ───────────
    @st.cache_data(show_spinner=False)
    def load_and_compute(ticker, ma_w, rsi_p, mf, ms, sig, asset_type):
        # convert crypto symbol if needed
        yf_sym = ticker
        if asset_type=="Crypto" and "/" in ticker:
            base, quote = ticker.split("/")
            quote = quote.upper()
            if quote in ("USDT","USD"):
                yf_sym = f"{base.upper()}-USD"
            else:
                yf_sym = f"{base.upper()}-{quote}"
        df = yf.download(yf_sym, period="6mo", progress=False)
        if df.empty or "Close" not in df:
            return pd.DataFrame()

        # MA
        ma_col = f"MA{ma_w}"
        df[ma_col] = df["Close"].rolling(ma_w).mean()
        # RSI
        d = df["Close"].diff()
        up = d.clip(lower=0); dn = -d.clip(upper=0)
        ema_up   = up.ewm(com=rsi_p-1, adjust=False).mean()
        ema_down = dn.ewm(com=rsi_p-1, adjust=False).mean()
        rsi_col  = f"RSI{rsi_p}"
        df[rsi_col] = 100 - 100/(1 + ema_up/ema_down)
        # MACD
        ema_f    = df["Close"].ewm(span=mf, adjust=False).mean()
        ema_s    = df["Close"].ewm(span=ms, adjust=False).mean()
        macd     = ema_f - ema_s
        macd_sig = macd.ewm(span=sig, adjust=False).mean()
        df["MACD"] = macd; df["MACD_Signal"] = macd_sig
        # Drop NAs
        for col in (ma_col, rsi_col, "MACD", "MACD_Signal"):
            if col not in df: return pd.DataFrame()
        return df.dropna(subset=[ma_col, rsi_col, "MACD", "MACD_Signal"]).reset_index(drop=True)

    # ─────────── Composite Signals ───────────
    def build_composite(df, ma_w, rsi_p):
        n = len(df)
        close, ma = df["Close"].to_numpy(), df[f"MA{ma_w}"].to_numpy()
        rsi, macd, sig = df[f"RSI{rsi_p}"].to_numpy(), df["MACD"].to_numpy(), df["MACD_Signal"].to_numpy()
        ma_sig = np.zeros(n,int); rsi_sig = np.zeros(n,int); macd_sig2 = np.zeros(n,int)
        comp, trade = np.zeros(n,int), np.zeros(n,int)
        for i in range(1,n):
            if close[i-1]<ma[i-1] and close[i]>ma[i]:   ma_sig[i]=1
            elif close[i-1]>ma[i-1] and close[i]<ma[i]: ma_sig[i]=-1
            if rsi[i]<30:   rsi_sig[i]=1
            elif rsi[i]>70: rsi_sig[i]=-1
            if macd[i-1]<sig[i-1] and macd[i]>sig[i]:   macd_sig2[i]=1
            elif macd[i-1]>sig[i-1] and macd[i]<sig[i]: macd_sig2[i]=-1
            comp[i] = ma_sig[i]+rsi_sig[i]+macd_sig2[i]
            trade[i] = np.sign(comp[i])
        df["MA_Signal"], df["RSI_Signal"] = ma_sig, rsi_sig
        df["MACD_Signal2"], df["Composite"] = macd_sig2, comp
        df["Trade"] = trade
        return df

    # ─────────── Backtest ───────────
    def backtest(df):
        df = df.copy()
        df["Return"]   = df["Close"].pct_change().fillna(0)
        df["Position"] = df["Trade"].shift(1).fillna(0).clip(0,1)
        df["StratRet"] = df["Position"] * df["Return"]
        df["CumBH"]    = (1 + df["Return"]).cumprod()
        df["CumStrat"] = (1 + df["StratRet"]).cumprod()
        dd        = df["CumStrat"]/df["CumStrat"].cummax() - 1
        max_dd    = dd.min()*100
        sd        = df["StratRet"].std()
        sharpe    = (df["StratRet"].mean()/sd*np.sqrt(252)) if sd else np.nan
        win_rate  = (df["StratRet"]>0).mean()*100
        return df, max_dd, sharpe, win_rate

    # ─────────────────── Single‐Ticker Backtest ───────────────────
    st.markdown("## Single‐Ticker Backtest")
    ticker = st.text_input("Ticker (e.g. AAPL or BTC/USDT)","AAPL").upper()

    if ticker:
        # Live price
        yf_sym = ticker.replace("/","-") if asset_type=="Crypto" else ticker
        hist = yf.download(yf_sym, period="1d", progress=False)
        price = hist["Close"].iloc[-1] if not hist.empty else None
        if price is not None:
            st.subheader(f"💲 Live Price: ${price:.2f}")

        # Dual‐source news
        raw = getattr(yf.Ticker(yf_sym), "news", []) or []
        shown = 0
        if raw:
            st.markdown("### 📰 Recent News & Sentiment (YFinance)")
            for art in raw:
                t,l = art.get("title",""), art.get("link","")
                if not (t and l): continue
                txt = art.get("summary",t)
                score = analyzer.polarity_scores(txt)["compound"]
                emo = "🔺" if score>0.1 else ("🔻" if score<-0.1 else "➖")
                st.markdown(f"- [{t}]({l}) {emo}")
                shown += 1
                if shown>=5: break
        if shown==0:
            st.markdown("### 📰 Recent News (RSS)")
            feed = feedparser.parse(f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={yf_sym}&region=US&lang=en-US")
            for e in feed.entries:
                st.markdown(f"- [{e.title}]({e.link})")
                shown+=1
                if shown>=5: break
        if shown==0:
            st.info("No recent news found.")

    if st.button("▶️ Run Composite Backtest"):
        df_raw = load_and_compute(ticker, ma_window, rsi_period, macd_fast, macd_slow, macd_signal, asset_type)
        if df_raw.empty:
            st.error(f"No data for '{ticker}'"); st.stop()
        df_c, max_dd, sharpe, win_rt = backtest(build_composite(df_raw, ma_window, rsi_period))
        rec = rec_map[int(df_c["Trade"].iloc[-1])]
        st.success(f"**{ticker}**: {rec}")

        # Show metrics & charts
        fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(10,12), sharex=True)
        ax1.plot(df_c["Close"], label="Close")
        ax1.plot(df_c[f"MA{ma_window}"], label=f"MA{ma_window}")
        ax1.legend(); ax1.set_title("Price & MA")
        ax2.bar(df_c.index, df_c["Composite"])
        ax2.set_title("Composite Score")
        ax3.plot(df_c["CumBH"],":", label="Buy&Hold")
        ax3.plot(df_c["CumStrat"],"-", label="Strategy")
        ax3.legend(); ax3.set_title("Equity Curves")
        plt.xticks(rotation=45); plt.tight_layout()
        st.pyplot(fig)

        st.markdown(f"""
- **Buy & Hold:**    {(df_c['CumBH'].iloc[-1]-1)*100:.2f}%  
- **Strategy:**      {(df_c['CumStrat'].iloc[-1]-1)*100:.2f}%  
- **Sharpe:**        {sharpe:.2f}  
- **Max Drawdown:**  {max_dd:.2f}%  
- **Win Rate:**      {win_rt:.1f}%  
""")

    # ─────────── Batch Backtest ───────────
    st.markdown("---\n## Batch Backtest")
    batch = st.text_area("Tickers (comma-separated)", "AAPL, MSFT, TSLA, SPY, QQQ").upper()
    if st.button("▶️ Run Batch Backtest"):
        perf=[]
        for t in [x.strip() for x in batch.split(",") if x.strip()]:
            df_t = load_and_compute(t,ma_window,rsi_period,macd_fast,macd_slow,macd_signal,asset_type)
            if df_t.empty: continue
            df_tc, md, sh, wr = backtest(build_composite(df_t, ma_window, rsi_period))
            perf.append({
                "Ticker":t,
                "Composite": int(df_tc["Composite"].iloc[-1]),
                "Signal":    rec_map[int(df_tc["Trade"].iloc[-1])],
                "Buy&Hold%": (df_tc["CumBH"].iloc[-1]-1)*100,
                "Strat%":    (df_tc["CumStrat"].iloc[-1]-1)*100,
                "Sharpe":    sh,
                "MaxDD":     md,
                "WinRate":   wr
            })
        if perf:
            df_p = pd.DataFrame(perf).set_index("Ticker")
            st.dataframe(df_p, use_container_width=True)
            st.download_button("📥 Download CSV", df_p.to_csv(), "batch.csv")
        else:
            st.error("No valid data for batch tickers.")

    # ─────────── Midday Movers ───────────
    st.markdown("---\n## 🌤️ Midday Movers")
    mover_list = st.text_area("Tickers to monitor", "AAPL, MSFT, TSLA, SPY, QQQ").upper()
    if st.button("🔄 Get Midday Movers"):
        movers=[]
        for sym in [s.strip() for s in mover_list.split(",") if s.strip()]:
            td = yf.Ticker(sym).history(period="1d",interval="5m")
            if td.empty:
                td = yf.Ticker(sym).history(period="2d",interval="5m")
                td = td[td.index>=pd.Timestamp.utcnow().normalize()]
            if td.empty:
                st.warning(f"No intraday data for {sym}")
                continue
            o, c = td["Open"].iloc[0], td["Close"].iloc[-1]
            movers.append({"Ticker":sym,"Open":o,"Current":c,"Change%":(c-o)/o*100})
        if movers:
            df_m=pd.DataFrame(movers).set_index("Ticker")
            st.dataframe(df_m.sort_values("Change%",ascending=False),use_container_width=True)
        else:
            st.info("No valid intraday data.")

    # ─────────── Portfolio Simulator ───────────
    st.markdown("---\n## 📊 Portfolio Simulator")
    st.info("Enter your positions in CSV: ticker,shares,cost_basis")
    holdings = st.text_area("e.g.\nAAPL,10,150\nMSFT,5,300", height=100)
    if st.button("▶️ Simulate Portfolio"):
        rows=[r.strip().split(",") for r in holdings.splitlines() if r.strip()]
        data=[]
        for tkr,sh,cb in rows:
            tkr,sh,cb=tkr.upper(),float(sh),float(cb)
            hist=yf.Ticker(tkr).history(period="1d")
            if hist.empty: continue
            price=hist["Close"].iloc[-1]
            invested, value = sh*cb, sh*price
            pnl, pnl_pct = value-invested, (value-invested)/invested*100
            df_raw=load_and_compute(tkr,ma_window,rsi_period,macd_fast,macd_slow,macd_signal,asset_type)
            comp_sugg = "N/A" if df_raw.empty else rec_map[int(build_composite(df_raw,ma_window,rsi_period)["Trade"].iloc[-1])]
            suggestion = ("🔴 SELL" if pnl_pct>profit_target else
                          "🟢 BUY"  if pnl_pct<-loss_limit else comp_sugg)
            data.append({
                "Ticker":tkr,"Shares":sh,"CostBasis":cb,
                "Price":price,"MarketValue":value,"Invested":invested,
                "P/L":pnl,"P/L%":pnl_pct,
                "Composite":comp_sugg,"Suggestion":suggestion
            })
        if data:
            df_port=pd.DataFrame(data).set_index("Ticker")
            st.dataframe(df_port,use_container_width=True)
            st.metric("Total MV",f"${df_port['MarketValue'].sum():,.2f}")
            st.metric("Total Invested",f"${df_port['Invested'].sum():,.2f}")
            st.metric("Total P/L",f"${df_port['MarketValue'].sum()-df_port['Invested'].sum():,.2f}")
            fig,ax=plt.subplots()
            df_port["MarketValue"].plot.pie(autopct="%.1f%%",ax=ax)
            ax.set_ylabel(""); ax.set_title("Allocation")
            st.pyplot(fig)
        else:
            st.error("No valid holdings.")

    # ─────────── Hyperparameter Optimization ───────────
    st.markdown("---\n## 🛠️ Hyperparameter Optimization")
    ma_list  = st.sidebar.multiselect("MA windows",[5,10,15,20,30],default=[ma_window],key="grid_ma")
    rsi_list = st.sidebar.multiselect("RSI lookbacks",[7,14,21,28],default=[rsi_period],key="grid_rsi")
    mf_list  = st.sidebar.multiselect("MACD fast spans",[5,10,12,16],default=[macd_fast],key="grid_mf")
    ms_list  = st.sidebar.multiselect("MACD slow spans",[20,26,32,40],default=[macd_slow],key="grid_ms")
    sig_list = st.sidebar.multiselect("MACD sig spans",[5,9,12,16],default=[macd_signal],key="grid_sig")
    if st.button("🏃‍♂️ Run Grid Search"):
        if not ticker:
            st.error("Enter a ticker first."); st.stop()
        results=[]
        for mw in ma_list:
            for rp in rsi_list:
                for f in mf_list:
                    for s in ms_list:
                        for sg in sig_list:
                            df_i=load_and_compute(ticker,mw,rp,f,s,sg,asset_type)
                            if df_i.empty: continue
                            df_ci,_,shp,wr = backtest(build_composite(df_i,mw,rp))
                            results.append({
                                "MA":mw,"RSI":rp,"MACD Fast":f,"MACD Slow":s,"MACD Sig":sg,
                                "Strat%":(df_ci["CumStrat"].iloc[-1]-1)*100,
                                "Sharpe":shp,"WinRate":wr
                            })
        if results:
            df_g=pd.DataFrame(results).sort_values("Strat%",ascending=False).head(10)
            st.dataframe(df_g,use_container_width=True)
            st.download_button("Download CSV",df_g.to_csv(), "grid.csv")
        else:
            st.error("No valid combos.")

    # ─────────── Watchlist Summary ───────────
    st.markdown("---\n## ⏰ Watchlist Summary")
    watch = st.text_area("Enter tickers","AAPL, MSFT, TSLA, SPY, QQQ").upper()
    if st.button("📬 Generate Watchlist Summary"):
        table=[]
        for t in [x.strip() for x in watch.split(",") if x.strip()]:
            df_t=load_and_compute(t,ma_window,rsi_period,macd_fast,macd_slow,macd_signal,asset_type)
            if df_t.empty:
                table.append({"Ticker":t,"Composite":None,"Signal":"N/A"})
                continue
            df_w,_,_,_ = backtest(build_composite(df_t,ma_window,rsi_period))
            comp=int(df_w["Composite"].iloc[-1])
            sig = rec_map[int(df_w["Trade"].iloc[-1])]
            table.append({"Ticker":t,"Composite":comp,"Signal":sig})
        df_wl=pd.DataFrame(table).set_index("Ticker")
        st.dataframe(df_wl,use_container_width=True)
        for t in df_wl.index:
            df_t=load_and_compute(t,ma_window,rsi_period,macd_fast,macd_slow,macd_signal,asset_type)
            if df_t.empty: continue
            df_c=build_composite(df_t,ma_window,rsi_period)
            last=df_c.iloc[-1]
            ma_s,rsi_s,macd_s = int(last["MA_Signal"]),int(last["RSI_Signal"]),int(last["MACD_Signal2"])
            rsi_txt = ("N/A" if np.isnan(last[f"RSI{rsi_period}"]) else
                       f"RSI={last[f'RSI{rsi_period}']:.1f}")
            macd_txt = ("↑" if macd_s>0 else "↓" if macd_s<0 else "–")
            with st.expander(f"🔎 {t} Reasoning ({df_wl.loc[t,'Signal']})"):
                st.write(f"- MA signal: {ma_s}  - RSI: {rsi_txt}  - MACD: {macd_txt}")
                st.write(f"- Composite Score: {comp}")
