import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import feedparser  # ← added for RSS fallback
import ccxt       # ← added for crypto data

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
… (unchanged help text) …
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
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ─────────── Sidebar Controls ───────────
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

    # ─────────── Asset Type Selector (NEW) ───────────
    asset_type = st.sidebar.selectbox(
        "Asset Type",
        ["Stock", "Crypto"],
        help="Use 'Crypto' for symbols like BTC/USDT, ETH/USDT"
    )

    # ─────────── Profit/Loss Sliders for Portfolio ───────────
    profit_target = st.sidebar.slider(
        "Profit target (%)", min_value=1, max_value=100, value=5,
        help="If unrealized P/L% exceeds this → SELL"
    )
    loss_limit = st.sidebar.slider(
        "Loss limit (%)", min_value=1, max_value=100, value=5,
        help="If unrealized P/L% falls below –this → BUY"
    )

    st.title("🚀 QuantaraX — Composite Signal Engine")
    st.write("MA + RSI + MACD Composite Signals & Backtest")

    # ─────────── Data Loading ───────────
    @st.cache_data(show_spinner=False)
    def load_and_compute(ticker, ma_w, rsi_p, mf, ms, sig, asset_type):
        # ── Crypto branch ──
        if asset_type == "Crypto":
            ex = ccxt.binance()
            six_months_ago = pd.Timestamp.utcnow() - pd.DateOffset(months=6)
            since = ex.parse8601(six_months_ago.isoformat())
            bars = ex.fetch_ohlcv(ticker, timeframe='1d', since=since)
            df = pd.DataFrame(bars, columns=["ts","Open","High","Low","Close","Volume"])
            df["Date"] = pd.to_datetime(df["ts"], unit="ms")
            df.set_index("Date", inplace=True)
        else:
            df = yf.download(ticker, period="6mo", progress=False)

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
        cols = [ma_col, rsi_col, "MACD", "MACD_Signal"]
        prs  = [c for c in cols if c in df.columns]
        if prs:
            try: df = df.dropna(subset=prs).reset_index(drop=True)
            except KeyError: pass

        return df

    # ─────────── Composite Signals ───────────
    def build_composite(df, ma_w, rsi_p):
        …  # unchanged
        return df

    # ─────────── Backtest ───────────
    def backtest(df):
        …  # unchanged
        return df, max_dd, sharpe, win_rt

    # ─────────────────── Single‐Ticker Backtest ───────────────────
    st.markdown("## Single‐Ticker Backtest")
    ticker = st.text_input("Ticker (e.g. AAPL or BTC/USDT)", "AAPL").upper()

    if ticker:
        # ── Price fetch branch ──
        if asset_type == "Crypto":
            ex    = ccxt.binance()
            tick  = ex.fetch_ticker(ticker)
            price = tick.get("last", None)
        else:
            info  = yf.Ticker(ticker).info
            price = info.get("regularMarketPrice")

        if price is not None:
            st.subheader(f"💲 Live Price: ${price:.2f}")

        # ─── News feed (unchanged) ───
        raw_news = getattr(yf.Ticker(ticker), "news", []) or []
        shown = 0
        …  # unchanged dual‐source news block

    if st.button("▶️ Run Composite Backtest"):
        df_raw = load_and_compute(
            ticker,
            ma_window,
            rsi_period,
            macd_fast,
            macd_slow,
            macd_signal,
            asset_type,         # ← pass asset type
        )
        …  # remaining backtest & plotting unchanged

    # ─────────── Batch Backtest ───────────
    st.markdown("---")
    st.markdown("## Batch Backtest")
    batch = st.text_area("Tickers (comma-separated)", "AAPL, MSFT, TSLA, SPY, QQQ").upper()
    if st.button("▶️ Run Batch Backtest"):
        perf = []
        for t in [x.strip() for x in batch.split(",") if x.strip()]:
            df_t = load_and_compute(
                t,
                ma_window,
                rsi_period,
                macd_fast,
                macd_slow,
                macd_signal,
                asset_type,     # ← pass asset type
            )

    # ─────────── Midday Movers ───────────
    st.markdown("---")
    st.markdown("## 🌤️ Midday Movers (Intraday % Change)")

    mover_list = st.text_area(
        "Tickers to monitor (comma-separated)",
        "AAPL, MSFT, TSLA, SPY, QQQ"
    ).upper()

    if st.button("🔄 Get Midday Movers"):
        movers = []
        for sym in [s.strip() for s in mover_list.split(",") if s.strip()]:
            tk = yf.Ticker(sym)

            # Try 1-day intraday first
            intraday = tk.history(period="1d", interval="5m")

            # Fallback: grab 2 days and then filter to today's date
            if intraday.empty:
                intraday = tk.history(period="2d", interval="5m")
                if not intraday.empty:
                    today = pd.Timestamp.utcnow().normalize()
                    intraday = intraday[intraday.index >= today]

            if intraday.empty:
                st.warning(f"No intraday data for {sym}. (Market closed or too early!)")
                continue

            open_price = intraday["Open"].iloc[0]
            last_price = intraday["Close"].iloc[-1]
            change_pct = (last_price - open_price) / open_price * 100

            movers.append({
                "Ticker":   sym,
                "Open":     open_price,
                "Current":  last_price,
                "Change %": change_pct
            })

        if movers:
            df_m = pd.DataFrame(movers)
            df_m["Change %"] = pd.to_numeric(df_m["Change %"], errors="coerce")
            df_m = df_m.dropna(subset=["Change %"])
            df_m = df_m.set_index("Ticker").sort_values("Change %", ascending=False)
            st.dataframe(df_m, use_container_width=True)
        else:
            st.info("No valid intraday data found for those tickers.")


    # ─────────── Portfolio Simulator ───────────
    st.markdown("---")
    st.markdown("## 📊 Portfolio Simulator")
    st.info("Enter your positions in CSV: ticker,shares,cost_basis")
    holdings = st.text_area("e.g.\nAAPL,10,150\nMSFT,5,300", height=100)
    if st.button("▶️ Simulate Portfolio"):
        rows = [r.strip().split(",") for r in holdings.splitlines() if r.strip()]
        data=[]
        for ticker_, shares, cost in rows:
            tkr = ticker_.upper().strip()
            try:
                s=float(shares); c=float(cost)
            except: continue
            hist = yf.Ticker(tkr).history(period="1d")
            if hist.empty: continue
            price=hist["Close"].iloc[-1]
            invested=s*c; value=s*price; pnl=value-invested
            pnl_pct=(pnl/invested*100) if invested else np.nan
            df_raw=load_and_compute(tkr,ma_window,rsi_period,macd_fast,macd_slow,macd_signal)
            if df_raw.empty: comp_sugg="N/A"
            else:
                df_c=build_composite(df_raw,ma_window,rsi_period)
                comp_sugg=rec_map.get(int(df_c["Trade"].iloc[-1]),"🟡 HOLD")
            if pnl_pct > profit_target:
                suggestion="🔴 SELL"
            elif pnl_pct < -loss_limit:
                suggestion="🟢 BUY"
            else:
                suggestion=comp_sugg if comp_sugg in rec_map.values() else "🟡 HOLD"
            data.append({
                "Ticker":tkr,"Shares":s,"Cost Basis":c,"Price":price,
                "Market Value":value,"Invested":invested,"P/L":pnl,
                "P/L %":pnl_pct,"Composite Sig":comp_sugg,"Suggestion":suggestion
            })
        if data:
            df_port=pd.DataFrame(data).set_index("Ticker")
            st.dataframe(df_port, use_container_width=True)
            st.metric("Total Market Value", f"${df_port['Market Value'].sum():,.2f}")
            st.metric("Total Invested",     f"${df_port['Invested'].sum():,.2f}")
            st.metric("Total P/L",          f"${df_port['Market Value'].sum()-df_port['Invested'].sum():,.2f}")
            fig, ax=plt.subplots()
            df_port["Market Value"].plot.pie(autopct="%.1f%%", ax=ax)
            ax.set_ylabel(""); ax.set_title("Portfolio Allocation")
            st.pyplot(fig)
        else:
            st.error("No valid holdings provided.")

    # ─────────── Hyperparameter Optimization ───────────
    st.markdown("---")
    st.markdown("## 🛠️ Hyperparameter Optimization")
    ma_list  = st.sidebar.multiselect("MA windows",     [5,10,15,20,30], default=[ma_window], key="grid_ma")
    rsi_list = st.sidebar.multiselect("RSI lookbacks",  [7,14,21,28],   default=[rsi_period], key="grid_rsi")
    mf_list  = st.sidebar.multiselect("MACD fast spans",[8,12,16,20], default=[macd_fast], key="grid_mf")
    ms_list  = st.sidebar.multiselect("MACD slow spans",[20,26,32,40], default=[macd_slow], key="grid_ms")
    sig_list = st.sidebar.multiselect("MACD sig spans", [5,9,12,16],   default=[macd_signal], key="grid_sig")
    if st.button("🏃‍♂️ Run Grid Search"):
        if not ticker:
            st.error("Enter a ticker first."); st.stop()
        df_full=load_and_compute(ticker,ma_window,rsi_period,macd_fast,macd_slow,macd_signal)
        if df_full.empty:
            st.error(f"No data for '{ticker}'"); st.stop()
        results=[]
        with st.spinner("Testing parameter combos…"):
            for mw in ma_list:
                for rp in rsi_list:
                    for mf_ in mf_list:
                        for ms_ in ms_list:
                            for s_ in sig_list:
                                df_i=load_and_compute(ticker,mw,rp,mf_,ms_,s_)
                                if df_i.empty: continue
                                df_ci, md_i, sh_i, wr_i=backtest(build_composite(df_i,mw,rp))
                                results.append({
                                    "MA":mw,"RSI":rp,
                                    "MACD Fast":mf_,"MACD Slow":ms_,"MACD Sig":s_,
                                    "Strategy %":(df_ci["CumStrat"].iloc[-1]-1)*100,
                                    "Sharpe":sh_i,"Max Drawdown":md_i,"Win Rate":wr_i
                                })
        if results:
            df_grid=pd.DataFrame(results).sort_values("Strategy %",ascending=False).head(10)
            st.dataframe(df_grid, use_container_width=True)
            st.download_button("Download CSV", df_grid.to_csv(index=False), "grid.csv")
        else:
            st.error("No valid parameter combinations found.")

    # ─────────── Watchlist Summary ───────────
    st.markdown("---")
    st.markdown("## ⏰ Watchlist Summary")
    watch = st.text_area("Enter tickers", "AAPL, MSFT, TSLA, SPY, QQQ").upper()
    if st.button("📬 Generate Watchlist Summary"):
        table=[]
        for t in [x.strip() for x in watch.split(",") if x.strip()]:
            df_t=load_and_compute(t,ma_window,rsi_period,macd_fast,macd_slow,macd_signal)
            if df_t.empty:
                table.append({"Ticker":t,"Composite":None,"Signal":"N/A"})
                continue
            df_w,_,_,_=backtest(build_composite(df_t,ma_window,rsi_period))
            comp=int(df_w["Composite"].iloc[-1])
            sig=rec_map[int(df_w["Trade"].iloc[-1])]
            table.append({"Ticker":t,"Composite":comp,"Signal":sig})
        df_watch=pd.DataFrame(table).set_index("Ticker")
        st.dataframe(df_watch, use_container_width=True)
        for t in df_watch.index:
            df_t=load_and_compute(t,ma_window,rsi_period,macd_fast,macd_slow,macd_signal)
            if df_t.empty: continue
            df_c=build_composite(df_t,ma_window,rsi_period)
            last=df_c.iloc[-1]
            ma_s=int(last["MA_Signal"]); rsi_s=int(last["RSI_Signal"]); macd_s=int(last["MACD_Signal2"])
            try:
                rsi_v=float(last[f"RSI{rsi_period}"]); valid=True
            except:
                valid=False
            ma_txt={1:f"Price ↑ above {ma_window}-day MA.",0:"No crossover.",-1:f"Price ↓ below {ma_window}-day MA."}[ma_s]
            if valid:
                rsi_txt={1:f"RSI ({rsi_v:.1f}) < 30 → oversold.",0:f"RSI ({rsi_v:.1f}) neutral.",-1:f"RSI ({rsi_v:.1f}) > 70 → overbought."}[rsi_s]
            else:
                rsi_txt="RSI data unavailable."
            macd_txt={1:"MACD line crossed **above** its signal line.",0:"No crossover.",-1:"MACD line crossed **below** its signal line."}[macd_s]
            with st.expander(f"🔎 {t} Reasoning ({df_watch.loc[t,'Signal']})"):
                st.write(f"- **MA:**  {ma_txt}")
                st.write(f"- **RSI:** {rsi_txt}")
                st.write(f"- **MACD:** {macd_txt}")
                st.write(f"- **Composite Score:** {df_watch.loc[t,'Composite']}")
