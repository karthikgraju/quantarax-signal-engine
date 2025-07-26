import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import feedparser  # ← added for RSS fallback

# ───────────────────────────── Page Setup ─────────────────────────────
st.set_page_config(page_title="QuantarX Composite Signals BETA v2", layout="centered")
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
    st.header("How QuantarX Works")
    st.markdown(r"""
... (Help content omitted for brevity) ...
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

    # ─────────── Asset Type Selector ───────────
    asset_type = st.sidebar.selectbox(
        "Asset Type",
        ["Stock", "Crypto"],
        help="Use 'Crypto' for symbols like BTC/USDT, ETH/USDT"
    )

    # ─────────── Profit/Loss Sliders ───────────
    profit_target = st.sidebar.slider(
        "Profit target (%)", min_value=1, max_value=100, value=5,
        help="If unrealized P/L% exceeds this → SELL"
    )
    loss_limit = st.sidebar.slider(
        "Loss limit (%)", min_value=1, max_value=100, value=5,
        help="If unrealized P/L% falls below –this → BUY"
    )

    # ... (Other engine sections: Data Loading, Single‐Ticker, Batch, Midday Movers) ...

    # ─────────── Portfolio Simulator ───────────
    st.markdown("---")
    st.markdown("## 📊 Portfolio Simulator")
    st.info("Enter your positions in CSV: ticker,shares,cost_basis")
    holdings = st.text_area("e.g.\nAAPL,10,150\nMSFT,5,300", height=100)

    if st.button("▶️ Simulate Portfolio"):
        rows = [r.strip().split(",") for r in holdings.splitlines() if r.strip()]
        data = []
        for idx, row in enumerate(rows):
            if len(row) != 3:
                st.warning(f"Skipping invalid row {idx+1}: {row}")
                continue
            ticker_, shares, cost = row
            tkr = ticker_.upper().strip()
            # Validate numeric inputs
            try:
                s = float(shares)
                c = float(cost)
            except ValueError:
                st.warning(f"Invalid numbers on row {idx+1}: {row}")
                continue
            # Fetch latest price
            hist = yf.Ticker(tkr).history(period="1d")
            if hist.empty or "Close" not in hist:
                st.warning(f"No price data for {tkr}")
                continue
            price = hist["Close"].iloc[-1]
            invested = s * c
            if invested == 0:
                st.warning(f"Invested amount is zero for {tkr}, skipping")
                continue
            value = s * price
            pnl = value - invested
            pnl_pct = (pnl / invested * 100)

            # Compute composite signal
            try:
                df_raw = load_and_compute(tkr, ma_window, rsi_period, macd_fast, macd_slow, macd_signal, asset_type)
                if df_raw.empty:
                    comp_sugg = "N/A"
                else:
                    df_c = build_composite(df_raw, ma_window, rsi_period)
                    comp_sugg = rec_map.get(int(df_c["Trade"].iloc[-1]), "🟡 HOLD")
            except Exception as e:
                st.error(f"Error computing signal for {tkr}: {e}")
                comp_sugg = "N/A"

            # Determine suggestion
            if pnl_pct > profit_target:
                suggestion = "🔴 SELL"
            elif pnl_pct < -loss_limit:
                suggestion = "🟢 BUY"
            else:
                suggestion = comp_sugg if comp_sugg in rec_map.values() else "🟡 HOLD"

            data.append({
                "Ticker": tkr,
                "Shares": s,
                "Cost Basis": c,
                "Price": price,
                "Market Value": value,
                "Invested": invested,
                "P/L": pnl,
                "P/L %": pnl_pct,
                "Composite Sig": comp_sugg,
                "Suggestion": suggestion
            })

        if data:
            df_port = pd.DataFrame(data).set_index("Ticker")
            st.dataframe(df_port, use_container_width=True)
            st.metric("Total Market Value", f"${df_port['Market Value'].sum():,.2f}")
            st.metric("Total Invested",     f"${df_port['Invested'].sum():,.2f}")
            st.metric("Total P/L",          f"${(df_port['Market Value'].sum() - df_port['Invested'].sum()):,.2f}")

            # Pie chart for allocation
            fig, ax = plt.subplots()
            df_port['Market Value'].plot.pie(autopct='%.1f%%', ax=ax)
            ax.set_ylabel('')
            ax.set_title('Portfolio Allocation')
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
