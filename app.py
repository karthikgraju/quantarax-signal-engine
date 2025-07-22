import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ───────────────────────────── Page Setup ─────────────────────────────
st.set_page_config(page_title="QuantaraX Composite Signals", layout="wide")
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
    # ───────────────────── Defaults & Session State ─────────────────────
    DEFAULTS = dict(ma_window=10, rsi_period=14, macd_fast=12, macd_slow=26, macd_signal=9)
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ───────────────────── Sidebar Controls ────────────────────────────
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

    # ─────────── Data Loading & Indicator Computation ────────────────
    @st.cache_data(show_spinner=False)
    def load_and_compute(ticker, ma_w, rsi_p, mf, ms, sig):
        df = yf.download(ticker, period="6mo", progress=False)
        if df.empty or "Close" not in df:
            return pd.DataFrame()
        # MA
        ma_col = f"MA{ma_w}"
        df[ma_col] = df["Close"].rolling(ma_w).mean()
        # RSI
        delta = df["Close"].diff().fillna(0)
        up    = delta.clip(lower=0)
        down  = -delta.clip(upper=0)
        ema_up   = up.ewm(com=rsi_p-1, adjust=False).mean()
        ema_down = down.ewm(com=rsi_p-1, adjust=False).mean()
        rsi_col  = f"RSI{rsi_p}"
        df[rsi_col] = 100 - 100/(1 + ema_up/ema_down)
        # MACD
        ema_f    = df["Close"].ewm(span=mf, adjust=False).mean()
        ema_s    = df["Close"].ewm(span=ms, adjust=False).mean()
        macd     = ema_f - ema_s
        macd_sig = macd.ewm(span=sig, adjust=False).mean()
        df["MACD"]        = macd
        df["MACD_Signal"] = macd_sig
        # Drop NA
        to_drop = [c for c in [ma_col, rsi_col, "MACD", "MACD_Signal"] if c in df]
        if to_drop:
            df = df.dropna(subset=to_drop).reset_index(drop=True)
        return df

    def build_composite(df, ma_w, rsi_p):
        n = len(df)
        close = df["Close"].to_numpy()
        ma    = df[f"MA{ma_w}"].to_numpy()
        rsi   = df[f"RSI{rsi_p}"].to_numpy()
        macd  = df["MACD"].to_numpy()
        sig   = df["MACD_Signal"].to_numpy()

        ma_sig    = np.zeros(n, int)
        rsi_sig   = np.zeros(n, int)
        macd_sig2 = np.zeros(n, int)
        comp      = np.zeros(n, int)
        trade     = np.zeros(n, int)

        for i in range(1, n):
            # MA crossover
            if close[i-1] < ma[i-1] and close[i] > ma[i]:
                ma_sig[i] = 1
            elif close[i-1] > ma[i-1] and close[i] < ma[i]:
                ma_sig[i] = -1
            # RSI thresholds
            if rsi[i] < 30:
                rsi_sig[i] = 1
            elif rsi[i] > 70:
                rsi_sig[i] = -1
            # MACD crossover
            if macd[i-1] < sig[i-1] and macd[i] > sig[i]:
                macd_sig2[i] = 1
            elif macd[i-1] > sig[i-1] and macd[i] < sig[i]:
                macd_sig2[i] = -1

            comp[i]  = ma_sig[i] + rsi_sig[i] + macd_sig2[i]
            trade[i] = np.sign(comp[i])

        df["MA_Signal"]    = ma_sig
        df["RSI_Signal"]   = rsi_sig
        df["MACD_Signal2"] = macd_sig2
        df["Composite"]    = comp
        df["Trade"]        = trade
        return df

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

    # ─────────────────── Single‐Ticker + Media Mood ────────────────────
    ticker = st.text_input("Ticker (e.g. AAPL)", "AAPL").upper()
    if ticker:
        # live price
        live = yf.Ticker(ticker).info.get("regularMarketPrice")
        if live is not None:
            st.subheader(f"💲 Live Price: ${live:.2f}")

        # raw news + sentiment
        news_items = yf.Ticker(ticker).news or []
        records = []
        for art in news_items:
            ts = art.get("providerPublishTime")
            if ts:
                date = datetime.fromtimestamp(ts).date()
                text = art.get("summary", art.get("title",""))
                score = analyzer.polarity_scores(text)["compound"]
                records.append({"date": date, "sentiment": score})

        # two-col layout
        col_price, col_sent = st.columns([2, 1])
        with col_price:
            # historic candlestick + MA20
            hist = yf.Ticker(ticker).history(period="3mo")
            fig_price = go.Figure([
                go.Candlestick(
                    x=hist.index, open=hist["Open"], high=hist["High"],
                    low=hist["Low"], close=hist["Close"], name="Price"
                ),
                go.Scatter(
                    x=hist.index,
                    y=hist["Close"].rolling(ma_window).mean(),
                    name=f"MA{ma_window}"
                )
            ])
            fig_price.update_layout(
                title=f"{ticker} Price & MA{ma_window}",
                xaxis_rangeslider_visible=False
            )
            st.plotly_chart(fig_price, use_container_width=True)

            # raw headlines expander
            with st.expander("📰 Recent News & Sentiment"):
                if not news_items:
                    st.info("No recent news found.")
                else:
                    for art in news_items[:5]:
                        title = art.get("title","")[:80]
                        link  = art.get("link","")
                        text  = art.get("summary", title)
                        score = analyzer.polarity_scores(text)["compound"]
                        emoji = "🔺" if score>0.1 else ("🔻" if score< -0.1 else "➖")
                        st.markdown(f"- [{title}]({link}) {emoji}  \n_{text[:100]}…_")

        with col_sent:
            if records:
                df_sent = pd.DataFrame(records)
                idx = pd.date_range(end=datetime.today().date(), periods=30)
                df_daily = df_sent.groupby("date").sentiment.mean().reindex(idx, fill_value=0)
                df_mood  = df_daily.rolling(7, min_periods=1).mean()
                fig_sent = go.Figure(
                    go.Scatter(x=df_mood.index, y=df_mood.values, mode="lines+markers")
                )
                fig_sent.update_layout(
                    title="Media Mood (7-day avg)", yaxis_title="Sentiment"
                )
                st.plotly_chart(fig_sent, use_container_width=True)
            else:
                st.info("No sentiment data to plot.")

    # run the composite backtest
    if st.button("▶️ Run Composite Backtest"):
        df0 = load_and_compute(ticker, ma_window, rsi_period, macd_fast, macd_slow, macd_signal)
        if df0.empty:
            st.error(f"No data for '{ticker}'."); st.stop()
        dfc, max_dd, sharpe, win_rt = backtest(build_composite(df0, ma_window, rsi_period))

        rec = {1:"🟢 BUY", 0:"🟡 HOLD", -1:"🔴 SELL"}[int(dfc["Trade"].iloc[-1])]
        st.success(f"**{ticker}**: {rec}")

        # details
        ms, rs, cs = int(dfc["MA_Signal"].iloc[-1]), int(dfc["RSI_Signal"].iloc[-1]), int(dfc["MACD_Signal2"].iloc[-1])
        rsi_val = dfc[f"RSI{rsi_period}"].iloc[-1]
        ma_txt = {1:f"Price crossed above MA{ma_window}",0:"No crossover", -1:f"Price crossed below MA{ma_window}"}
        rsi_txt = {1:f"RSI {rsi_val:.1f}<30 oversold",0:f"RSI {rsi_val:.1f} neutral", -1:f"RSI {rsi_val:.1f}>70 overbought"}
        macd_txt= {1:"MACD ↑ signal",0:"No MACD crossover", -1:"MACD ↓ signal"}

        with st.expander("🔎 Why This Signal?"):
            st.write(f"- **MA:**   {ma_txt[ms]}")
            st.write(f"- **RSI:**  {rsi_txt[rs]}")
            st.write(f"- **MACD:** {macd_txt[cs]}")
            st.write(f"- **Composite Score:** {int(dfc['Composite'].iloc[-1])}")

        st.markdown(f"""
- **Buy & Hold:**    {(dfc['CumBH'].iloc[-1]-1)*100:.2f}%
- **Strategy:**      {(dfc['CumStrat'].iloc[-1]-1)*100:.2f}%
- **Sharpe:**        {sharpe:.2f}
- **Max Drawdown:**  {max_dd:.2f}%
- **Win Rate:**      {win_rt:.1f}%
""")

        # fallback Matplotlib equity chart
        fig, axs = plt.subplots(2,1, figsize=(10,6), sharex=True)
        axs[0].plot(dfc["CumBH"], label="Buy & Hold")
        axs[0].plot(dfc["CumStrat"], label="Strategy")
        axs[0].legend(); axs[0].set_title("Equity Curves")
        axs[1].bar(dfc.index, dfc["Composite"])
        axs[1].set_title("Composite Vote")
        plt.xticks(rotation=45); plt.tight_layout()
        st.pyplot(fig)

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
            df_c2, md2, sh2, wr2 = backtest(build_composite(df_t, ma_window, rsi_period))
            perf.append({
                "Ticker":    t,
                "Composite": int(df_c2["Composite"].iloc[-1]),
                "Signal":    {1:"BUY",0:"HOLD",-1:"SELL"}[int(df_c2["Trade"].iloc[-1])],
                "BH %":      (df_c2["CumBH"].iloc[-1]-1)*100,
                "Strat %":   (df_c2["CumStrat"].iloc[-1]-1)*100,
                "Sharpe":    sh2,
                "MaxDD %":   md2,
                "Win %":     wr2
            })
        if not perf:
            st.error("No valid data.")
        else:
            df_perf = pd.DataFrame(perf).set_index("Ticker")
            st.dataframe(df_perf, use_container_width=True)
            st.download_button("Download CSV", df_perf.to_csv(), "batch.csv")

    # ───────────────────────────── Grid Search ─────────────────────────────
    st.markdown("---")
    st.markdown("## 🛠️ Hyperparameter Optimization")
    ma_list  = st.sidebar.multiselect("MA windows",     [5,10,15,20,30], default=[ma_window], key="grid_ma")
    rsi_list = st.sidebar.multiselect("RSI lookbacks",  [7,14,21,28],   default=[rsi_period], key="grid_rsi")
    mf_list  = st.sidebar.multiselect("MACD fast spans",[8,12,16,20],  default=[macd_fast],   key="grid_mf")
    ms_list  = st.sidebar.multiselect("MACD slow spans",[20,26,32,40],default=[macd_slow],  key="grid_ms")
    sig_list = st.sidebar.multiselect("MACD sig spans",[5,9,12,16],   default=[macd_signal],key="grid_sig")

    if st.button("🏃‍♂️ Run Grid Search"):
        if not ticker:
            st.error("Enter a ticker."); st.stop()
        df_full = load_and_compute(ticker, ma_window, rsi_period, macd_fast, macd_slow, macd_signal)
        if df_full.empty:
            st.error(f"No data for '{ticker}'"); st.stop()
        results = []
        with st.spinner("Testing combos…"):
            for mw in ma_list:
                for rp in rsi_list:
                    for mf in mf_list:
                        for ms in ms_list:
                            for s in sig_list:
                                df_i = load_and_compute(ticker, mw, rp, mf, ms, s)
                                if df_i.empty:
                                    continue
                                df_ci, md_i, sh_i, wr_i = backtest(build_composite(df_i, mw, rp))
                                strat = (df_ci["CumStrat"].iloc[-1]-1)*100
                                results.append({
                                    "MA": mw, "RSI": rp, "MACD_Fast": mf,
                                    "MACD_Slow": ms, "MACD_Sig": s,
                                    "Strategy %": strat, "Sharpe": sh_i,
                                    "MaxDD %": md_i, "Win %": wr_i
                                })
        if not results:
            st.error("No valid combos.")
        else:
            df_grid = pd.DataFrame(results).sort_values("Strategy %", ascending=False).head(10)
            st.dataframe(df_grid, use_container_width=True)
            st.download_button("Download full CSV", df_grid.to_csv(index=False), "grid.csv")

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
            df_w, _, _, _ = backtest(build_composite(df_t, ma_window, rsi_period))
            comp = int(df_w["Composite"].iloc[-1])
            sig  = {1:"BUY",0:"HOLD",-1:"SELL"}[int(df_w["Trade"].iloc[-1])]
            tbl.append({"Ticker": t, "Composite": comp, "Signal": sig})

        df_watch = pd.DataFrame(tbl).set_index("Ticker")
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
            rsi_v  = last[f"RSI{rsi_period}"]

            ma_txt = {1:f"Price ↑ above MA{ma_window}",0:"No crossover", -1:f"Price ↓ below MA{ma_window}"}[ma_s]
            rsi_txt= {1:f"RSI ({rsi_v:.1f}) <30 oversold",0:f"RSI ({rsi_v:.1f}) neutral", -1:f"RSI ({rsi_v:.1f}) >70 overbought"}[rsi_s]
            macd_txt = {1:"MACD ↑ signal",0:"No crossover", -1:"MACD ↓ signal"}[macd_s]

            with st.expander(f"🔎 {t} Reasoning ({df_watch.loc[t,'Signal']})"):
                st.write(f"- **MA:**  {ma_txt}")
                st.write(f"- **RSI:** {rsi_txt}")
                st.write(f"- **MACD:** {macd_txt}")
                st.write(f"- **Composite Score:** {df_watch.loc[t,'Composite']}")

