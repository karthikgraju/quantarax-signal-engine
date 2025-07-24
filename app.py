import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import feedparser

# ───────────────────────────── Page Setup ─────────────────────────────
st.set_page_config(page_title="QuantaraX Composite Signals BETA v2", layout="centered")
analyzer = SentimentIntensityAnalyzer()

# ───────────────────────────── Mappings ─────────────────────────────
rec_map = {
    1:  "🟢 BUY",
    0:  "🟡 HOLD",
   -1:  "🔴 SELL",
}

# ───────────────────────────── Tabs ─────────────────────────────
tab_engine, tab_help = st.tabs(["🚀 Engine", "❓ How It Works"])

# ───────────────────────────── Help Tab ─────────────────────────────
with tab_help:
    st.header("How QuantaraX Works")
    st.markdown(r"""
Welcome to **QuantaraX**, the MVP from a hands-on team of quants, data scientists, and former traders on a mission to **democratize** institutional-grade quantitative tools for **every** investor.

---

🎯 **Our Purpose & Mission**  
QuantaraX exists to:
- **Demystify** technical analysis by **combining** multiple indicators into one clear, composite vote.
- **Reduce emotional bias** with consistent, rules-based signals.
- **Empower** users through education—see the “why” behind BUY/HOLD/SELL.
- **Accelerate** decisions with live prices, sentiment-weighted news, and portfolio simulations.
- **Scale** from MVP → full platform: real-time alerts, multi-asset, broker connectivity.

---

🔧 **Choosing Slider Settings**  
Every slider trades **responsiveness** vs. **smoothness**:

| Slider            | Function                                | Guidance                                             |
|-------------------|-----------------------------------------|------------------------------------------------------|
| **MA window**     | Days for moving average                 | Lower → more signals/noise; Higher → smoother/fewer  |
| **RSI lookback**  | Period for RSI’s EMA                    | Short → choppy; Long → stable                        |
| **MACD fast span**| EMA span for MACD fast line             | Lower → quick; Higher → slow                         |
| **MACD slow span**| EMA span for MACD slow line             | Keep ≥10 days above fast span                        |
| **MACD sig span** | EMA span for MACD signal line           | Lower → quick cross; Higher → fewer whipsaws         |
| **Profit target** | P/L% override → SELL                    | Set personal upside (5–20%)                          |
| **Loss limit**    | P/L% override → BUY                     | Set personal stop (3–10%)                            |

> **Tip:** start with defaults (10,14,12/26/9), tweak one at a time, and watch backtest metrics.

---

🏆 **Objectives**  
1. Deliver MVP for demos this week.  
2. Onboard 100+ beta users in 30 days.  
3. Integrate real-time streaming & push alerts (Q3).  
4. Expand to crypto, forex, alt-data (Q4).  
5. Build community features—strategy sharing, crowd sentiment.
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
    ma_window   = st.sidebar.slider("MA window",       5, 50, st.session_state["ma_window"],   key="ma_window")
    rsi_period  = st.sidebar.slider("RSI lookback",    5, 30, st.session_state["rsi_period"],  key="rsi_period")
    macd_fast   = st.sidebar.slider("MACD fast span",  5, 20, st.session_state["macd_fast"],   key="macd_fast")
    macd_slow   = st.sidebar.slider("MACD slow span", 20, 40, st.session_state["macd_slow"],   key="macd_slow")
    macd_signal = st.sidebar.slider("MACD sig span",    5, 20, st.session_state["macd_signal"], key="macd_signal")

    # ─── Profit/Loss Overrides (Portfolio) ───
    profit_target = st.sidebar.slider("Profit target (%)", 1, 100, 5,
                                      help="If unrealized P/L% > this → SELL")
    loss_limit    = st.sidebar.slider("Loss limit (%)",    1, 100, 5,
                                      help="If unrealized P/L% < –this → BUY")

    st.title("🚀 QuantaraX — Composite Signal Engine")
    st.write("MA + RSI + MACD Composite Signals & Backtest")

    # ─────────── Data Loader ───────────
    @st.cache_data(show_spinner=False)
    def load_and_compute(ticker, ma_w, rsi_p, mf, ms, sig):
        df = yf.download(ticker, period="6mo", progress=False)
        if df.empty or "Close" not in df:
            return pd.DataFrame()

        # MA
        df[f"MA{ma_w}"] = df["Close"].rolling(ma_w).mean()

        # RSI
        d       = df["Close"].diff()
        up, dn  = d.clip(lower=0), -d.clip(upper=0)
        ema_up  = up.ewm(com=rsi_p-1, adjust=False).mean()
        ema_dn  = dn.ewm(com=rsi_p-1, adjust=False).mean()
        df[f"RSI{rsi_p}"] = 100 - 100/(1 + ema_up/ema_dn)

        # MACD
        ema_f     = df["Close"].ewm(span=mf, adjust=False).mean()
        ema_s     = df["Close"].ewm(span=ms, adjust=False).mean()
        macd_line = ema_f - ema_s
        df["MACD"]        = macd_line
        df["MACD_Signal"] = macd_line.ewm(span=sig, adjust=False).mean()

        # Only drop rows where *all* of our indicator columns exist but have NaNs
        cols = [f"MA{ma_w}", f"RSI{rsi_p}", "MACD", "MACD_Signal"]
        valid = [c for c in cols if c in df.columns]
        if valid:
            df = df.dropna(subset=valid).reset_index(drop=True)

        return df

    # ─────────── Composite Signals ───────────
    def build_composite(df, ma_w, rsi_p):
        n = len(df)
        c  = df["Close"].to_numpy()
        ma = df[f"MA{ma_w}"].to_numpy()
        rsi = df[f"RSI{rsi_p}"].to_numpy()
        m, sig = df["MACD"].to_numpy(), df["MACD_Signal"].to_numpy()

        ma_s   = np.zeros(n, int)
        rsi_s  = np.zeros(n, int)
        macd_s = np.zeros(n, int)
        comp   = np.zeros(n, int)
        trade  = np.zeros(n, int)

        for i in range(1, n):
            if c[i-1] < ma[i-1] and c[i] > ma[i]:
                ma_s[i] = 1
            elif c[i-1] > ma[i-1] and c[i] < ma[i]:
                ma_s[i] = -1

            if rsi[i] < 30:
                rsi_s[i] = 1
            elif rsi[i] > 70:
                rsi_s[i] = -1

            if m[i-1] < sig[i-1] and m[i] > sig[i]:
                macd_s[i] = 1
            elif m[i-1] > sig[i-1] and m[i] < sig[i]:
                macd_s[i] = -1

            comp[i]  = ma_s[i] + rsi_s[i] + macd_s[i]
            trade[i] = np.sign(comp[i])

        df["MA_Signal"]     = ma_s
        df["RSI_Signal"]    = rsi_s
        df["MACD_Signal2"]  = macd_s
        df["Composite"]     = comp
        df["Trade"]         = trade
        return df

    # ─────────── Backtester ───────────
    def backtest(df):
        df = df.copy()
        df["Return"]   = df["Close"].pct_change().fillna(0)
        df["Position"] = df["Trade"].shift(1).fillna(0).clip(0,1)
        df["StratRet"] = df["Position"] * df["Return"]
        df["CumBH"]    = (1 + df["Return"]).cumprod()
        df["CumStrat"] = (1 + df["StratRet"]).cumprod()
        dd     = df["CumStrat"] / df["CumStrat"].cummax() - 1
        max_dd = dd.min() * 100
        sd     = df["StratRet"].std()
        sharpe = (df["StratRet"].mean() / sd * np.sqrt(252)) if sd else np.nan
        win_rt = (df["StratRet"] > 0).mean() * 100
        return df, max_dd, sharpe, win_rt

    # ─────────────────── Single‐Ticker Backtest ───────────────────
    st.markdown("## Single‐Ticker Backtest")
    ticker = st.text_input("Ticker (e.g. AAPL)", "AAPL").upper()

    if ticker:
        info  = yf.Ticker(ticker).info
        price = info.get("regularMarketPrice")
        if price is not None:
            st.subheader(f"💲 Live Price: ${price:.2f}")

        # ─── Recent News & Sentiment ───
        raw_news = getattr(yf.Ticker(ticker), "news", []) or []
        shown = 0

        if raw_news:
            st.markdown("### 📰 Recent News & Sentiment")
            for art in raw_news:
                title = art.get("title", "")
                link  = art.get("link", "")
                if not (title and link):
                    continue
                txt   = art.get("summary", title)
                score = analyzer.polarity_scores(txt)["compound"]
                emoji = "🔺" if score > 0.1 else ("🔻" if score < -0.1 else "➖")
                st.markdown(f"- [{title}]({link}) {emoji}")
                shown += 1
                if shown >= 5:
                    break

        if shown == 0:
            st.markdown("### 📰 No yfinance news, falling back to RSS…")
            rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
            feed    = feedparser.parse(rss_url)
            for entry in feed.entries:
                st.markdown(f"- [{entry.title}]({entry.link})")
                shown += 1
                if shown >= 5:
                    break
            if shown == 0:
                st.info("No recent news found.")

    if st.button("▶️ Run Composite Backtest"):
        df_raw = load_and_compute(ticker, ma_window, rsi_period, macd_fast, macd_slow, macd_signal)
        if df_raw.empty:
            st.error(f"No data for '{ticker}'")
            st.stop()

        df_c, max_dd, sharpe, win_rt = backtest(build_composite(df_raw, ma_window, rsi_period))
        rec = rec_map[int(df_c["Trade"].iloc[-1])]
        st.success(f"**{ticker}**: {rec}")

        # Explain why
        ma_s, rsi_s, macd_s = (
            int(df_c["MA_Signal"].iloc[-1]),
            int(df_c["RSI_Signal"].iloc[-1]),
            int(df_c["MACD_Signal2"].iloc[-1])
        )
        try:
            rsi_v = float(df_c[f"RSI{rsi_period}"].iloc[-1])
            valid_rsi = True
        except:
            valid_rsi = False

        ma_txt = {
            1:  f"Price ↑ above {ma_window}-day MA.",
            0:  "No crossover.",
           -1:  f"Price ↓ below {ma_window}-day MA."
        }[ma_s]

        if valid_rsi:
            rsi_txt = {
                1: f"RSI ({rsi_v:.1f}) < 30 → oversold.",
                0: f"RSI ({rsi_v:.1f}) neutral.",
               -1: f"RSI ({rsi_v:.1f}) > 70 → overbought."
            }[rsi_s]
        else:
            rsi_txt = "RSI data unavailable."

        macd_txt = {
            1: "MACD ↑ signal.",
            0: "No crossover.",
           -1: "MACD ↓ signal."
        }[macd_s]

        with st.expander("🔎 Why This Signal?"):
            st.write(f"- **MA:**  {ma_txt}")
            st.write(f"- **RSI:** {rsi_txt}")
            st.write(f"- **MACD:** {macd_txt}")
            st.write(f"- **Composite Score:** {int(df_c['Composite'].iloc[-1])}")

        # Stats & charts
        st.markdown(f"""
- **Buy & Hold:**    {(df_c['CumBH'].iloc[-1]-1)*100:.2f}%  
- **Strategy:**      {(df_c['CumStrat'].iloc[-1]-1)*100:.2f}%  
- **Sharpe:**        {sharpe:.2f}  
- **Max Drawdown:**  {max_dd:.2f}%  
- **Win Rate:**      {win_rt:.1f}%  
""")
        fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        axs[0].plot(df_c["Close"], label="Close")
        axs[0].plot(df_c[f"MA{ma_window}"], label=f"MA{ma_window}")
        axs[0].legend(); axs[0].set_title("Price & MA")
        axs[1].bar(df_c.index, df_c["Composite"], color="purple"); axs[1].set_title("Composite")
        axs[2].plot(df_c["CumBH"], ":", label="BH")
        axs[2].plot(df_c["CumStrat"], "-", label="Strat")
        axs[2].legend(); axs[2].set_title("Equity")
        plt.xticks(rotation=45); plt.tight_layout(); st.pyplot(fig)

    # ─────────── Batch Backtest ───────────
    st.markdown("---")
    st.markdown("## Batch Backtest")
    batch = st.text_area("Tickers (comma-separated)", "AAPL, MSFT, TSLA, SPY, QQQ").upper()
    if st.button("▶️ Run Batch Backtest"):
        perf = []
        for t in [x.strip() for x in batch.split(",") if x.strip()]:
            df_t = load_and_compute(t, ma_window, rsi_period, macd_fast, macd_slow, macd_signal)
            if df_t.empty:
                continue
            df_tc, md, sh, wr = backtest(build_composite(df_t, ma_window, rsi_period))
            perf.append({
                "Ticker":         t,
                "Composite":      int(df_tc["Composite"].iloc[-1]),
                "Signal":         rec_map[int(df_tc["Trade"].iloc[-1])],
                "Buy & Hold %":   (df_tc["CumBH"].iloc[-1]-1)*100,
                "Strategy %":     (df_tc["CumStrat"].iloc[-1]-1)*100,
                "Sharpe":         sh,
                "Max Drawdown":   md,
                "Win Rate":       wr
            })
        if perf:
            df_perf = pd.DataFrame(perf).set_index("Ticker")
            st.dataframe(df_perf, use_container_width=True)
            st.download_button("Download CSV", df_perf.to_csv(), "batch.csv")
        else:
            st.error("No valid data for batch tickers.")

    # ─────────── Midday Movers ───────────
    st.markdown("---")
    st.markdown("## 🌤️ Midday Movers (Intraday % Change)")
    mover_list = st.text_area("Tickers to monitor (comma-separated)", "AAPL, MSFT, TSLA, SPY, QQQ").upper()
    if st.button("🔄 Get Midday Movers"):
        movers = []
        for sym in [s.strip() for s in mover_list.split(",") if s.strip()]:
            tk = yf.Ticker(sym)
            intraday = tk.history(period="1d", interval="5m")
            if intraday.empty:
                intraday = tk.history(period="2d", interval="5m")
                if not intraday.empty:
                    today = pd.Timestamp.utcnow().normalize()
                    intraday = intraday[intraday.index >= today]
            if intraday.empty:
                st.warning(f"No intraday data for {sym}.")
                continue
            open_p = intraday["Open"].iloc[0]
            last_p = intraday["Close"].iloc[-1]
            movers.append({
                "Ticker":   sym,
                "Open":     open_p,
                "Current":  last_p,
                "Change %": (last_p - open_p) / open_p * 100
            })
        if movers:
            df_m = pd.DataFrame(movers).set_index("Ticker")
            df_m["Change %"] = pd.to_numeric(df_m["Change %"], errors="coerce")
            st.dataframe(df_m.sort_values("Change %", ascending=False), use_container_width=True)
        else:
            st.info("No valid intraday data found.")

    # ─────────── Portfolio Simulator ───────────
    st.markdown("---")
    st.markdown("## 📊 Portfolio Simulator")
    st.info("Enter CSV: ticker,shares,cost_basis")
    holdings = st.text_area("e.g.\nAAPL,10,150\nMSFT,5,300", height=100)
    if st.button("▶️ Simulate Portfolio"):
        rows, data = [r.split(",") for r in holdings.splitlines() if r.strip()], []
        for tkr, sh, cb in rows:
            s = float(sh); c = float(cb)
            hist = yf.Ticker(tkr).history(period="1d")
            if hist.empty:
                continue
            price    = hist["Close"].iloc[-1]
            invested = s * c
            val      = s * price
            pnl      = val - invested
            pct      = (pnl / invested * 100) if invested else np.nan
            df_raw = load_and_compute(tkr, ma_window, rsi_period, macd_fast, macd_slow, macd_signal)
            comp_sugg = (
                rec_map[int(build_composite(df_raw, ma_window, rsi_period)["Trade"].iloc[-1])]
                if not df_raw.empty else "N/A"
            )
            if pct > profit_target:
                sugg = "🔴 SELL"
            elif pct < -loss_limit:
                sugg = "🟢 BUY"
            else:
                sugg = comp_sugg
            data.append({
                "Ticker":        tkr,
                "Shares":        s,
                "Cost Basis":    c,
                "Price":         price,
                "Market Value":  val,
                "Invested":      invested,
                "P/L":           pnl,
                "P/L %":         pct,
                "Composite Sig": comp_sugg,
                "Suggestion":    sugg
            })
        if data:
            df_port = pd.DataFrame(data).set_index("Ticker")
            st.dataframe(df_port, use_container_width=True)
            st.metric("Total Market Value", f"${df_port['Market Value'].sum():,.2f}")
            st.metric("Total Invested",     f"${df_port['Invested'].sum():,.2f}")
            st.metric("Total P/L",          f"${df_port['Market Value'].sum()-df_port['Invested'].sum():,.2f}")
            fig, ax = plt.subplots()
            df_port["Market Value"].plot.pie(autopct="%.1f%%", ax=ax)
            ax.set_ylabel("")
            st.pyplot(fig)
        else:
            st.error("No valid holdings.")

    # ─────────── Hyperparameter Optimization ───────────
    st.markdown("---")
    st.markdown("## 🛠️ Hyperparameter Optimization")
    ma_list  = st.sidebar.multiselect("MA windows",      [5,10,15,20,30], default=[ma_window], key="grid_ma")
    rsi_list = st.sidebar.multiselect("RSI lookbacks",   [7,14,21,28],   default=[rsi_period], key="grid_rsi")
    mf_list  = st.sidebar.multiselect("MACD fast spans", [8,12,16,20],  default=[macd_fast],  key="grid_mf")
    ms_list  = st.sidebar.multiselect("MACD slow spans", [20,26,32,40], default=[macd_slow],  key="grid_ms")
    sig_list = st.sidebar.multiselect("MACD sig spans",  [5,9,12,16],   default=[macd_signal], key="grid_sig")
    if st.button("🏃‍♂️ Run Grid Search"):
        if not ticker:
            st.error("Enter a ticker."); st.stop()
        df_full = load_and_compute(ticker, ma_window, rsi_period, macd_fast, macd_slow, macd_signal)
        if df_full.empty:
            st.error(f"No data for '{ticker}'"); st.stop()
        results = []
        with st.spinner("Testing parameter combos…"):
            for mw in ma_list:
                for rp in rsi_list:
                    for mf_ in mf_list:
                        for ms_ in ms_list:
                            for s_ in sig_list:
                                df_i = load_and_compute(ticker, mw, rp, mf_, ms_, s_)
                                if df_i.empty:
                                    continue
                                df_ci, md_i, sh_i, wr_i = backtest(build_composite(df_i, mw, rp))
                                results.append({
                                    "MA":            mw,
                                    "RSI":           rp,
                                    "MACD Fast":     mf_,
                                    "MACD Slow":     ms_,
                                    "MACD Sig":      s_,
                                    "Strategy %":    (df_ci["CumStrat"].iloc[-1] - 1) * 100,
                                    "Sharpe":        sh_i,
                                    "Max Drawdown":  md_i,
                                    "Win Rate":      wr_i
                                })
        if results:
            df_grid = pd.DataFrame(results).sort_values("Strategy %", ascending=False).head(10)
            st.dataframe(df_grid, use_container_width=True)
            st.download_button("Download CSV", df_grid.to_csv(index=False), "grid.csv")
        else:
            st.error("No valid parameter combinations found.")

    # ─────────── Watchlist Summary ───────────
    st.markdown("---")
    st.markdown("## ⏰ Watchlist Summary")
    watch = st.text_area("Enter tickers", "AAPL, MSFT, TSLA, SPY, QQQ").upper()
    if st.button("📬 Generate Watchlist Summary"):
        table = []
        for t in [x.strip() for x in watch.split(",") if x.strip()]:
            df_t = load_and_compute(t, ma_window, rsi_period, macd_fast, macd_slow, macd_signal)
            if df_t.empty:
                table.append({"Ticker": t, "Composite": None, "Signal": "N/A"})
                continue
            df_w, _, _, _ = backtest(build_composite(df_t, ma_window, rsi_period))
            comp = int(df_w["Composite"].iloc[-1])
            sig  = rec_map[int(df_w["Trade"].iloc[-1])]
            table.append({"Ticker": t, "Composite": comp, "Signal": sig})
        df_watch = pd.DataFrame(table).set_index("Ticker")
        st.dataframe(df_watch, use_container_width=True)

        for t in df_watch.index:
            df_t = load_and_compute(t, ma_window, rsi_period, macd_fast, macd_slow, macd_signal)
            if df_t.empty:
                continue
            df_c = build_composite(df_t, ma_window, rsi_period)
            last = df_c.iloc[-1]
            ma_s, rsi_s, macd_s = int(last["MA_Signal"]), int(last["RSI_Signal"]), int(last["MACD_Signal2"])
            try:
                rsi_v = float(last[f"RSI{rsi_period}"])
                valid = True
            except:
                valid = False
            ma_txt = {1: f"↑ above {ma_window}-day MA", 0: "No crossover", -1: f"↓ below {ma_window}-day MA"}[ma_s]
            if valid:
                rsi_txt = {1: "RSI <30 (oversold)", 0: "RSI neutral", -1: "RSI >70 (overbought)"}[rsi_s]
            else:
                rsi_txt = "RSI data unavailable"
            macd_txt = {1: "MACD ↑ signal", 0: "No crossover", -1: "MACD ↓ signal"}[macd_s]
            with st.expander(f"🔎 {t} Reasoning ({df_watch.loc[t,'Signal']})"):
                st.write(f"- **MA:**  {ma_txt}")
                st.write(f"- **RSI:** {rsi_txt}")
                st.write(f"- **MACD:** {macd_txt}")
                st.write(f"- **Composite Score:** {df_watch.loc[t,'Composite']}")
