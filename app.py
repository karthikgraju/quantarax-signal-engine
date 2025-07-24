import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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

**How do I know what slider to use?**  
Every slider trades off **responsiveness** vs. **smoothness**. Below are some quick rules of thumb:

| Slider                 | What it does                                | If you want…                                                              |
|------------------------|---------------------------------------------|---------------------------------------------------------------------------|
| **MA window**          | # of days for moving average               | • **Lower** (5–10) → more responsive, **more** false signals • **Higher** (20–50) → smoother, **fewer** signals |
| **RSI lookback**       | Period for RSI’s EMA calculation           | • **Short** (5–10) → choppier, react to short-term swings • **Long** (20–30) → stable, ignores minor noise        |
| **MACD fast span**     | EMA span for MACD’s fast line              | • **Lower** (5–10) → very quick to shift • **Higher** (15–20) → slower changes                                      |
| **MACD slow span**     | EMA span for MACD’s slow line              | • Don’t set too close to fast span — keep at least +10 days difference                                                  |
| **MACD sig span**      | EMA span for MACD’s signal line            | • **Lower** (5–9) → quick crossover triggers • **Higher** (12–16) → avoids whipsaws                                  |
| **Profit target**      | Unrealized P/L% at which to **override** to SELL  | • Set your personal upside threshold—e.g. 5–20% • Smaller → take profits quickly, larger → ride trends longer       |
| **Loss limit**         | Unrealized P/L% at which to **override** to BUY   | • Set your personal risk tolerance—e.g. 3–10% • Smaller → tighter stops, larger → more wiggle room                  |

> **Tip:** start with the **defaults** (MA=10, RSI=14, MACD=12/26/9), and **tweak one at a time**.  
> Watch how your backtest return, drawdown and win-rate change, then lock in the combination that matches your style.

---

## 🏆 Objectives

1. **Deliver** a polished MVP by week’s end for investor demos.  
2. **Onboard** 100+ beta users in the next 30 days and iterate on feedback.  
3. **Integrate** real-time streaming data & push notifications (Q3).  
4. **Expand** to crypto, forex, and alternative data sets (Q4).  
5. **Build** community-driven features—strategy sharing, crowd sentiment.

---

## 1. Indicators

1. **Moving Average (MA) Crossover**  
   - **What it is**: The N-day simple moving average of closing prices.  
   - **Signal logic**:  
     - **Bull (+1)** when price crosses **up** through the MA (momentum turning positive).  
     - **Bear (–1)** when price crosses **down** through the MA.  
     - **Neutral (0)** otherwise.

2. **Relative Strength Index (RSI)**  
   - **What it is**: A momentum oscillator over a lookback period, measuring gains vs. losses.  
   - **Signal logic**:  
     - **Bull (+1)** if RSI < 30 → oversold territory.  
     - **Bear (–1)** if RSI > 70 → overbought territory.  
     - **Neutral (0)** when 30 ≤ RSI ≤ 70.

3. **MACD Crossover**  
   - **What it is**: Difference between a fast and slow EMA, plus its signal‐line EMA.  
   - **Signal logic**:  
     - **Bull (+1)** when MACD line crosses **above** its signal line.  
     - **Bear (–1)** when MACD line crosses **below** its signal line.  
     - **Neutral (0)** otherwise.

Each indicator outputs –1, 0 or +1. We sum (“composite”) to a range of –3…+3, then take **sign(composite)**:  
• +1/+2/+3 → **BUY** 🟢  
• 0 → **HOLD** 🟡  
• –1/–2/–3 → **SELL** 🔴

---

## 2. Slider Controls (Sidebar)

- **MA window**  
  Number of days for the simple moving average.  
  Larger windows → smoother MA, fewer crossovers.

- **RSI lookback**  
  Period over which gains/losses are averaged.  
  Shorter lookback → more reactive; longer → smoother.

- **MACD fast span**  
  EMA span for the “fast” line. Typical defaults: 12.

- **MACD slow span**  
  EMA span for the “slow” line. Typical defaults: 26.

- **MACD sig span**  
  EMA span for the signal line. Typical defaults: 9.

Click **Reset to defaults** to restore the above to (10, 14, 12, 26, 9).

---

## 3. Single-Ticker Backtest

1. Enter a **Ticker** (e.g. `AAPL`).  
2. **Run Composite Backtest**  
   - Downloads last 6 months of data, computes MA/RSI/MACD signals.  
   - Shows:
     - **Live price** and top 5 recent news items (with sentiment emoji).  
     - **Final recommendation** (BUY/HOLD/SELL).  
     - **Why?** breakdown of each indicator in an expander.  
     - Performance stats:
       - Buy-&-Hold vs. strategy returns
       - Sharpe ratio, max drawdown, win rate
     - Charts: price & MA, composite vote, and equity curves.

---

## 4. Batch Backtest

- Enter a **comma-separated list** of tickers (e.g. `AAPL, MSFT, TSLA`).  
- Click **Run Batch Backtest** to compare them side-by-side:
  - Last composite score & signal
  - Buy-&-hold vs. strategy returns
  - Sharpe, max drawdown, win rate
- Download results as CSV.

---

## 5. Portfolio Simulator

1. **Profit target (%)** and **Loss limit (%)** sliders  
   - If unrealized P/L% > _target_ → force a **SELL** suggestion.  
   - If unrealized P/L% < –_limit_ → force a **BUY** suggestion.  
2. Enter your positions in CSV format, one per line:  
   ```txt
   TICKER,shares,cost_basis
   AAPL,10,150
   MSFT,5,300 
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

    st.title("🚀 QuantaraX — Composite Signal Engine")
    st.write("MA + RSI + MACD Composite Signals & Backtest")

    # ─────────── Data Loading ───────────
    @st.cache_data(show_spinner=False)
    def load_and_compute(ticker, ma_w, rsi_p, mf, ms, sig):
        df = yf.download(ticker, period="6mo", progress=False)
        if df.empty or "Close" not in df:
            return pd.DataFrame()
        # MA
        ma_col = f"MA{ma_w}"
        df[ma_col] = df["Close"].rolling(ma_w).mean()
        # RSI
        d = df["Close"].diff()
        up = d.clip(lower=0)
        dn = -d.clip(upper=0)
        ema_up   = up.ewm(com=rsi_p-1, adjust=False).mean()
        ema_down = dn.ewm(com=rsi_p-1, adjust=False).mean()
        rsi_col  = f"RSI{rsi_p}"
        df[rsi_col] = 100 - 100/(1 + ema_up/ema_down)
        # MACD
        ema_f    = df["Close"].ewm(span=mf, adjust=False).mean()
        ema_s    = df["Close"].ewm(span=ms, adjust=False).mean()
        macd     = ema_f - ema_s
        macd_sig = macd.ewm(span=sig, adjust=False).mean()
        df["MACD"] = macd
        df["MACD_Signal"] = macd_sig
        # Drop NAs
        cols = [ma_col, rsi_col, "MACD", "MACD_Signal"]
        prs  = [c for c in cols if c in df.columns]
        if prs:
            try:
                df = df.dropna(subset=prs).reset_index(drop=True)
            except KeyError:
                pass
        return df

    # ─────────── Composite Signals ───────────
    def build_composite(df, ma_w, rsi_p):
        n = len(df)
        close, ma = df["Close"].to_numpy(), df[f"MA{ma_w}"].to_numpy()
        rsi = df[f"RSI{rsi_p}"].to_numpy()
        macd, sig = df["MACD"].to_numpy(), df["MACD_Signal"].to_numpy()
        ma_sig = np.zeros(n,int)
        rsi_sig = np.zeros(n,int)
        macd_sig2 = np.zeros(n,int)
        comp = np.zeros(n,int)
        trade = np.zeros(n,int)
        for i in range(1,n):
            if close[i-1] < ma[i-1] and close[i] > ma[i]:
                ma_sig[i] = 1
            elif close[i-1] > ma[i-1] and close[i] < ma[i]:
                ma_sig[i] = -1
            if rsi[i] < 30:
                rsi_sig[i] = 1
            elif rsi[i] > 70:
                rsi_sig[i] = -1
            if macd[i-1] < sig[i-1] and macd[i] > sig[i]:
                macd_sig2[i] = 1
            elif macd[i-1] > sig[i-1] and macd[i] < sig[i]:
                macd_sig2[i] = -1
            comp[i] = ma_sig[i] + rsi_sig[i] + macd_sig2[i]
            trade[i] = np.sign(comp[i])
        df["MA_Signal"]    = ma_sig
        df["RSI_Signal"]   = rsi_sig
        df["MACD_Signal2"] = macd_sig2
        df["Composite"]    = comp
        df["Trade"]        = trade
        return df

    # ─────────── Backtest ───────────
    def backtest(df):
        df = df.copy()
        df["Return"]   = df["Close"].pct_change().fillna(0)
        df["Position"] = df["Trade"].shift(1).fillna(0).clip(0,1)
        df["StratRet"] = df["Position"] * df["Return"]
        df["CumBH"]    = (1 + df["Return"]).cumprod()
        df["CumStrat"] = (1 + df["StratRet"]).cumprod()
        dd = df["CumStrat"] / df["CumStrat"].cummax() - 1
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
                if shown >= 5: break
            if shown == 0:
                st.info("No recent news found.")
        else:
            st.info("No recent news found.")

    if st.button("▶️ Run Composite Backtest"):
        df_raw, = load_and_compute(ticker, ma_window, rsi_period, macd_fast, macd_slow, macd_signal),
        if df_raw.empty:
            st.error(f"No data for '{ticker}'")
            st.stop()

        df_c, max_dd, sharpe, win_rt = backtest(build_composite(df_raw, ma_window, rsi_period))
        rec = rec_map[int(df_c["Trade"].iloc[-1])]
        st.success(f"**{ticker}**: {rec}")

        # Explain
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
            1: f"Price ↑ above {ma_window}-day MA.",
            0: "No crossover.",
           -1: f"Price ↓ below {ma_window}-day MA."
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
                "Ticker":        t,
                "Composite":     int(df_tc["Composite"].iloc[-1]),
                "Signal":        rec_map[int(df_tc["Trade"].iloc[-1])],
                "Buy & Hold %":  (df_tc["CumBH"].iloc[-1]-1)*100,
                "Strategy %":    (df_tc["CumStrat"].iloc[-1]-1)*100,
                "Sharpe":        sh,
                "Max Drawdown":  md,
                "Win Rate":      wr
            })
        if perf:
            df_perf = pd.DataFrame(perf).set_index("Ticker")
            st.dataframe(df_perf, use_container_width=True)
            st.download_button("Download CSV", df_perf.to_csv(), "batch.csv")
        else:
            st.error("No valid data for batch tickers.")

# ─────────── Midday Movers ───────────
st.markdown("---")
st.markdown("## 🌗 Midday Movers")
st.info("Enter comma-separated tickers to see today’s % change from open.")

mid_tickers = st.text_input("Tickers (e.g. AAPL, MSFT, TSLA)", "AAPL, MSFT, TSLA").upper()
if st.button("🔍 Get Midday Movers"):
    tickers = [t.strip() for t in mid_tickers.split(",") if t.strip()]
    movers  = []
    for t in tickers:
        # grab today’s 5-minute bars
        hist = yf.download(t, period="1d", interval="5m", progress=False)
        if hist.empty:
            continue
        open_price    = hist["Open"].iloc[0]
        current_price = hist["Close"].iloc[-1]
        change_pct    = (current_price - open_price) / open_price * 100
        movers.append({
            "Ticker":      t,
            "Open":        open_price,
            "Current":     current_price,
            "Change %":    change_pct
        })

    if movers:
        df_movers = pd.DataFrame(movers).set_index("Ticker")
        df_movers = df_movers.sort_values("Change %", ascending=False)
        st.dataframe(df_movers, use_container_width=True)
    else:
        st.warning("No data available for those tickers.")


    # ─────────── Portfolio Simulator ───────────
    st.markdown("---")
    st.markdown("## 📊 Portfolio Simulator")

    profit_target = st.sidebar.slider(
        "Profit target (%)", min_value=1, max_value=100, value=5,
        help="If unrealized P/L% exceeds this → SELL"
    )
    loss_limit = st.sidebar.slider(
        "Loss limit (%)", min_value=1, max_value=100, value=5,
        help="If unrealized P/L% falls below –this → BUY"
    )

    st.info("Enter your positions in CSV: ticker,shares,cost_basis")
    holdings = st.text_area("e.g.\nAAPL,10,150\nMSFT,5,300", height=100)

    if st.button("▶️ Simulate Portfolio"):
        rows = [r.strip().split(",") for r in holdings.splitlines() if r.strip()]
        data = []
        for ticker_, shares, cost in rows:
            tkr = ticker_.upper().strip()
            try:
                s = float(shares)
                c = float(cost)
            except:
                continue

            hist = yf.Ticker(tkr).history(period="1d")
            if hist.empty:
                continue
            price    = hist["Close"].iloc[-1]
            invested = s * c
            value    = s * price
            pnl      = value - invested
            pnl_pct  = (pnl / invested * 100) if invested else np.nan

            df_raw = load_and_compute(tkr, ma_window, rsi_period, macd_fast, macd_slow, macd_signal)
            if df_raw.empty:
                comp_sugg = "N/A"
            else:
                df_c     = build_composite(df_raw, ma_window, rsi_period)
                last_sig = int(df_c["Trade"].iloc[-1])
                comp_sugg = rec_map.get(last_sig, "🟡 HOLD")

            if pnl_pct > profit_target:
                suggestion = "🔴 SELL"
            elif pnl_pct < -loss_limit:
                suggestion = "🟢 BUY"
            else:
                suggestion = comp_sugg if comp_sugg in rec_map.values() else "🟡 HOLD"

            data.append({
                "Ticker":        tkr,
                "Shares":        s,
                "Cost Basis":    c,
                "Price":         price,
                "Market Value":  value,
                "Invested":      invested,
                "P/L":           pnl,
                "P/L %":         pnl_pct,
                "Composite Sig": comp_sugg,
                "Suggestion":    suggestion,
            })

        if data:
            df_port = pd.DataFrame(data).set_index("Ticker")
            st.dataframe(df_port, use_container_width=True)
            total_mv  = df_port["Market Value"].sum()
            total_inv = df_port["Invested"].sum()
            st.metric("Total Market Value", f"${total_mv:,.2f}")
            st.metric("Total Invested",     f"${total_inv:,.2f}")
            st.metric("Total P/L",          f"${total_mv - total_inv:,.2f}")
            fig, ax = plt.subplots()
            df_port["Market Value"].plot.pie(autopct="%.1f%%", ax=ax)
            ax.set_ylabel("")
            ax.set_title("Portfolio Allocation")
            st.pyplot(fig)
        else:
            st.error("No valid holdings provided.")

    # ─────────── Hyperparameter Optimization ───────────
    st.markdown("---")
    st.markdown("## 🛠️ Hyperparameter Optimization")
    ma_list  = st.sidebar.multiselect("MA windows",     [5,10,15,20,30], default=[ma_window], key="grid_ma")
    rsi_list = st.sidebar.multiselect("RSI lookbacks",  [7,14,21,28],   default=[rsi_period], key="grid_rsi")
    mf_list  = st.sidebar.multiselect("MACD fast spans",[8,12,16,20], default=[macd_fast], key="grid_mf")
    ms_list  = st.sidebar.multiselect("MACD slow spans",[20,26,32,40], default=[macd_slow],   key="grid_ms")
    sig_list = st.sidebar.multiselect("MACD sig spans", [5,9,12,16],   default=[macd_signal], key="grid_sig")

    if st.button("🏃‍♂️ Run Grid Search"):
        if not ticker:
            st.error("Enter a ticker first."); st.stop()
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
                                if df_i.empty: continue
                                df_ci, md_i, sh_i, wr_i = backtest(build_composite(df_i, mw, rp))
                                results.append({
                                    "MA": mw,
                                    "RSI": rp,
                                    "MACD Fast": mf_,
                                    "MACD Slow": ms_,
                                    "MACD Sig":  s_,
                                    "Strategy %": (df_ci["CumStrat"].iloc[-1] - 1)*100,
                                    "Sharpe":     sh_i,
                                    "Max Drawdown": md_i,
                                    "Win Rate":     wr_i
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
            ma_s   = int(last["MA_Signal"])
            rsi_s  = int(last["RSI_Signal"])
            macd_s = int(last["MACD_Signal2"])
            try:
                rsi_v = float(last[f"RSI{rsi_period}"])
                valid = True
            except:
                valid = False
            ma_txt = {
                1: f"Price ↑ above {ma_window}-day MA.",
                0: "No crossover.",
               -1: f"Price ↓ below {ma_window}-day MA."
            }[ma_s]
            if valid:
                rsi_txt = {
                    1: f"RSI ({rsi_v:.1f}) < 30 → oversold.",
                    0: f"RSI ({rsi_v:.1f}) neutral.",
                   -1: f"RSI ({rsi_v:.1f}) > 70 → overbought."
                }[rsi_s]
            else:
                rsi_txt = "RSI data unavailable."
            macd_txt = {
                1: "MACD line crossed **above** its signal line.",
                0: "No crossover.",
               -1: "MACD line crossed **below** its signal line."
            }[macd_s]
            with st.expander(f"🔎 {t} Reasoning ({df_watch.loc[t,'Signal']})"):
                st.write(f"- **MA:**  {ma_txt}")
                st.write(f"- **RSI:** {rsi_txt}")
                st.write(f"- **MACD:** {macd_txt}")
                st.write(f"- **Composite Score:** {df_watch.loc[t,'Composite']}")
