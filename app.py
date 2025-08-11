import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import feedparser

# ───────────────────────────── Page Setup ─────────────────────────────
st.set_page_config(page_title="QuantaraX Composite Signals BETA v2+", layout="wide")
analyzer = SentimentIntensityAnalyzer()

# ───────────────────────────── Mappings ─────────────────────────────
rec_map = { 1: "🟢 BUY", 0: "🟡 HOLD", -1: "🔴 SELL" }

# ───────────────────────────── Tabs ─────────────────────────────
tab_engine, tab_help = st.tabs(["🚀 Engine", "❓ How It Works"])

# ───────────────────────────── Help Tab ─────────────────────────────
with tab_help:
    st.header("How QuantaraX Works")
    st.markdown(r"""
Welcome to **QuantaraX** — now with **weighted composites**, **ATR risk controls**, **shorting + trading costs**, and **interval/period** controls.

**What’s new vs your last build**
- Weighted composite: MA/RSI/MACD (+ optional Bollinger Bands) with per-indicator weights and a trigger threshold.
- Risk overlay: ATR stop/target multiples and **trading costs (bps)** included in backtests.
- Shorting: optional, with correct PnL math.
- More controls: choose **period** (6mo/1y/2y/5y) and **interval** (1d/1h).
- Richer metrics: CAGR, Sharpe, Max Drawdown, Win Rate, Trades, Time-in-Market.
- Stronger data handling, caching and crypto pair mapping (`BTC/USDT` → `BTC-USD`).

Made in Toronto, Canada by KG
---
""")

# ───────────────────────────── Engine Tab ─────────────────────────────
with tab_engine:

    # ─────────── Defaults & Session State ───────────
    DEFAULTS = dict(ma_window=10, rsi_period=14, macd_fast=12, macd_slow=26, macd_signal=9)
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ─────────── Sidebar Controls ───────────
    st.sidebar.header("Controls")
    if st.sidebar.button("🔄 Reset to defaults"):
        for k, v in DEFAULTS.items():
            st.session_state[k] = v

    st.sidebar.subheader("Indicator Parameters")
    ma_window   = st.sidebar.slider("MA window",      5, 60, st.session_state["ma_window"],   key="ma_window")
    rsi_period  = st.sidebar.slider("RSI lookback",   5, 30, st.session_state["rsi_period"],  key="rsi_period")
    macd_fast   = st.sidebar.slider("MACD fast span", 5, 20, st.session_state["macd_fast"],   key="macd_fast")
    macd_slow   = st.sidebar.slider("MACD slow span", 20, 50, st.session_state["macd_slow"],  key="macd_slow")
    macd_signal = st.sidebar.slider("MACD sig span",  5, 20, st.session_state["macd_signal"], key="macd_signal")

    st.sidebar.subheader("Composite v2 (advanced)")
    use_weighted = st.sidebar.toggle("Use weighted composite", value=True)
    include_bb   = st.sidebar.toggle("Include Bollinger Bands in composite", value=True)
    w_ma   = st.sidebar.slider("Weight • MA",   0.0, 2.0, 1.0, 0.1)
    w_rsi  = st.sidebar.slider("Weight • RSI",  0.0, 2.0, 1.0, 0.1)
    w_macd = st.sidebar.slider("Weight • MACD", 0.0, 2.0, 1.0, 0.1)
    w_bb   = st.sidebar.slider("Weight • BB",   0.0, 2.0, 0.5, 0.1) if include_bb else 0.0
    comp_thr = st.sidebar.slider("Composite trigger (enter/exit)", 0.0, 3.0, 1.0, 0.1)

    st.sidebar.subheader("Risk & Costs")
    allow_short = st.sidebar.toggle("Allow shorts", value=False)
    cost_bps    = st.sidebar.slider("Trading cost (bps/side)", 0.0, 25.0, 5.0, 0.5)
    sl_atr_mult = st.sidebar.slider("Stop • ATR ×", 0.0, 5.0, 2.0, 0.1)
    tp_atr_mult = st.sidebar.slider("Target • ATR ×", 0.0, 8.0, 3.0, 0.1)

    st.sidebar.subheader("Data")
    period_sel   = st.sidebar.selectbox("History", ["6mo","1y","2y","5y"], index=1)
    interval_sel = st.sidebar.selectbox("Interval", ["1d","1h"], index=0)

    # ─────────── Profit/Loss Sliders for Portfolio ───────────
    st.sidebar.subheader("Portfolio Guardrails")
    profit_target = st.sidebar.slider("Profit target (%)", 1, 100, 10,
                                      help="If unrealized P/L% exceeds this → SELL")
    loss_limit    = st.sidebar.slider("Loss limit (%)", 1, 100, 5,
                                      help="If unrealized P/L% falls below –this → BUY")

    st.title("🚀 QuantaraX — Composite Signal Engine (v2+)")

    # ─────────── Helpers ───────────
    def _map_symbol(sym: str) -> str:
        s = sym.strip().upper()
        if "/" in s:  # e.g., BTC/USDT → BTC-USD
            base, quote = s.split("/")
            quote = "USD" if quote in ("USDT","USD") else quote
            return f"{base}-{quote}"
        return s

    @st.cache_data(show_spinner=False, ttl=900)
    def load_and_compute(ticker, ma_w, rsi_p, mf, ms, sig, period, interval, include_bb):
        sym = _map_symbol(ticker)
        df = yf.download(sym, period=period, interval=interval, auto_adjust=False, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
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
        rs = ema_up / ema_down.replace(0, np.nan)
        df[rsi_col] = 100 - 100/(1 + rs)

        # MACD
        ema_f    = df["Close"].ewm(span=mf, adjust=False).mean()
        ema_s    = df["Close"].ewm(span=ms, adjust=False).mean()
        macd     = ema_f - ema_s
        macd_sig = macd.ewm(span=sig, adjust=False).mean()
        df["MACD"] = macd; df["MACD_Signal"] = macd_sig

        # ATR (for risk)
        prev_close = df["Close"].shift(1)
        tr = pd.concat([(df["High"]-df["Low"]).abs(),
                        (df["High"]-prev_close).abs(),
                        (df["Low"]-prev_close).abs()], axis=1).max(axis=1)
        df["ATR"] = tr.ewm(alpha=1/14, adjust=False).mean()

        # Bollinger Bands (optional)
        if include_bb:
            bb_w = 20; bb_k = 2.0
            m = df["Close"].rolling(bb_w).mean()
            sd = df["Close"].rolling(bb_w).std(ddof=0)
            df["BB_M"], df["BB_U"], df["BB_L"] = m, m + bb_k*sd, m - bb_k*sd

        # Drop NAs
        cols = [ma_col, rsi_col, "MACD", "MACD_Signal"]
        if include_bb: cols += ["BB_M","BB_U","BB_L"]
        df = df.dropna(subset=[c for c in cols if c in df.columns]).reset_index(drop=False)
        return df

    # ─────────── Composite Signals ───────────
    def build_composite(df, ma_w, rsi_p, *,
                        use_weighted=True, w_ma=1.0, w_rsi=1.0, w_macd=1.0, w_bb=0.0,
                        include_bb=False, threshold=0.0, allow_short=False):
        df = df.copy()
        n = len(df)
        close, ma = df["Close"].to_numpy(), df[f"MA{ma_w}"].to_numpy()
        rsi = df[f"RSI{rsi_p}"].to_numpy()
        macd = df["MACD"].to_numpy()
        sigl = df["MACD_Signal"].to_numpy()

        # Classic discrete components (match your original semantics)
        ma_sig   = np.zeros(n, int)
        rsi_sig  = np.zeros(n, int)
        macd_sig2= np.zeros(n, int)
        bb_sig   = np.zeros(n, int)

        for i in range(1, n):
            # MA crossover
            if close[i-1] < ma[i-1] and close[i] > ma[i]:   ma_sig[i] = 1
            elif close[i-1] > ma[i-1] and close[i] < ma[i]: ma_sig[i] = -1
            # RSI zones
            if rsi[i] < 30:   rsi_sig[i] = 1
            elif rsi[i] > 70: rsi_sig[i] = -1
            # MACD crossover
            if macd[i-1] < sigl[i-1] and macd[i] > sigl[i]:   macd_sig2[i] = 1
            elif macd[i-1] > sigl[i-1] and macd[i] < sigl[i]: macd_sig2[i] = -1
            # BB extremes (if present)
            if include_bb and {"BB_U","BB_L"}.issubset(df.columns):
                if close[i] < df["BB_L"].iloc[i]: bb_sig[i] = 1
                elif close[i] > df["BB_U"].iloc[i]: bb_sig[i] = -1

        # Weighted composite
        if use_weighted:
            comp = w_ma*ma_sig + w_rsi*rsi_sig + w_macd*macd_sig2 + (w_bb*bb_sig if include_bb else 0)
        else:
            comp = ma_sig + rsi_sig + macd_sig2  # original

        # Trade signal
        if allow_short:
            trade = np.where(comp >= threshold, 1, np.where(comp <= -threshold, -1, 0))
        else:
            trade = np.where(comp >= threshold, 1, 0)

        df["MA_Signal"] = ma_sig
        df["RSI_Signal"] = rsi_sig
        df["MACD_Signal2"] = macd_sig2
        if include_bb: df["BB_Signal"] = bb_sig
        df["Composite"] = comp.astype(float)
        df["Trade"] = trade.astype(int)
        return df

    # ─────────── Backtest (supports shorts + costs + ATR exits) ───────────
    def backtest(df, *, allow_short=False, cost_bps=0.0, sl_atr_mult=0.0, tp_atr_mult=0.0):
        d = df.copy()
        d["Return"] = d["Close"].pct_change().fillna(0.0)

        # Positioning
        if allow_short:
            d["Position"] = d["Trade"].shift(1).fillna(0).clip(-1,1)
            strat_ret = np.where(d["Position"]>=0, d["Return"], -d["Return"])
        else:
            d["Position"] = d["Trade"].shift(1).fillna(0).clip(0,1)
            strat_ret = d["Position"] * d["Return"]

        # Costs on position changes
        cost = cost_bps / 10000.0
        pos_change = d["Position"].diff().fillna(0).abs()
        # Opening + closing both cost a side each
        cost_ret = -cost * (pos_change > 0).astype(float) * 2.0
        d["StratRet"] = strat_ret + cost_ret

        # Simple ATR exits (if enabled): when hit, force flat next bar
        if sl_atr_mult > 0 or tp_atr_mult > 0:
            flat = np.zeros(len(d), dtype=int)
            entry_px = np.nan
            for i in range(len(d)):
                p = d["Position"].iat[i]
                c = d["Close"].iat[i]
                a = d["ATR"].iat[i] if "ATR" in d.columns else np.nan
                if p != 0 and np.isnan(entry_px):
                    entry_px = c
                if p == 0:
                    entry_px = np.nan
                if p != 0 and not np.isnan(a):
                    if p == 1:
                        if c <= entry_px - sl_atr_mult*a or c >= entry_px + tp_atr_mult*a:
                            flat[i] = 1
                            entry_px = np.nan
                    else:  # short
                        if c >= entry_px + sl_atr_mult*a or c <= entry_px - tp_atr_mult*a:
                            flat[i] = 1
                            entry_px = np.nan
            if flat.any():
                d.loc[flat==1, "Position"] = 0

        d["CumBH"]    = (1 + d["Return"]).cumprod()
        d["CumStrat"] = (1 + d["StratRet"]).cumprod()
        dd  = d["CumStrat"] / d["CumStrat"].cummax() - 1
        max_dd = float(dd.min()*100)
        ann = 252 if pd.api.types.is_datetime64_any_dtype(d.iloc[:,0]) else 252
        m  = d["StratRet"].mean()*ann
        v  = d["StratRet"].std(ddof=0)*np.sqrt(ann)
        sharpe = float(m/v) if v>0 else np.nan
        win_rt = float((d["StratRet"]>0).mean()*100)
        trades = int((pos_change>0).sum())
        tim = float((d["Position"]!=0).mean()*100)
        # CAGR
        n = len(d); cagr = (d["CumStrat"].iat[-1] ** (ann/max(n,1)) - 1) * 100 if n>0 else np.nan
        return d, max_dd, sharpe, win_rt, trades, tim, cagr

    # ─────────────────── Single‐Ticker Backtest ───────────────────
    st.markdown("## Single‐Ticker Backtest")
    ticker = st.text_input("Ticker (e.g. AAPL or BTC/USDT)", "AAPL").upper()

    if ticker:
        # Live-ish price from 1d bar (more reliable than .info)
        h = yf.download(_map_symbol(ticker), period="1d", progress=False)
        if not h.empty and "Close" in h:
            st.subheader(f"💲 Live Price: ${float(h['Close'].iloc[-1]):.2f}")

        # ─── Dual‐Source News Feed ───
        raw_news = getattr(yf.Ticker(_map_symbol(ticker)), "news", []) or []
        shown = 0
        if raw_news:
            st.markdown("### 📰 Recent News & Sentiment (YFinance)")
            for art in raw_news:
                t = art.get("title",""); l = art.get("link","")
                if not (t and l): continue
                txt = art.get("summary", t)
                score = analyzer.polarity_scores(txt)["compound"]
                emoji = "🔺" if score>0.1 else ("🔻" if score<-0.1 else "➖")
                st.markdown(f"- [{t}]({l}) {emoji}")
                shown += 1
                if shown >= 5: break
        if shown == 0:
            st.markdown("### 📰 Recent News (RSS)")
            rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={_map_symbol(ticker)}&region=US&lang=en-US"
            feed = feedparser.parse(rss_url)
            for entry in feed.entries[:5]:
                st.markdown(f"- [{entry.title}]({entry.link})")
                shown += 1
        if shown == 0:
            st.info("No recent news found.")

    if st.button("▶️ Run Composite Backtest"):
        df_raw = load_and_compute(ticker, ma_window, rsi_period, macd_fast, macd_slow, macd_signal,
                                  period_sel, interval_sel, include_bb)
        if df_raw.empty:
            st.error(f"No data for '{ticker}'"); st.stop()

        df_sig = build_composite(df_raw, ma_window, rsi_period,
                                 use_weighted=use_weighted, w_ma=w_ma, w_rsi=w_rsi, w_macd=w_macd, w_bb=w_bb,
                                 include_bb=include_bb, threshold=comp_thr, allow_short=allow_short)

        df_c, max_dd, sharpe, win_rt, trades, tim, cagr = backtest(
            df_sig, allow_short=allow_short, cost_bps=cost_bps,
            sl_atr_mult=sl_atr_mult, tp_atr_mult=tp_atr_mult
        )

        last_trade = int(df_c["Trade"].iloc[-1])
        rec = rec_map.get(last_trade if not allow_short else (1 if last_trade>0 else (-1 if last_trade<0 else 0)), "🟡 HOLD")
        st.success(f"**{ticker}**: {rec}")

        # Explanation
        last = df_sig.iloc[-1]
        ma_s, rsi_s, macd_s = int(last["MA_Signal"]), int(last["RSI_Signal"]), int(last["MACD_Signal2"])
        ma_txt = {1:f"Price ↑ crossed above MA{ma_window}.", 0:"No MA crossover.", -1:f"Price ↓ crossed below MA{ma_window}."}[ma_s]
        rsi_txt = {1:f"RSI ({last[f'RSI{rsi_period}']:.1f}) < 30 → oversold.",
                   0:f"RSI ({last[f'RSI{rsi_period}']:.1f}) neutral.",
                  -1:f"RSI ({last[f'RSI{rsi_period}']:.1f}) > 70 → overbought."}[rsi_s]
        macd_txt = {1:"MACD ↑ crossed above signal.", 0:"No MACD crossover.", -1:"MACD ↓ crossed below signal."}[macd_s]
        with st.expander("🔎 Why This Signal?"):
            st.write(f"- **MA:**  {ma_txt}")
            st.write(f"- **RSI:** {rsi_txt}")
            st.write(f"- **MACD:** {macd_txt}")
            if include_bb and "BB_Signal" in df_sig.columns:
                bb_s = int(last["BB_Signal"])
                bb_txt = {1:"Close under lower band (mean-revert long).",0:"Inside bands.",-1:"Close over upper band (mean-revert short)."}[bb_s]
                st.write(f"- **BB:** {bb_txt}")
            st.write(f"- **Composite (weighted):** {float(last['Composite']):.2f}  (threshold={comp_thr:.1f})")

        # Metrics
        colA, colB, colC, colD, colE, colF = st.columns(6)
        colA.metric("CAGR", f"{cagr:.2f}%")
        colB.metric("Sharpe", f"{sharpe:.2f}")
        colC.metric("Max DD", f"{max_dd:.2f}%")
        colD.metric("Win Rate", f"{win_rt:.1f}%")
        colE.metric("Trades", f"{trades}")
        colF.metric("Time in Mkt", f"{tim:.1f}%")

        st.markdown(f"""
- **Buy & Hold:**    {(df_c['CumBH'].iloc[-1]-1)*100:.2f}%  
- **Strategy:**      {(df_c['CumStrat'].iloc[-1]-1)*100:.2f}%
""")

        # Plots
        fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(11,12), sharex=True)
        ax1.plot(df_c["Date"] if "Date" in df_c.columns else df_c.index, df_c["Close"], label="Close")
        ax1.plot(df_c["Date"] if "Date" in df_c.columns else df_c.index, df_c[f"MA{ma_window}"], label=f"MA{ma_window}")
        if include_bb and {"BB_U","BB_L"}.issubset(df_c.columns):
            ax1.plot(df_c["BB_U"], label="BB Upper"); ax1.plot(df_c["BB_L"], label="BB Lower")
        ax1.legend(); ax1.set_title("Price & Indicators")

        ax2.bar(df_c["Date"] if "Date" in df_c.columns else df_c.index, df_c["Composite"]); ax2.set_title("Composite (weighted)")

        ax3.plot(df_c["CumBH"], ":", label="BH")
        ax3.plot(df_c["CumStrat"], "-", label="Strat"); ax3.legend(); ax3.set_title("Equity")
        plt.xticks(rotation=45); plt.tight_layout()
        st.pyplot(fig)

    # ─────────── Batch Backtest ───────────
    st.markdown("---"); st.markdown("## Batch Backtest")
    batch = st.text_area("Tickers (comma-separated)", "AAPL, MSFT, TSLA, SPY, QQQ").upper()
    if st.button("▶️ Run Batch Backtest"):
        perf=[]
        for t in [x.strip() for x in batch.split(",") if x.strip()]:
            df_t = load_and_compute(t, ma_window, rsi_period, macd_fast, macd_slow, macd_signal,
                                    period_sel, interval_sel, include_bb)
            if df_t.empty: continue
            df_tc = build_composite(df_t, ma_window, rsi_period,
                                    use_weighted=use_weighted, w_ma=w_ma, w_rsi=w_rsi, w_macd=w_macd, w_bb=w_bb,
                                    include_bb=include_bb, threshold=comp_thr, allow_short=allow_short)
            df_tc, md, sh, wr, trd, tim, cagr = backtest(df_tc, allow_short=allow_short, cost_bps=cost_bps,
                                                         sl_atr_mult=sl_atr_mult, tp_atr_mult=tp_atr_mult)
            perf.append({
                "Ticker":t,
                "Composite":float(df_tc["Composite"].iloc[-1]),
                "Signal": rec_map.get(int(np.sign(df_tc["Composite"].iloc[-1])), "🟡 HOLD"),
                "Buy & Hold %": (df_tc["CumBH"].iloc[-1]-1)*100,
                "Strategy %":   (df_tc["CumStrat"].iloc[-1]-1)*100,
                "Sharpe":       sh,
                "Max Drawdown": md,
                "Win Rate":     wr,
                "Trades":       trd,
                "CAGR %":       cagr
            })
        if perf:
            df_perf = pd.DataFrame(perf).set_index("Ticker").sort_values("Strategy %", ascending=False)
            st.dataframe(df_perf, use_container_width=True)
            st.download_button("Download CSV", df_perf.to_csv(), "batch.csv")
        else:
            st.error("No valid data for batch tickers.")

    # ─────────── Midday Movers ───────────
    st.markdown("---"); st.markdown("## 🌤️ Midday Movers (Intraday % Change)")
    mover_list = st.text_area("Tickers to monitor (comma-separated)", "AAPL, MSFT, TSLA, SPY, QQQ").upper()
    if st.button("🔄 Get Midday Movers"):
        movers=[]
        for sym in [s.strip() for s in mover_list.split(",") if s.strip()]:
            tk = yf.Ticker(_map_symbol(sym))
            intraday = tk.history(period="1d", interval="5m")
            if intraday.empty:
                intraday = tk.history(period="2d", interval="5m")
                if not intraday.empty:
                    today = pd.Timestamp.utcnow().normalize()
                    intraday = intraday[intraday.index >= today]
            if intraday.empty:
                st.warning(f"No intraday data for {sym}. (Market closed or too early!)"); continue
            open_price = intraday["Open"].iloc[0]; last_price = intraday["Close"].iloc[-1]
            change_pct = (last_price - open_price) / open_price * 100
            movers.append({"Ticker": sym, "Open": open_price, "Current": last_price, "Change %": change_pct})
        if movers:
            df_m = pd.DataFrame(movers)
            df_m["Change %"] = pd.to_numeric(df_m["Change %"], errors="coerce")
            df_m = df_m.dropna(subset=["Change %"]).set_index("Ticker").sort_values("Change %", ascending=False)
            st.dataframe(df_m, use_container_width=True)
        else:
            st.info("No valid intraday data found for those tickers.")

    # ─────────── Portfolio Simulator ───────────
    st.markdown("---"); st.markdown("## 📊 Portfolio Simulator")
    st.info("Enter your positions in CSV: ticker,shares,cost_basis")
    holdings = st.text_area("e.g.\nAAPL,10,150\nMSFT,5,300", height=100)
    if st.button("▶️ Simulate Portfolio"):
        rows = [r.strip().split(",") for r in holdings.splitlines() if r.strip()]
        data=[]
        for idx, row in enumerate(rows, 1):
            if len(row) != 3:
                st.warning(f"Skipping invalid row {idx}: {row}"); continue
            ticker_, shares, cost = row
            tkr = _map_symbol(ticker_.upper().strip())
            try:
                s=float(shares); c=float(cost)
            except: 
                st.warning(f"Invalid numbers on row {idx}: {row}"); continue
            hist = yf.Ticker(tkr).history(period="1d")
            if hist.empty: 
                st.warning(f"No price for {tkr}"); continue
            price=float(hist["Close"].iloc[-1])
            invested=s*c; value=s*price; pnl=value-invested
            pnl_pct=(pnl/invested*100) if invested else np.nan
            df_raw=load_and_compute(tkr, ma_window, rsi_period, macd_fast, macd_slow, macd_signal,
                                    period_sel, interval_sel, include_bb)
            if df_raw.empty: comp_sugg="N/A"
            else:
                df_c=build_composite(df_raw, ma_window, rsi_period,
                                     use_weighted=use_weighted, w_ma=w_ma, w_rsi=w_rsi, w_macd=w_macd, w_bb=w_bb,
                                     include_bb=include_bb, threshold=comp_thr, allow_short=allow_short)
                score=float(df_c["Composite"].iloc[-1])
                comp_sugg = "🟢 BUY" if score>=comp_thr else ("🔴 SELL" if score<=-comp_thr else "🟡 HOLD")
            if pnl_pct > profit_target:     suggestion="🔴 SELL"
            elif pnl_pct < -loss_limit:     suggestion="🟢 BUY"
            else:                           suggestion=comp_sugg
            data.append({
                "Ticker":tkr,"Shares":s,"Cost Basis":c,"Price":price,
                "Market Value":value,"Invested":invested,"P/L":pnl,
                "P/L %":pnl_pct,"Composite Sig":comp_sugg,"Suggestion":suggestion
            })
        if data:
            df_port=pd.DataFrame(data).set_index("Ticker")
            st.dataframe(df_port, use_container_width=True)
            c1,c2,c3 = st.columns(3)
            c1.metric("Total Market Value", f"${df_port['Market Value'].sum():,.2f}")
            c2.metric("Total Invested",     f"${df_port['Invested'].sum():,.2f}")
            c3.metric("Total P/L",          f"${df_port['Market Value'].sum()-df_port['Invested'].sum():,.2f}")
            fig, ax=plt.subplots(figsize=(5,5))
            df_port["Market Value"].plot.pie(autopct="%.1f%%", ax=ax)
            ax.set_ylabel(""); ax.set_title("Portfolio Allocation")
            st.pyplot(fig)
        else:
            st.error("No valid holdings provided.")

    # ─────────── Hyperparameter Optimization ───────────
    st.markdown("---"); st.markdown("## 🛠️ Hyperparameter Optimization")
    ma_list  = st.sidebar.multiselect("MA windows",     [5,10,15,20,30], default=[ma_window], key="grid_ma")
    rsi_list = st.sidebar.multiselect("RSI lookbacks",  [7,14,21,28],   default=[rsi_period], key="grid_rsi")
    mf_list  = st.sidebar.multiselect("MACD fast spans",[8,12,16,20],   default=[macd_fast],  key="grid_mf")
    ms_list  = st.sidebar.multiselect("MACD slow spans",[20,26,32,40],  default=[macd_slow],  key="grid_ms")
    sig_list = st.sidebar.multiselect("MACD sig spans", [5,9,12,16],    default=[macd_signal],key="grid_sig")
    if st.button("🏃‍♂️ Run Grid Search"):
        if not ticker:
            st.error("Enter a ticker first."); st.stop()
        df_full=load_and_compute(ticker, ma_window, rsi_period, macd_fast, macd_slow, macd_signal,
                                 period_sel, interval_sel, include_bb)
        if df_full.empty:
            st.error(f"No data for '{ticker}'"); st.stop()
        results=[]
        with st.spinner("Testing parameter combos…"):
            for mw in ma_list:
                for rp_ in rsi_list:
                    for mf_ in mf_list:
                        for ms_ in ms_list:
                            for s_ in sig_list:
                                df_i=load_and_compute(ticker, mw, rp_, mf_, ms_, s_, period_sel, interval_sel, include_bb)
                                if df_i.empty: continue
                                df_ci=build_composite(df_i, mw, rp_,
                                                      use_weighted=use_weighted, w_ma=w_ma, w_rsi=w_rsi,
                                                      w_macd=w_macd, w_bb=w_bb, include_bb=include_bb,
                                                      threshold=comp_thr, allow_short=allow_short)
                                df_ci, md_i, sh_i, wr_i, trd_i, tim_i, cagr_i = backtest(
                                    df_ci, allow_short=allow_short, cost_bps=cost_bps,
                                    sl_atr_mult=sl_atr_mult, tp_atr_mult=tp_atr_mult)
                                results.append({
                                    "MA":mw,"RSI":rp_,"MACD Fast":mf_,"MACD Slow":ms_,"MACD Sig":s_,
                                    "Strategy %":(df_ci["CumStrat"].iloc[-1]-1)*100,
                                    "Sharpe":sh_i,"Max Drawdown":md_i,"Win Rate":wr_i,"CAGR %":cagr_i
                                })
        if results:
            df_grid=pd.DataFrame(results).sort_values("Strategy %",ascending=False).head(10)
            st.dataframe(df_grid, use_container_width=True)
            st.download_button("Download CSV", df_grid.to_csv(index=False), "grid.csv")
        else:
            st.error("No valid parameter combinations found.")

    # ─────────── Watchlist Summary ───────────
    st.markdown("---"); st.markdown("## ⏰ Watchlist Summary")
    watch = st.text_area("Enter tickers", "AAPL, MSFT, TSLA, SPY, QQQ").upper()
    if st.button("📬 Generate Watchlist Summary"):
        table=[]
        for t in [x.strip() for x in watch.split(",") if x.strip()]:
            df_t=load_and_compute(t, ma_window, rsi_period, macd_fast, macd_slow, macd_signal,
                                  period_sel, interval_sel, include_bb)
            if df_t.empty:
                table.append({"Ticker":t,"Composite":None,"Signal":"N/A"}); continue
            df_w=build_composite(df_t, ma_window, rsi_period,
                                 use_weighted=use_weighted, w_ma=w_ma, w_rsi=w_rsi, w_macd=w_macd, w_bb=w_bb,
                                 include_bb=include_bb, threshold=comp_thr, allow_short=allow_short)
            comp=float(df_w["Composite"].iloc[-1])
            sig = "🟢 BUY" if comp>=comp_thr else ("🔴 SELL" if comp<=-comp_thr else "🟡 HOLD")
            table.append({"Ticker":t,"Composite":comp,"Signal":sig})
        df_watch=pd.DataFrame(table).set_index("Ticker")
        st.dataframe(df_watch, use_container_width=True)
        for t in df_watch.index:
            df_t=load_and_compute(t, ma_window, rsi_period, macd_fast, macd_slow, macd_signal,
                                  period_sel, interval_sel, include_bb)
            if df_t.empty: continue
            df_c=build_composite(df_t, ma_window, rsi_period,
                                 use_weighted=use_weighted, w_ma=w_ma, w_rsi=w_rsi, w_macd=w_macd, w_bb=w_bb,
                                 include_bb=include_bb, threshold=comp_thr, allow_short=allow_short)
            last=df_c.iloc[-1]
            ma_s=int(last["MA_Signal"]); rsi_s=int(last["RSI_Signal"]); macd_s=int(last["MACD_Signal2"])
            rsi_v=float(last[f"RSI{rsi_period}"])
            ma_txt={1:f"Price ↑ above MA{ma_window}.",0:"No crossover.",-1:f"Price ↓ below MA."}[ma_s]
            rsi_txt={1:f"RSI ({rsi_v:.1f}) < 30 → oversold.",0:f"RSI neutral.",-1:f"RSI ({rsi_v:.1f}) > 70 → overbought."}[rsi_s]
            macd_txt={1:"MACD ↑ above signal.",0:"No crossover.",-1:"MACD ↓ below signal."}[macd_s]
            with st.expander(f"🔎 {t} Reasoning ({df_watch.loc[t,'Signal']})"):
                st.write(f"- **MA:**  {ma_txt}")
                st.write(f"- **RSI:** {rsi_txt}")
                st.write(f"- **MACD:** {macd_txt}")
                if include_bb and "BB_Signal" in df_c.columns:
                    bb_s=int(last["BB_Signal"])
                    bb_txt={1:"Under lower band.",0:"Inside bands.",-1:"Over upper band."}[bb_s]
                    st.write(f"- **BB:** {bb_txt}")
                st.write(f"- **Composite Score:** {df_watch.loc[t,'Composite']:.2f}")
