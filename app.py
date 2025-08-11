# app.py — QuantaraX Composite Signals BETA v3+
# -------------------------------------------------------------
# pip install: streamlit yfinance pandas numpy matplotlib feedparser vaderSentiment

import math
from typing import List

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import feedparser

# ───────────────────────────── Page Setup ─────────────────────────────
st.set_page_config(page_title="QuantaraX Composite Signals BETA v3+", layout="wide")
analyzer = SentimentIntensityAnalyzer()

# ───────────────────────────── Mappings ─────────────────────────────
rec_map = {1: "🟢 BUY", 0: "🟡 HOLD", -1: "🔴 SELL"}

# ───────────────────────────── Tabs ─────────────────────────────
tab_engine, tab_help = st.tabs(["🚀 Engine", "❓ How It Works"])

# ───────────────────────────── Help Tab ─────────────────────────────
with tab_help:
    st.header("How QuantaraX Works")
    st.markdown(r"""
Welcome to **QuantaraX** — now with **weighted composites**, **ATR risk controls**, **shorting + trading costs**, **interval/period** controls, **multi-timeframe confirmation**, **walk-forward optimization**, and a **risk-parity** portfolio allocator.

### Quick guide
- **Composite (v2+)**: MA, RSI, MACD (+ optional Bollinger) → weighted sum. Enter when composite ≥ threshold (and ≤ −threshold for shorts).
- **Risk overlay**: ATR stop/target, trading costs (bps), optional volatility targeting.
- **Walk-forward**: grid search on rolling in-sample, stitch out-of-sample equity.
- **MTF**: compare Daily vs Hourly composite direction.
- **Risk parity**: equalize risk contributions across tickers.

> Use the sidebar to tweak indicators, weights, risk, and data period/interval.
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
    include_bb   = st.sidebar.toggle("Include Bollinger Bands", value=True)
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
    vol_target  = st.sidebar.slider("Vol targeting (annual, e.g. 0.00–0.50)", 0.0, 0.5, 0.0, 0.05)

    st.sidebar.subheader("Data")
    period_sel   = st.sidebar.selectbox("History", ["6mo","1y","2y","5y"], index=1)
    interval_sel = st.sidebar.selectbox("Interval", ["1d","1h"], index=0)

    st.sidebar.subheader("Portfolio Guardrails")
    profit_target = st.sidebar.slider("Profit target (%)", 1, 100, 10, help="If unrealized P/L% exceeds this → SELL")
    loss_limit    = st.sidebar.slider("Loss limit (%)",  1, 100, 5,  help="If unrealized P/L% falls below –this → BUY")

    st.title("🚀 QuantaraX — Composite Signal Engine (v3+)")

    # ─────────── Helpers ───────────
    def _map_symbol(sym: str) -> str:
        s = sym.strip().upper()
        if "/" in s:  # e.g., BTC/USDT → BTC-USD
            base, quote = s.split("/")
            quote = "USD" if quote in ("USDT","USD") else quote
            return f"{base}-{quote}"
        return s

    @st.cache_data(show_spinner=False, ttl=900)
    def load_prices(symbol: str, period: str, interval: str) -> pd.DataFrame:
        sym = _map_symbol(symbol)
        df = yf.download(sym, period=period, interval=interval, auto_adjust=False, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        return df.dropna()

    def compute_indicators(df: pd.DataFrame, ma_w: int, rsi_p: int, mf: int, ms: int, sig: int,
                           use_bb: bool = True) -> pd.DataFrame:
        d = df.copy()
        if d.empty or not set(["Open","High","Low","Close"]).issubset(d.columns):
            return pd.DataFrame()

        # MA
        d[f"MA{ma_w}"] = d["Close"].rolling(ma_w).mean()
        # RSI
        chg = d["Close"].diff()
        up, dn = chg.clip(lower=0), -chg.clip(upper=0)
        ema_up   = up.ewm(com=rsi_p-1, adjust=False).mean()
        ema_down = dn.ewm(com=rsi_p-1, adjust=False).mean()
        rs = ema_up / ema_down.replace(0, np.nan)
        d[f"RSI{rsi_p}"] = 100 - 100 / (1 + rs)
        # MACD
        ema_f = d["Close"].ewm(span=mf, adjust=False).mean()
        ema_s = d["Close"].ewm(span=ms, adjust=False).mean()
        macd_line = ema_f - ema_s
        d["MACD"] = macd_line
        d["MACD_Signal"] = macd_line.ewm(span=sig, adjust=False).mean()
        # ATR
        pc = d["Close"].shift(1)
        tr = pd.concat([(d["High"]-d["Low"]).abs(), (d["High"]-pc).abs(), (d["Low"]-pc).abs()], axis=1).max(axis=1)
        d["ATR"] = tr.ewm(alpha=1/14, adjust=False).mean()
        # Bollinger
        if use_bb:
            w = 20; k = 2.0
            mid = d["Close"].rolling(w).mean()
            sd  = d["Close"].rolling(w).std(ddof=0)
            d["BB_M"], d["BB_U"], d["BB_L"] = mid, mid + k*sd, mid - k*sd
        return d.dropna()

    def build_composite(df: pd.DataFrame, ma_w: int, rsi_p: int,
                        *, use_weighted=True, w_ma=1.0, w_rsi=1.0, w_macd=1.0, w_bb=0.5,
                        include_bb=True, threshold=0.0, allow_short=False) -> pd.DataFrame:
        if df.empty:
            return df.copy()

        d = df.copy()
        n = len(d)
        close = d["Close"].to_numpy()
        ma    = d[f"MA{ma_w}"].to_numpy()
        rsi   = d[f"RSI{rsi_p}"].to_numpy()
        macd  = d["MACD"].to_numpy()
        sigl  = d["MACD_Signal"].to_numpy()

        ma_sig = np.zeros(n, int)
        rsi_sig = np.zeros(n, int)
        macd_sig2 = np.zeros(n, int)
        bb_sig = np.zeros(n, int)

        for i in range(1, n):
            if close[i-1] < ma[i-1] and close[i] > ma[i]:   ma_sig[i] = 1
            elif close[i-1] > ma[i-1] and close[i] < ma[i]: ma_sig[i] = -1
            if rsi[i] < 30:   rsi_sig[i] = 1
            elif rsi[i] > 70: rsi_sig[i] = -1
            if macd[i-1] < sigl[i-1] and macd[i] > sigl[i]:   macd_sig2[i] = 1
            elif macd[i-1] > sigl[i-1] and macd[i] < sigl[i]: macd_sig2[i] = -1
            if include_bb and {"BB_U","BB_L"}.issubset(d.columns):
                if close[i] < d["BB_L"].iloc[i]: bb_sig[i] = 1
                elif close[i] > d["BB_U"].iloc[i]: bb_sig[i] = -1

        comp = (w_ma*ma_sig + w_rsi*rsi_sig + w_macd*macd_sig2 + (w_bb*bb_sig if include_bb else 0)) if use_weighted \
               else (ma_sig + rsi_sig + macd_sig2)

        if allow_short:
            trade = np.where(comp >= threshold, 1, np.where(comp <= -threshold, -1, 0))
        else:
            trade = np.where(comp >= threshold, 1, 0)

        d["MA_Signal"], d["RSI_Signal"], d["MACD_Signal2"] = ma_sig, rsi_sig, macd_sig2
        if include_bb: d["BB_Signal"] = bb_sig
        d["Composite"] = comp.astype(float)
        d["Trade"] = trade.astype(int)
        return d

    # ───── Robust backtest (safe against empties / short series) ─────
    def backtest(df: pd.DataFrame, *, allow_short=False, cost_bps=0.0,
                 sl_atr_mult=0.0, tp_atr_mult=0.0, vol_target=0.0, interval="1d"):
        d = df.copy()

        if d.empty or "Close" not in d:
            sk = d.copy()
            for col in ["Return","Position","StratRet","CumBH","CumStrat"]:
                sk[col] = 0.0
            sk["CumBH"] = 1.0
            sk["CumStrat"] = 1.0
            return sk, 0.0, np.nan, np.nan, 0, 0.0, np.nan

        d["Return"] = d["Close"].pct_change().fillna(0.0)

        # Base position from trade signal
        if allow_short:
            d["Position"] = d.get("Trade", 0).shift(1).fillna(0).clip(-1,1)
            base_ret = np.where(d["Position"]>=0, d["Return"], -d["Return"])
        else:
            d["Position"] = d.get("Trade", 0).shift(1).fillna(0).clip(0,1)
            base_ret = d["Position"] * d["Return"]

        # Vol targeting (rolling 20 bars)
        if vol_target and vol_target > 0:
            look = 20
            daily_vol = d["Return"].rolling(look).std(ddof=0)
            ann = 252 if interval == "1d" else 252*6
            realized = daily_vol * math.sqrt(ann)
            scale = (vol_target / realized).clip(0, 3.0).fillna(0.0)  # cap leverage
            base_ret = base_ret * scale

        # Costs on trades
        cost = cost_bps/10000.0
        pos_change = d["Position"].diff().fillna(0).abs()
        tcost = -2.0*cost*(pos_change > 0).astype(float)  # open+close
        d["StratRet"] = pd.Series(base_ret, index=d.index).fillna(0.0) + tcost

        # ATR exits → flatten next bar
        if (sl_atr_mult>0 or tp_atr_mult>0) and "ATR" in d.columns:
            flat = np.zeros(len(d), dtype=int)
            entry = np.nan
            for i in range(len(d)):
                p, c = d["Position"].iat[i], d["Close"].iat[i]
                a = d["ATR"].iat[i] if "ATR" in d.columns else np.nan
                if p != 0 and np.isnan(entry): entry = c
                if p == 0: entry = np.nan
                if p != 0 and not np.isnan(a):
                    if p == 1 and (c <= entry - sl_atr_mult*a or c >= entry + tp_atr_mult*a):
                        flat[i] = 1; entry = np.nan
                    if p == -1 and (c >= entry + sl_atr_mult*a or c <= entry - tp_atr_mult*a):
                        flat[i] = 1; entry = np.nan
            if flat.any(): d.loc[flat==1, "Position"] = 0

        # Cum returns (robust)
        d["CumBH"]    = (1 + d["Return"]).replace([np.inf, -np.inf], np.nan).fillna(0.0).add(1).cumprod()
        d["CumStrat"] = (1 + d["StratRet"]).replace([np.inf, -np.inf], np.nan).fillna(0.0).add(1).cumprod()

        # Stats (robust)
        if d["CumStrat"].notna().any():
            dd = d["CumStrat"]/d["CumStrat"].cummax() - 1
            max_dd = float(dd.min()*100)
            last_cum = float(d["CumStrat"].dropna().iloc[-1])
        else:
            max_dd = 0.0
            last_cum = 1.0

        ann = 252 if interval == "1d" else 252*6
        mean_ann = float(d["StratRet"].mean() * ann)
        vol_ann  = float(d["StratRet"].std(ddof=0) * math.sqrt(ann))
        sharpe   = (mean_ann / vol_ann) if vol_ann > 0 else np.nan
        win_rt   = float((d["StratRet"] > 0).mean() * 100)
        trades   = int((pos_change > 0).sum())
        tim      = float((d["Position"] != 0).mean() * 100)

        n_eff = int(d["StratRet"].notna().sum())
        cagr = ((last_cum ** (ann / max(n_eff, 1))) - 1) * 100 if n_eff > 0 else np.nan

        return d, max_dd, sharpe, win_rt, trades, tim, cagr

    # ─────────────────── Single‐Ticker Backtest ───────────────────
    st.markdown("## Single‐Ticker Backtest")
    ticker = st.text_input("Ticker (e.g. AAPL or BTC/USDT)", "AAPL").upper()

    if ticker:
        h = yf.download(_map_symbol(ticker), period="1d", progress=False)
        if not h.empty and "Close" in h:
            st.subheader(f"💲 Live Price: ${float(h['Close'].iloc[-1]):.2f}")

        # Dual-source News Feed
        raw_news = getattr(yf.Ticker(_map_symbol(ticker)), "news", []) or []
        shown = 0
        if raw_news:
            st.markdown("### 📰 Recent News & Sentiment (YFinance)")
            for art in raw_news:
                t_ = art.get("title",""); l_ = art.get("link","")
                if not (t_ and l_): continue
                txt = art.get("summary", t_)
                score = analyzer.polarity_scores(txt)["compound"]
                emoji = "🔺" if score>0.1 else ("🔻" if score<-0.1 else "➖")
                st.markdown(f"- [{t_}]({l_}) {emoji}")
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
        px = load_prices(ticker, period_sel, interval_sel)
        if px.empty:
            st.error(f"No data for '{ticker}'"); st.stop()

        df_raw = compute_indicators(px, ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=include_bb)
        if df_raw.empty:
            st.error("Not enough data after indicators (try a longer period or smaller windows)."); st.stop()

        df_sig = build_composite(df_raw, ma_window, rsi_period,
                                 use_weighted=use_weighted, w_ma=w_ma, w_rsi=w_rsi, w_macd=w_macd, w_bb=w_bb,
                                 include_bb=include_bb, threshold=comp_thr, allow_short=allow_short)
        if df_sig.empty:
            st.error("Composite could not be built (insufficient rows)."); st.stop()

        df_c, max_dd, sharpe, win_rt, trades, tim, cagr = backtest(
            df_sig, allow_short=allow_short, cost_bps=cost_bps,
            sl_atr_mult=sl_atr_mult, tp_atr_mult=tp_atr_mult, vol_target=vol_target, interval=interval_sel
        )

        # Safe last trade
        last_trade = int(df_sig["Trade"].tail(1).iloc[0]) if "Trade" in df_sig.columns and not df_sig.empty else 0
        rec = rec_map.get(1 if last_trade>0 else (-1 if last_trade<0 else 0), "🟡 HOLD")
        st.success(f"**{ticker}**: {rec}")

        # Explanation (safe)
        last_row = df_sig.tail(1)
        if not last_row.empty:
            last = last_row.iloc[0]
            ma_s  = int(last.get("MA_Signal", 0))
            rsi_s = int(last.get("RSI_Signal", 0))
            macd_s= int(last.get("MACD_Signal2", 0))
            rsi_v = float(last.get(f"RSI{rsi_period}", np.nan))
            ma_txt  = {1:f"Price ↑ crossed above MA{ma_window}.", 0:"No MA crossover.", -1:f"Price ↓ crossed below MA{ma_window}."}.get(ma_s, "No MA crossover.")
            rsi_txt = "RSI data unavailable." if np.isnan(rsi_v) else {
                1:f"RSI ({rsi_v:.1f}) < 30 → oversold.",
                0:f"RSI ({rsi_v:.1f}) neutral.",
               -1:f"RSI ({rsi_v:.1f}) > 70 → overbought."
            }.get(rsi_s, f"RSI ({rsi_v:.1f}) neutral.")
            macd_txt= {1:"MACD ↑ crossed above signal.", 0:"No MACD crossover.", -1:"MACD ↓ crossed below signal."}.get(macd_s, "No MACD crossover.")
            with st.expander("🔎 Why This Signal?"):
                st.write(f"- **MA:**  {ma_txt}")
                st.write(f"- **RSI:** {rsi_txt}")
                st.write(f"- **MACD:** {macd_txt}")
                if include_bb and "BB_Signal" in df_sig.columns:
                    bb_s = int(last.get("BB_Signal", 0))
                    bb_txt = {1:"Close under lower band (mean-revert long).",0:"Inside bands.",-1:"Close over upper band (mean-revert short)."}[bb_s]
                    st.write(f"- **BB:** {bb_txt}")
                st.write(f"- **Composite (weighted):** {float(last.get('Composite', 0)):.2f}  (threshold={comp_thr:.1f})")

        # Metrics (safe)
        bh_last  = float(df_c["CumBH"].tail(1).iloc[0])  if "CumBH" in df_c and not df_c["CumBH"].empty else 1.0
        strat_last = float(df_c["CumStrat"].tail(1).iloc[0]) if "CumStrat" in df_c and not df_c["CumStrat"].empty else 1.0
        colA, colB, colC, colD, colE, colF = st.columns(6)
        colA.metric("CAGR", f"{(cagr if not np.isnan(cagr) else 0):.2f}%")
        colB.metric("Sharpe", f"{(sharpe if not np.isnan(sharpe) else 0):.2f}")
        colC.metric("Max DD", f"{max_dd:.2f}%")
        colD.metric("Win Rate", f"{win_rt:.1f}%")
        colE.metric("Trades", f"{trades}")
        colF.metric("Time in Mkt", f"{tim:.1f}%")

        st.markdown(f"""
- **Buy & Hold:**    {(bh_last-1)*100:.2f}%  
- **Strategy:**      {(strat_last-1)*100:.2f}%  
""")

        # Plots
        idx = df_c.index
        fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(11,12), sharex=True)
        ax1.plot(idx, df_c["Close"], label="Close")
        if f"MA{ma_window}" in df_c: ax1.plot(idx, df_c[f"MA{ma_window}"], label=f"MA{ma_window}")
        if include_bb and {"BB_U","BB_L"}.issubset(df_c.columns):
            ax1.plot(idx, df_c["BB_U"], label="BB Upper"); ax1.plot(idx, df_c["BB_L"], label="BB Lower")
        ax1.legend(); ax1.set_title("Price & Indicators")
        if "Composite" in df_c:
            ax2.bar(idx, df_c["Composite"]); ax2.set_title("Composite (weighted)")
        else:
            ax2.set_title("Composite (no data)")
        ax3.plot(idx, df_c["CumBH"], ":", label="BH")
        ax3.plot(idx, df_c["CumStrat"], "-", label="Strat"); ax3.legend(); ax3.set_title("Equity")
        plt.xticks(rotation=45); plt.tight_layout()
        st.pyplot(fig)

    # ─────────── Batch Backtest ───────────
    st.markdown("---")
    st.markdown("## Batch Backtest")
    batch = st.text_area("Tickers (comma-separated)", "AAPL, MSFT, TSLA, SPY, QQQ").upper()
    if st.button("▶️ Run Batch Backtest"):
        perf=[]
        for t in [x.strip() for x in batch.split(",") if x.strip()]:
            px = load_prices(t, period_sel, interval_sel)
            if px.empty: continue
            df_t = compute_indicators(px, ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=include_bb)
            if df_t.empty: continue
            df_tc = build_composite(df_t, ma_window, rsi_period,
                                    use_weighted=use_weighted, w_ma=w_ma, w_rsi=w_rsi, w_macd=w_macd, w_bb=w_bb,
                                    include_bb=include_bb, threshold=comp_thr, allow_short=allow_short)
            if df_tc.empty: continue
            bt, md, sh, wr, trd, tim, cagr = backtest(df_tc, allow_short=allow_short, cost_bps=cost_bps,
                                                      sl_atr_mult=sl_atr_mult, tp_atr_mult=tp_atr_mult,
                                                      vol_target=vol_target, interval=interval_sel)
            comp_last = float(bt["Composite"].tail(1).iloc[0]) if "Composite" in bt and not bt["Composite"].empty else 0.0
            bh_last = float(bt["CumBH"].tail(1).iloc[0]) if "CumBH" in bt and not bt["CumBH"].empty else 1.0
            strat_last = float(bt["CumStrat"].tail(1).iloc[0]) if "CumStrat" in bt and not bt["CumStrat"].empty else 1.0
            perf.append({
                "Ticker":t,
                "Composite":comp_last,
                "Signal": rec_map.get(int(np.sign(comp_last)), "🟡 HOLD"),
                "Buy & Hold %": (bh_last-1)*100,
                "Strategy %":   (strat_last-1)*100,
                "Sharpe":       sh,
                "Max Drawdown": md,
                "Win Rate":     wr,
                "Trades":       trd,
                "Time in Mkt %": tim,
                "CAGR %":       cagr
            })
        if perf:
            df_perf = pd.DataFrame(perf).set_index("Ticker").sort_values("Strategy %", ascending=False)
            st.dataframe(df_perf, use_container_width=True)
            st.download_button("Download CSV", df_perf.to_csv(), "batch.csv")
        else:
            st.error("No valid data for batch tickers.")

    # ─────────── Midday Movers ───────────
    st.markdown("---")
    st.markdown("## 🌤️ Midday Movers (Intraday % Change)")
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
            df_m = pd.DataFrame(movers).dropna(subset=["Change %"]).set_index("Ticker").sort_values("Change %", ascending=False)
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

            # Composite suggestion (safe)
            px = load_prices(tkr, period_sel, interval_sel)
            if px.empty:
                comp_sugg="N/A"
            else:
                df_i = compute_indicators(px, ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=include_bb)
                if df_i.empty:
                    comp_sugg="N/A"
                else:
                    df_csig = build_composite(df_i, ma_window, rsi_period,
                                              use_weighted=use_weighted, w_ma=w_ma, w_rsi=w_rsi, w_macd=w_macd, w_bb=w_bb,
                                              include_bb=include_bb, threshold=comp_thr, allow_short=allow_short)
                    if df_csig.empty:
                        comp_sugg="N/A"
                    else:
                        score = float(df_csig["Composite"].tail(1).iloc[0]) if "Composite" in df_csig else 0.0
                        comp_sugg = "🟢 BUY" if score>=comp_thr else ("🔴 SELL" if score<=-comp_thr else "🟡 HOLD")

            # Guardrails override
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
    st.markdown("---")
    st.markdown("## 🛠️ Hyperparameter Optimization")
    ma_list  = st.sidebar.multiselect("MA windows",     [5,10,15,20,30], default=[ma_window], key="grid_ma")
    rsi_list = st.sidebar.multiselect("RSI lookbacks",  [7,14,21,28],   default=[rsi_period], key="grid_rsi")
    mf_list  = st.sidebar.multiselect("MACD fast spans",[8,12,16,20],   default=[macd_fast],  key="grid_mf")
    ms_list  = st.sidebar.multiselect("MACD slow spans",[20,26,32,40],  default=[macd_slow],  key="grid_ms")
    sig_list = st.sidebar.multiselect("MACD sig spans", [5,9,12,16],    default=[macd_signal],key="grid_sig")
    if st.button("🏃‍♂️ Run Grid Search"):
        if not ticker:
            st.error("Enter a ticker first."); st.stop()
        px = load_prices(ticker, period_sel, interval_sel)
        if px.empty:
            st.error(f"No data for '{ticker}'"); st.stop()
        results=[]
        with st.spinner("Testing parameter combos…"):
            for mw in ma_list:
                for rp_ in rsi_list:
                    for mf_ in mf_list:
                        for ms_ in ms_list:
                            for s_ in sig_list:
                                dfi = compute_indicators(px, mw, rp_, mf_, ms_, s_, use_bb=include_bb)
                                if dfi.empty: continue
                                sigs = build_composite(dfi, mw, rp_,
                                                       use_weighted=use_weighted, w_ma=w_ma, w_rsi=w_rsi,
                                                       w_macd=w_macd, w_bb=w_bb, include_bb=include_bb,
                                                       threshold=comp_thr, allow_short=allow_short)
                                bt, md_i, sh_i, wr_i, trd_i, tim_i, cagr_i = backtest(
                                    sigs, allow_short=allow_short, cost_bps=cost_bps,
                                    sl_atr_mult=sl_atr_mult, tp_atr_mult=tp_atr_mult,
                                    vol_target=vol_target, interval=interval_sel
                                )
                                strat_last = float(bt["CumStrat"].tail(1).iloc[0]) if "CumStrat" in bt and not bt["CumStrat"].empty else 1.0
                                results.append({
                                    "MA":mw,"RSI":rp_,"MACD Fast":mf_,"MACD Slow":ms_,"MACD Sig":s_,
                                    "Strategy %":(strat_last-1)*100,
                                    "Sharpe":sh_i,"Max Drawdown":md_i,"Win Rate":wr_i,"CAGR %":cagr_i
                                })
        if results:
            df_grid=pd.DataFrame(results).sort_values("Strategy %",ascending=False).head(10)
            st.dataframe(df_grid, use_container_width=True)
            st.download_button("Download CSV", df_grid.to_csv(index=False), "grid.csv")
        else:
            st.error("No valid parameter combinations found.")

    # ─────────── Watchlist Summary (guarded) ───────────
    st.markdown("---")
    st.markdown("## ⏰ Watchlist Summary")
    watch = st.text_area("Enter tickers", "AAPL, MSFT, TSLA, SPY, QQQ").upper()
    if st.button("📬 Generate Watchlist Summary"):
        table=[]
        for t in [x.strip() for x in watch.split(",") if x.strip()]:
            px = load_prices(t, period_sel, interval_sel)
            if px.empty:
                table.append({"Ticker":t,"Composite":None,"Signal":"N/A"}); continue
            dft = compute_indicators(px, ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=include_bb)
            if dft.empty:
                table.append({"Ticker":t,"Composite":None,"Signal":"N/A"}); continue
            sigs= build_composite(dft, ma_window, rsi_period,
                                  use_weighted=use_weighted, w_ma=w_ma, w_rsi=w_rsi, w_macd=w_macd, w_bb=w_bb,
                                  include_bb=include_bb, threshold=comp_thr, allow_short=allow_short)
            last = sigs.tail(1)
            if last.empty or "Composite" not in last.columns:
                table.append({"Ticker":t,"Composite":None,"Signal":"N/A"}); continue
            comp=float(last["Composite"].iloc[0])
            sig = "🟢 BUY" if comp>=comp_thr else ("🔴 SELL" if comp<=-comp_thr else "🟡 HOLD")
            table.append({"Ticker":t,"Composite":comp,"Signal":sig})
        df_watch=pd.DataFrame(table).set_index("Ticker")
        st.dataframe(df_watch, use_container_width=True)

        # Reasoning (guarded)
        for t in df_watch.index:
            px = load_prices(t, period_sel, interval_sel)
            if px.empty: continue
            dft = compute_indicators(px, ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=include_bb)
            if dft.empty: continue
            sigs= build_composite(dft, ma_window, rsi_period,
                                  use_weighted=use_weighted, w_ma=w_ma, w_rsi=w_rsi, w_macd=w_macd, w_bb=w_bb,
                                  include_bb=include_bb, threshold=comp_thr, allow_short=allow_short)
            last=sigs.tail(1)
            if last.empty: continue
            row = last.iloc[0]
            ma_s=int(row.get("MA_Signal",0)); rsi_s=int(row.get("RSI_Signal",0)); macd_s=int(row.get("MACD_Signal2",0))
            rsi_v=float(row.get(f"RSI{rsi_period}", np.nan))
            ma_txt={1:f"Price ↑ above MA{ma_window}.",0:"No crossover.",-1:f"Price ↓ below MA."}.get(ma_s,"No crossover.")
            if np.isnan(rsi_v):
                rsi_txt="RSI data unavailable."
            else:
                rsi_txt={1:f"RSI ({rsi_v:.1f}) < 30 → oversold.",0:f"RSI ({rsi_v:.1f}) neutral.",-1:f"RSI ({rsi_v:.1f}) > 70 → overbought."}.get(rsi_s,f"RSI ({rsi_v:.1f}) neutral.")
            macd_txt={1:"MACD ↑ above signal.",0:"No crossover.",-1:"MACD ↓ below signal."}.get(macd_s,"No crossover.")
            with st.expander(f"🔎 {t} Reasoning ({df_watch.loc[t,'Signal']})"):
                st.write(f"- **MA:**  {ma_txt}")
                st.write(f"- **RSI:** {rsi_txt}")
                st.write(f"- **MACD:** {macd_txt}")
                if include_bb and "BB_Signal" in sigs.columns:
                    bb_s=int(row.get("BB_Signal",0))
                    bb_txt={1:"Under lower band.",0:"Inside bands.",-1:"Over upper band."}.get(bb_s,"Inside bands.")
                    st.write(f"- **BB:** {bb_txt}")
                st.write(f"- **Composite Score:** {df_watch.loc[t,'Composite']}")

    # ─────────── Multi-Timeframe Confirmation ───────────
    st.markdown("---")
    st.markdown("## ⏱️ Multi-Timeframe Confirmation")
    mtf_symbol = st.text_input("Symbol (for MTF check)", value=ticker or "AAPL")
    if st.button("🔍 Check MTF"):
        try:
            d1 = compute_indicators(load_prices(mtf_symbol, "1y", "1d"), ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=True)
            dH = compute_indicators(load_prices(mtf_symbol, "30d", "1h"), ma_window, rsi_period, macd_fast, macd_slow, macd_signal, use_bb=True)
            if d1.empty or dH.empty:
                st.warning("Not enough data for one or both timeframes."); 
            else:
                c1 = build_composite(d1, ma_window, rsi_period, use_weighted=True, w_ma=1.0, w_rsi=1.0, w_macd=1.0, w_bb=0.5, include_bb=True, threshold=1.0)
                cH = build_composite(dH, ma_window, rsi_period, use_weighted=True, w_ma=1.0, w_rsi=1.0, w_macd=1.0, w_bb=0.5, include_bb=True, threshold=1.0)
                daily  = float(c1["Composite"].tail(1).iloc[0]) if not c1.empty else 0.0
                hourly = float(cH["Composite"].tail(1).iloc[0]) if not cH.empty else 0.0
                agree = int(np.sign(daily) == np.sign(hourly))
                st.write(f"**Daily composite:** {daily:.2f}")
                st.write(f"**Hourly composite:** {hourly:.2f}")
                st.success("✅ Signals agree") if agree else st.warning("⚠️ Signals disagree")
        except Exception as e:
            st.error(f"MTF error: {e}")

    # ─────────── Walk-Forward Optimization (simple OOS) ───────────
    st.markdown("---")
    st.markdown("## 🧪 Walk-Forward Optimization (Out-of-Sample)")
    wf_symbol = st.text_input("WFO symbol", value=ticker or "AAPL")
    c1, c2 = st.columns(2)
    with c1:
        ins_bars = st.number_input("In-sample bars", 60, 252*3, 126, 1)
        oos_bars = st.number_input("OOS bars", 20, 252, 63, 1)
    with c2:
        w_thr = st.slider("Composite trigger (WFO)", 0.0, 3.0, 1.0, 0.1)
        wf_allow_short = st.toggle("Allow shorts (WFO)", value=False)
    run_wfo = st.button("🏃 Run Walk-Forward")

    def walk_forward_optimize(symbol: str,
                              ma_list: List[int], rsi_list: List[int],
                              mf_list: List[int], ms_list: List[int], sig_list: List[int],
                              insample_bars: int, oos_bars: int,
                              w_ma=1.0, w_rsi=1.0, w_macd=1.0, w_bb=0.5, threshold=1.0,
                              allow_short=False, cost_bps=5.0):
        px = load_prices(symbol, "2y", "1d")
        if px.empty: return pd.DataFrame(), pd.DataFrame()
        oos_curves = []; summary = []
        start = 200; i = start
        while i + insample_bars + oos_bars <= len(px):
            ins = px.iloc[i : i+insample_bars]
            oos = px.iloc[i+insample_bars : i+insample_bars+oos_bars]
            best = None; best_score = -1e9
            for mw in ma_list:
                for rp in rsi_list:
                    for mf in mf_list:
                        for ms in ms_list:
                            for s in sig_list:
                                ins_ind = compute_indicators(ins, mw, rp, mf, ms, s, use_bb=True)
                                if ins_ind.empty: continue
                                ins_sig = build_composite(ins_ind, mw, rp, use_weighted=True, w_ma=w_ma, w_rsi=w_rsi,
                                                          w_macd=w_macd, w_bb=w_bb, include_bb=True, threshold=threshold,
                                                          allow_short=allow_short)
                                ins_bt, md, sh, wr, tr, ti, cg = backtest(ins_sig, allow_short=allow_short, cost_bps=cost_bps)
                                ins_last = float(ins_bt["CumStrat"].tail(1).iloc[0]) if "CumStrat" in ins_bt and not ins_bt["CumStrat"].empty else 1.0
                                perf = (ins_last-1)*100
                                score = perf - abs(md)
                                if score > best_score:
                                    best_score = score
                                    best = (mw, rp, mf, ms, s, sh, perf, md)
            if best is None:
                i += oos_bars; continue
            mw, rp, mf, ms, s, sh, perf, mdd = best
            oos_ind = compute_indicators(oos, mw, rp, mf, ms, s, use_bb=True)
            if oos_ind.empty:
                i += oos_bars; continue
            oos_sig = build_composite(oos_ind, mw, rp, use_weighted=True, w_ma=w_ma, w_rsi=w_rsi, w_macd=w_macd,
                                      w_bb=w_bb, include_bb=True, threshold=threshold, allow_short=allow_short)
            oos_bt, mo_dd, mo_sh, *_ = backtest(oos_sig, allow_short=allow_short, cost_bps=cost_bps)
            eq_seg = oos_bt[["CumStrat"]].rename(columns={"CumStrat":"Equity"})
            if not eq_seg.empty: oos_curves.append(eq_seg)
            summary.append({
                "Window": f"{oos.index[0].date()} → {oos.index[-1].date()}",
                "MA": mw, "RSI": rp, "MACDf": mf, "MACDs": ms, "SIG": s,
                "OOS %": (eq_seg["Equity"].iloc[-1]-1)*100 if not eq_seg.empty else 0.0,
                "OOS Sharpe": mo_sh, "OOS MaxDD%": mo_dd
            })
            i += oos_bars
        eq = pd.concat(oos_curves, axis=0) if oos_curves else pd.DataFrame()
        sm = pd.DataFrame(summary)
        return eq, sm

    if run_wfo:
        try:
            eq, sm = walk_forward_optimize(
                wf_symbol,
                ma_list=[ma_window, max(5, ma_window-5), min(60, ma_window+5)],
                rsi_list=[rsi_period, max(5, rsi_period-7), min(30, rsi_period+7)],
                mf_list=[macd_fast, max(5, macd_fast-4), min(20, macd_fast+4)],
                ms_list=[macd_slow, max(20, macd_slow-6), min(50, macd_slow+6)],
                sig_list=[macd_signal, max(5, macd_signal-4), min(20, macd_signal+4)],
                insample_bars=int(ins_bars),
                oos_bars=int(oos_bars),
                w_ma=1.0, w_rsi=1.0, w_macd=1.0, w_bb=0.5,
                threshold=w_thr, allow_short=wf_allow_short, cost_bps=5.0
            )
            if not sm.empty:
                st.dataframe(sm, use_container_width=True)
            if not eq.empty:
                fig, ax = plt.subplots(figsize=(10,3))
                ax.plot(eq.index, eq["Equity"]); ax.set_title("Walk-Forward OOS Equity (stitched)")
                st.pyplot(fig)
            else:
                st.info("WFO produced no OOS segments (not enough data).")
        except Exception as e:
            st.error(f"WFO error: {e}")

    # ─────────── Portfolio Optimizer — Risk Parity ───────────
    st.markdown("---")
    st.markdown("## ⚖️ Portfolio Optimizer — Risk Parity")
    opt_tickers = st.text_input("Tickers to optimize (comma-sep)", "AAPL, MSFT, TSLA, SPY, QQQ").upper()
    if st.button("🧮 Optimize (Risk Parity)"):
        try:
            tickers = [t.strip() for t in opt_tickers.split(",") if t.strip()]
            rets = []; valid = []
            for t in tickers:
                px = load_prices(t, "1y", "1d")
                if px.empty: continue
                valid.append(t)
                rets.append(px["Close"].pct_change().dropna())
            if not rets:
                st.error("No valid tickers/data."); st.stop()
            R = pd.concat(rets, axis=1); R.columns = valid
            cov = R.cov()
            n = len(valid); w = np.ones(n)/n
            for _ in range(500):
                mrc = cov @ w
                rc  = w * mrc
                target = rc.mean()
                grad = rc - target
                w = np.clip(w - 0.05*grad, 0, None)
                s = w.sum()
                w = w / s if s > 1e-12 else np.ones(n)/n
                if np.linalg.norm(grad) < 1e-6:
                    break
            weights = pd.Series(w, index=valid, name="Weight")
            st.dataframe(weights.to_frame().T, use_container_width=True)
            fig, ax = plt.subplots(figsize=(5,5))
            weights.plot.pie(autopct="%.1f%%", ax=ax)
            ax.set_ylabel(""); ax.set_title("Risk-Parity Weights")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Optimizer error: {e}")
