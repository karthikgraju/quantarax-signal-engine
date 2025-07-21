import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

# ───────────────────────────── Page Setup ─────────────────────────────
st.set_page_config(page_title="QuantaraX Composite Signals", layout="centered")

# ───────────────────────────── Tabs ─────────────────────────────
tab_engine, tab_help = st.tabs(["🚀 Engine", "❓ How It Works"])

with tab_help:
    st.header("How QuantaraX Works")
    st.markdown("""
**QuantaraX** computes three classic technical indicators and then “votes” on them:

1. **Moving Average Crossover**  
   - Computes a simple MA over the last *N* days (adjustable via **MA window**).  
   - Bullish when price crosses *above* the MA; bearish when it crosses *below*.

2. **Relative Strength Index (RSI)**  
   - A momentum oscillator over a lookback period (adjustable via **RSI lookback**).  
   - *Oversold* (RSI < 30) → bullish; *Overbought* (RSI > 70) → bearish.

3. **MACD Crossover**  
   - Difference of two EMAs (fast & slow spans, adjustable via **MACD fast/slow span**),  
     plus a signal‐line EMA (adjustable via **MACD signal span**).  
   - Bullish when MACD crosses above its signal line; bearish on the flip.

Each indicator issues a **+1** (bull), **–1** (bear), or **0** (neutral). We sum them into a  
**Composite Vote** (from –3 to +3), then take a long/flat position based on the sign of  
that sum:

- **Composite ≥ +1** → Go **LONG**  
- **Composite ≤ –1** → Go **SHORT** (or exit if you prefer flat)  
- **Composite = 0** → **HOLD**

Below the tabs you can:

- **Run a single‐ticker backtest**  
- **Batch backtest** a list of tickers  
- **Grid‐search** the indicator parameters for the best simulated “Strategy %”  
""")

with tab_engine:
    # ───────────────────────────── Defaults & State ─────────────────────────────
    DEFAULTS = {
        "ma_window":   10,
        "rsi_period":  14,
        "macd_fast":   12,
        "macd_slow":   26,
        "macd_signal":  9
    }
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ───────────────────────────── Sidebar ─────────────────────────────
    st.sidebar.header("Controls")
    if st.sidebar.button("🔄 Reset to defaults"):
        for k, v in DEFAULTS.items():
            st.session_state[k] = v

    st.sidebar.header("Indicator Parameters")
    ma_window   = st.sidebar.slider("MA window",        5, 50, st.session_state["ma_window"],   key="ma_window")
    rsi_period  = st.sidebar.slider("RSI lookback",     5, 30, st.session_state["rsi_period"],  key="rsi_period")
    macd_fast   = st.sidebar.slider("MACD fast span",   5, 20, st.session_state["macd_fast"],   key="macd_fast")
    macd_slow   = st.sidebar.slider("MACD slow span",  20, 40, st.session_state["macd_slow"],   key="macd_slow")
    macd_signal = st.sidebar.slider("MACD signal span", 5, 20, st.session_state["macd_signal"], key="macd_signal")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Grid‐search parameters")
    ma_list  = st.sidebar.multiselect("MA windows",      [5,10,15,20,30], default=[ma_window],  key="grid_ma")
    rsi_list = st.sidebar.multiselect("RSI lookbacks",   [7,14,21,28],   default=[rsi_period], key="grid_rsi")
    mf_list  = st.sidebar.multiselect("MACD fast spans", [8,12,16,20],   default=[macd_fast],  key="grid_mf")
    ms_list  = st.sidebar.multiselect("MACD slow spans", [20,26,32,40], default=[macd_slow],  key="grid_ms")
    sg_list  = st.sidebar.multiselect("MACD sig spans",  [5,9,12,16],    default=[macd_signal],key="grid_sig")

    st.title("🚀 QuantaraX — Composite Signal Engine")

    # ───────────────────────────── Load & Compute ─────────────────────────────
    @st.cache_data(show_spinner=False)
    def load_and_compute(ticker, ma_w, rsi_p, mf, ms, sig):
        df = yf.download(ticker, period="6mo", progress=False)
        if df.empty or "Close" not in df:
            return pd.DataFrame()

        # Moving Average
        ma_col = f"MA{ma_w}"
        df[ma_col] = df["Close"].rolling(ma_w).mean()
        # RSI
        delta    = df["Close"].diff()
        up       = delta.clip(lower=0)
        down     = -delta.clip(upper=0)
        df[f"RSI{rsi_p}"] = 100 - 100/(1 + up.ewm(com=rsi_p-1).mean()/down.ewm(com=rsi_p-1).mean())
        # MACD
        ema_f = df["Close"].ewm(span=mf, adjust=False).mean()
        ema_s = df["Close"].ewm(span=ms, adjust=False).mean()
        macd  = ema_f - ema_s
        df["MACD"]        = macd
        df["MACD_Signal"] = macd.ewm(span=sig, adjust=False).mean()

        # drop only the columns we created
        required = [c for c in [ma_col, f"RSI{rsi_p}", "MACD", "MACD_Signal"] if c in df.columns]
        return df.dropna(subset=required).reset_index(drop=True)

    # ───────────────────────────── Build Composite ─────────────────────────────
    def build_composite(df, ma_w, rsi_p):
        n        = len(df)
        close    = df["Close"].to_numpy()
        ma_arr   = df[f"MA{ma_w}"].to_numpy()
        rsi_arr  = df[f"RSI{rsi_p}"].to_numpy()
        macd_arr = df["MACD"].to_numpy()
        sig_arr  = df["MACD_Signal"].to_numpy()

        ma_sig, rsi_sig, macd_sig2 = np.zeros(n), np.zeros(n), np.zeros(n)
        comp, trade                = np.zeros(n), np.zeros(n)

        for i in range(1,n):
            if close[i-1] < ma_arr[i-1] and close[i] > ma_arr[i]:
                ma_sig[i]= 1
            elif close[i-1] > ma_arr[i-1] and close[i] < ma_arr[i]:
                ma_sig[i]=-1

            if rsi_arr[i] < 30:
                rsi_sig[i]= 1
            elif rsi_arr[i] > 70:
                rsi_sig[i]=-1

            if macd_arr[i-1] < sig_arr[i-1] and macd_arr[i] > sig_arr[i]:
                macd_sig2[i]= 1
            elif macd_arr[i-1] > sig_arr[i-1] and macd_arr[i] < sig_arr[i]:
                macd_sig2[i]=-1

            comp[i]  = ma_sig[i]+rsi_sig[i]+macd_sig2[i]
            trade[i] = np.sign(comp[i])

        df["Composite"], df["Trade"] = comp, trade
        return df

    # ───────────────────────────── Backtest ─────────────────────────────
    def backtest(df):
        df = df.copy()
        df["Return"]   = df["Close"].pct_change().fillna(0)
        df["Position"] = df["Trade"].shift(1).fillna(0).clip(0,1)
        df["StratRet"] = df["Position"] * df["Return"]
        df["CumBH"]    = (1 + df["Return"]).cumprod()
        df["CumStrat"] = (1 + df["StratRet"]).cumprod()
        dd      = df["CumStrat"]/df["CumStrat"].cummax() - 1
        max_dd  = dd.min()*100
        sr      = df["StratRet"].mean()/df["StratRet"].std()*np.sqrt(252) if df["StratRet"].std()>0 else np.nan
        wr      = (df["StratRet"]>0).mean()*100
        return df, max_dd, sr, wr

    # ───────────────────────────── Single‐Ticker ─────────────────────────────
    st.markdown("## Single‐Ticker Backtest")
    ticker = st.text_input("Ticker", "AAPL").upper()
    if st.button("▶️ Run"):
        df = load_and_compute(ticker,ma_window,rsi_period,macd_fast,macd_slow,macd_signal)
        if df.empty:
            st.error("No data.")
        else:
            df = build_composite(df,ma_window,rsi_period)
            df, max_dd, sr, wr = backtest(df)
            rec = {1:"🟢 BUY",0:"🟡 HOLD",-1:"🔴 SELL"}[int(df["Trade"].iloc[-1])]
            st.success(f"{ticker}: {rec}")
            st.write(f"**Strategy %:** {(df['CumStrat'].iloc[-1]-1)*100:.2f}%   •   **Sharpe:** {sr:.2f}   •   **Max DD:** {max_dd:.2f}%   •   **Win Rate:** {wr:.1f}%")
            fig, ax = plt.subplots(2,1,figsize=(8,6),sharex=True)
            ax[0].plot(df["Close"],label="Close"); ax[0].plot(df[f"MA{ma_window}"],label=f"MA{ma_window}"); ax[0].legend()
            ax[1].plot(df["CumBH"],"--",label="B&H"); ax[1].plot(df["CumStrat"], label="Strat"); ax[1].legend()
            plt.xticks(rotation=45); plt.tight_layout()
            st.pyplot(fig)

    # ───────────────────────────── Batch & Grid Search … place them here … ─────────────────────────────
    # … your batch‐backtest and grid‐search code unchanged, 
    #    but be sure to call `build_composite(df, ma_w, rsi_p)` in each loop …
