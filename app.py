# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Quantarax — Composite Signal Engine")

# ---- Sidebar & reset-to-defaults ----
with st.sidebar:
    if st.button("↺ Reset to defaults"):
        st.experimental_rerun()

    st.header("Indicator Parameters")
    ma_window   = st.slider("MA window",    5, 50, 10, key="ma")
    rsi_period  = st.slider("RSI lookback", 5, 30, 14, key="rsi")
    macd_fast   = st.slider("MACD fast span",  5, 20, 12, key="mf")
    macd_slow   = st.slider("MACD slow span", 20, 40, 26, key="ms")
    macd_signal = st.slider("MACD signal span", 5, 20, 9, key="mc")

# ---- Utility functions ----
@st.cache_data(show_spinner=False)
def load_and_compute(ticker, ma_w, rsi_p, mf, ms, mc):
    # 1) fetch
    end   = datetime.datetime.today()
    start = end - datetime.timedelta(days=365)
    df    = yf.download(ticker, start=start, end=end)

    # 2) compute indicators
    df["MA10"] = df["Close"].rolling(ma_w).mean()
    # RSI
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.rolling(rsi_p).mean()
    avg_loss = loss.rolling(rsi_p).mean()
    rs = avg_gain / avg_loss
    df["RSI14"] = 100 - (100/(1+rs))
    # MACD
    ema_fast = df["Close"].ewm(span=mf, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=ms, adjust=False).mean()
    df["MACD"]        = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=mc, adjust=False).mean()

    # 3) only drop rows where *all* of the ones we actually created are NaN
    required = [c for c in ["MA10","RSI14","MACD","MACD_Signal"] if c in df.columns]
    return df.dropna(subset=required).reset_index()

def composite_backtest(df):
    # vote = MA + RSI + MACD
    # MA vote: +1 if price > MA, -1 if price < MA
    df["MA_vote"]   = np.where(df["Close"] > df["MA10"],  1, -1)
    # RSI vote: +1 if RSI < 30 (oversold buy), -1 if RSI > 70 (overbought sell), else 0
    df["RSI_vote"]  = np.where(df["RSI14"] < 30, 1,
                      np.where(df["RSI14"] > 70, -1, 0))
    # MACD vote: +1 if MACD line crosses above signal line, -1 opposite, else 0
    cross_up   = (df["MACD"].shift(1) < df["MACD_Signal"].shift(1)) & (df["MACD"] > df["MACD_Signal"])
    cross_down = (df["MACD"].shift(1) > df["MACD_Signal"].shift(1)) & (df["MACD"] < df["MACD_Signal"])
    df["MACD_vote"] = np.where(cross_up, 1, np.where(cross_down, -1, 0))

    # composite
    df["Composite"] = df[["MA_vote","RSI_vote","MACD_vote"]].sum(axis=1)

    # create a strategy position: +1 if Composite >= 1, -1 if Composite <= -1, else 0
    df["Position"] = np.where(df["Composite"] >= 1,  1,
                       np.where(df["Composite"] <= -1, -1, 0))

    # compute returns
    df["Market_Ret"]   = df["Close"].pct_change().shift(-1)
    df["Strategy_Ret"] = df["Market_Ret"] * df["Position"]

    # accumulate
    df["Market_Cum"]   = (1 + df["Market_Ret"].fillna(0)).cumprod()
    df["Strategy_Cum"] = (1 + df["Strategy_Ret"].fillna(0)).cumprod()

    return df

def perf_stats(df):
    # final returns
    buy_hold = df["Market_Cum"].iloc[-1] - 1
    strat    = df["Strategy_Cum"].iloc[-1] - 1
    # sharpe (annualized)
    sr = (df["Strategy_Ret"].mean() / df["Strategy_Ret"].std()) * np.sqrt(252) if df["Strategy_Ret"].std()>0 else 0
    # drawdown
    dd = (df["Strategy_Cum"].cummax() - df["Strategy_Cum"]).max()
    # win rate
    wr = (df["Strategy_Ret"]>0).mean()
    return buy_hold, strat, sr, dd, wr

# ---- Page Title ----
st.title("🚀 Quantarax — Composite Signal Engine")
st.markdown("MA + RSI + MACD Composite Signals & Backtest")

# ---- SINGLE-TICKER SECTION ----
st.subheader("🔍 Single-Ticker Backtest")
ticker = st.text_input("Ticker (e.g. AAPL)", "AAPL").upper()

if st.button("▶️ Run Composite Backtest"):
    df = load_and_compute(ticker, ma_window, rsi_period, macd_fast, macd_slow, macd_signal)
    st.write("**Debug: columns present →**", df.columns.tolist())
    if df.empty:
        st.error("No data returned for that ticker.")
    else:
        df = composite_backtest(df)
        bh, st_ret, sharpe, mdd, winr = perf_stats(df)
        st.success(f"{ticker}:  {['SELL','HOLD','BUY'][np.sign(df['Composite'].iloc[-1])+1]}")
        st.markdown(f"**Buy & Hold:** {bh*100:.2f}%   |   **Strategy:** {st_ret*100:.2f}%")
        st.markdown(f"Sharpe: {sharpe:.2f}   |   Max Drawdown: {mdd*100:.2f}%   |   Win Rate: {winr*100:.1f}%")

        # price vs MA
        fig1, ax1 = plt.subplots()
        df.plot(x="Date", y=["Close","MA10"], ax=ax1)
        ax1.set_title("Price & MA10")
        st.pyplot(fig1)

        # composite vote bar
        fig2, ax2 = plt.subplots()
        df["Composite"].plot(kind="bar", ax=ax2, width=1)
        ax2.set_title("Composite Vote")
        st.pyplot(fig2)

        # equity curves
        fig3, ax3 = plt.subplots()
        df.plot(x="Date", y=["Market_Cum","Strategy_Cum"], ax=ax3)
        ax3.set_title("Equity Curves")
        st.pyplot(fig3)

# ---- BATCH SECTION ----
st.subheader("📊 Batch Backtest")
tickers = st.text_area("Enter tickers (comma-separated)", "AAPL, MSFT, TSLA, SPY, QQQ")
if st.button("▶️ Run Batch Backtest"):
    rows = []
    for t in [t.strip().upper() for t in tickers.split(",")]:
        try:
            df_t = load_and_compute(t, ma_window, rsi_period, macd_fast, macd_slow, macd_signal)
            df_t = composite_backtest(df_t)
            bh, stt, s, d, w = perf_stats(df_t)
            last_vote = df_t["Composite"].iloc[-1]
            sig = ["SELL","HOLD","BUY"][np.sign(last_vote)+1]
            rows.append({
                "Ticker": t,
                "Composite vote": int(last_vote),
                "Signal": sig,
                "BuyHold %": f"{bh*100:.3f}",
                "Strat %": f"{stt*100:.3f}",
                "Sharpe": f"{s:.3f}",
                "Max Drawdown %": f"{d*100:.3f}",
                "Win Rate %": f"{w*100:.1f}"
            })
        except Exception:
            rows.append({"Ticker": t, "Error": "❌ data"})
    df_out = pd.DataFrame(rows)
    st.dataframe(df_out, use_container_width=True)
    csv = df_out.to_csv(index=False).encode("utf-8")
    st.download_button("Download performance CSV", csv, f"batch_{datetime.date.today()}.csv", "text/csv")
