import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from itertools import product

st.set_page_config(page_title="🎯 QuantaraX — Composite Signal Engine", layout="wide")

# ─── SESSION‐STATE DEFAULTS ────────────────────────────────────────────────────
DEFAULTS = {
    "ma_window": 10,
    "rsi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_sig": 9,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── SIDEBAR CONTROLS ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("🔄 Reset to defaults"):
        for k, v in DEFAULTS.items():
            st.session_state[k] = v

    st.markdown("### Indicator Parameters")
    st.session_state.ma_window  = st.slider("MA window",        5, 50,  st.session_state.ma_window)
    st.session_state.rsi_period = st.slider("RSI lookback",     5, 30,  st.session_state.rsi_period)
    st.session_state.macd_fast  = st.slider("MACD fast span",   5, 20,  st.session_state.macd_fast)
    st.session_state.macd_slow  = st.slider("MACD slow span",  20, 40,  st.session_state.macd_slow)
    st.session_state.macd_sig   = st.slider("MACD signal span", 5, 20,  st.session_state.macd_sig)

    st.markdown("---")
    st.markdown("### Grid‐search parameters")
    ma_grid   = st.multiselect("MA windows to test",      [5,10,15,20,30,50], default=[st.session_state.ma_window])
    rsi_grid  = st.multiselect("RSI lookbacks to test",   [5,14,21,28],     default=[st.session_state.rsi_period])
    mf_grid   = st.multiselect("MACD fast spans to test", [5,12,15],        default=[st.session_state.macd_fast])
    ms_grid   = st.multiselect("MACD slow spans to test", [20,26,31,40],    default=[st.session_state.macd_slow])
    sg_grid   = st.multiselect("MACD sig spans to test",  [5,9,12,15,20],   default=[st.session_state.macd_sig])

# ─── DATA LOADER & INDICATOR CALCS ─────────────────────────────────────────────
@st.cache_data
def load_and_compute(ticker: str,
                     ma_window:int,
                     rsi_period:int,
                     macd_fast:int,
                     macd_slow:int,
                     macd_sig:int) -> pd.DataFrame:
    # 1 year of history
    end   = datetime.datetime.today()
    start = end - datetime.timedelta(days=365)
    df = yf.download(ticker, start=start, end=end)
    if df.empty:
        raise ValueError(f"No data for {ticker}")

    # 10-day MA
    df[f"MA{ma_window}"] = df["Close"].rolling(ma_window).mean()

    # RSI
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.rolling(rsi_period).mean()
    avg_loss = loss.rolling(rsi_period).mean()
    rs = avg_gain / avg_loss
    df[f"RSI{rsi_period}"] = 100 - (100/(1+rs))

    # MACD
    exp1 = df["Close"].ewm(span=macd_fast, adjust=False).mean()
    exp2 = df["Close"].ewm(span=macd_slow, adjust=False).mean()
    df["MACD"]        = exp1 - exp2
    df["MACD_Signal"] = df["MACD"].ewm(span=macd_sig, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

    # *only drop on columns that actually exist*:
    required = ["Close",
                f"MA{ma_window}",
                f"RSI{rsi_period}",
                "MACD",
                "MACD_Signal"]
    present = [c for c in required if c in df.columns]
    df = df.dropna(subset=present).reset_index()

    return df

# ─── BACKTEST FUNCTION ──────────────────────────────────────────────────────────
def run_backtest(df: pd.DataFrame, ma_window:int, rsi_period:int) -> dict:
    # MA crossover signals
    df["MA_Signal"] = np.where(
        (df["Close"].shift(1) < df[f"MA{ma_window}"].shift(1)) &
        (df["Close"] >      df[f"MA{ma_window}"]), 1,
        np.where(
            (df["Close"].shift(1) > df[f"MA{ma_window}"].shift(1)) &
            (df["Close"] <      df[f"MA{ma_window}"]), -1,
            0
        )
    )
    # RSI overbought/oversold
    df["RSI_Signal"] = np.where(df[f"RSI{rsi_period}"] < 30, 1,
                         np.where(df[f"RSI{rsi_period}"] > 70, -1, 0))
    # MACD crossover
    df["MACD_Signal_Ind"] = np.where(
        (df["MACD"].shift(1) < df["MACD_Signal"].shift(1)) &
        (df["MACD"] >      df["MACD_Signal"]), 1,
        np.where(
            (df["MACD"].shift(1) > df["MACD_Signal"].shift(1)) &
            (df["MACD"] <      df["MACD_Signal"]), -1, 0
        )
    )

    # Composite
    df["Composite"] = df[["MA_Signal","RSI_Signal","MACD_Signal_Ind"]].sum(axis=1)
    df["Signal"]    = np.where(df["Composite"]>0, "BUY",
                       np.where(df["Composite"]<0, "SELL","HOLD"))

    # Strategy equity curve
    df["Position"] = df["Signal"].shift(1).map({"BUY":1,"SELL":-1,"HOLD":0}).ffill().fillna(0)
    df["Return"]   = df["Close"].pct_change() * df["Position"]
    df["Equity"]   = (1 + df["Return"]).cumprod()
    df["BH"]       = (1 + df["Close"].pct_change()).cumprod()

    # Metrics
    strat_ret = df["Equity"].iloc[-1] - 1
    bh_ret    = df["BH"].iloc[-1] - 1
    sharpe    = df["Return"].mean()/df["Return"].std()*np.sqrt(252)
    dd        = df["Equity"].cummax()
    max_dd    = ((df["Equity"] - dd)/dd).min()
    win_rate  = (df["Return"] > 0).mean()

    # Price+MA plot
    fig_p, axp = plt.subplots()
    axp.plot(df["Date"], df["Close"], label="Close")
    axp.plot(df["Date"], df[f"MA{ma_window}"], "--", label=f"MA{ma_window}")
    axp.set_title("Price & MA"); axp.legend()

    # Equity curves
    fig_e, axe = plt.subplots()
    axe.plot(df["Date"], df["BH"],      "--", label="Buy & Hold")
    axe.plot(df["Date"], df["Equity"], "-",  label="Strategy")
    axe.set_title("Equity Curves"); axe.legend()

    return {
        "df":         df,
        "fig_price":  fig_p,
        "fig_equity": fig_e,
        "metrics": {
            "bh_ret":   bh_ret,
            "strat_ret":strat_ret,
            "sharpe":   sharpe,
            "max_dd":   max_dd,
            "win_rate": win_rate
        }
    }

# ─── APP LAYOUT ─────────────────────────────────────────────────────────────────
st.title("🚀 QuantaraX — Composite Signal Engine")

# Single‐ticker section
st.subheader("🔍 Single-Ticker Backtest")
ticker = st.text_input("Ticker (e.g. AAPL)", "AAPL").upper()
if st.button("▶️ Run Composite Backtest"):
    try:
        df    = load_and_compute(ticker,
                                 st.session_state.ma_window,
                                 st.session_state.rsi_period,
                                 st.session_state.macd_fast,
                                 st.session_state.macd_slow,
                                 st.session_state.macd_sig)
        out   = run_backtest(df,
                             st.session_state.ma_window,
                             st.session_state.rsi_period)
        m     = out["metrics"]
        vote  = df["Composite"].iat[-1]
        sig   = "BUY"  if vote> 0 else "SELL" if vote< 0 else "HOLD"
        emoji = "🟢"   if sig=="BUY" else "🔴" if sig=="SELL" else "🟡"
        st.success(f"{ticker}: {emoji} {sig}")

        st.markdown(
            f"**Buy & Hold:** {m['bh_ret']:.2%}   •   "
            f"**Strategy:** {m['strat_ret']:.2%}   •   "
            f"**Sharpe:** {m['sharpe']:.2f}   •   "
            f"**Max Drawdown:** {m['max_dd']:.2%}   •   "
            f"**Win Rate:** {m['win_rate']:.2%}"
        )

        st.pyplot(out["fig_price"])
        st.pyplot(out["fig_equity"])

        csv = out["df"].to_csv(index=False)
        st.download_button("⬇️ Download signals CSV", data=csv,
                           file_name=f"{ticker}_signals.csv")
    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("---")

# Batch‐ticker section
st.subheader("🗂️ Batch Backtest")
tickers = st.text_area("Enter tickers (comma-separated)", "AAPL, MSFT, TSLA, SPY, QQQ")
if st.button("▶️ Run Batch Backtest"):
    perf = []
    for t in [x.strip().upper() for x in tickers.split(",") if x.strip()]:
        try:
            df  = load_and_compute(t,
                                   st.session_state.ma_window,
                                   st.session_state.rsi_period,
                                   st.session_state.macd_fast,
                                   st.session_state.macd_slow,
                                   st.session_state.macd_sig)
            out = run_backtest(df,
                               st.session_state.ma_window,
                               st.session_state.rsi_period)
            vote = df["Composite"].iat[-1]
            perf.append({
                "Ticker":         t,
                "Composite":      int(vote),
                "Signal":         "BUY" if vote>0 else "SELL" if vote<0 else "HOLD",
                "BuyHold %":      out["metrics"]["bh_ret"]*100,
                "Strat %":        out["metrics"]["strat_ret"]*100,
                "Sharpe":         out["metrics"]["sharpe"],
                "Max Drawdown %": out["metrics"]["max_dd"]*100,
                "Win Rate %":     out["metrics"]["win_rate"]*100
            })
        except Exception as e:
            perf.append({"Ticker":t, "Error":str(e)})

    df_perf = pd.DataFrame(perf)
    st.dataframe(df_perf, use_container_width=True)
    st.download_button("⬇️ Download performance CSV",
                       data=df_perf.to_csv(index=False),
                       file_name="batch_performance.csv")

st.markdown("---")

# Grid‐search section
st.subheader("🔧 Hyper-parameter Optimization")
gs_t = st.text_input("Grid-Search Ticker", "AAPL").upper()
if st.button("🏃 Run Grid Search"):
    grid_out = []
    for mw, rp, mf, ms, sg in product(ma_grid, rsi_grid, mf_grid, ms_grid, sg_grid):
        try:
            df  = load_and_compute(gs_t, mw, rp, mf, ms, sg)
            out = run_backtest(df, mw, rp)
            grid_out.append({
                "MA":          mw,
                "RSI":         rp,
                "MACD_fast":   mf,
                "MACD_slow":   ms,
                "MACD_sig":    sg,
                "Strat %":     out["metrics"]["strat_ret"]*100,
                "Sharpe":      out["metrics"]["sharpe"],
                "Win Rate %":  out["metrics"]["win_rate"]*100
            })
        except:
            continue

    if not grid_out:
        st.warning("No valid runs")
    else:
        dfg = pd.DataFrame(grid_out).sort_values("Sharpe", ascending=False)
        st.dataframe(dfg, use_container_width=True)
        st.download_button("⬇️ Download grid-search CSV",
                           data=dfg.to_csv(index=False),
                           file_name="grid_search.csv")
