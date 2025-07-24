# QuantaraX Composite Signals – BETA v2

[![Streamlit](https://img.shields.io/badge/Streamlit-App%20Beta-orange?logo=streamlit)](#)  
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](#)  
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🚀 What is QuantaraX?

QuantaraX is a lightweight, browser-based MVP for retail investors, built by quants and former traders, that:

- **Demystifies technical analysis** by blending Moving Average, RSI, and MACD signals into one clear “BUY / HOLD / SELL” recommendation.  
- **Backtests** strategies over 6 months of historical data with performance stats and equity curves.  
- **Runs batch comparisons** across multiple tickers.  
- **Simulates portfolios**, letting you input your own positions and see profit/loss, allocation pie charts, and suggestions.  
- **Surfaces live price and sentiment-weighted news** via VADER.  
- **Serves as a Progressive Web App (PWA)** for mobile “install” and offline caching.

This repo contains the full Streamlit app source code and all assets needed to run locally, on Streamlit Cloud, or as a PWA.

---

## 🎯 Features & Screens

1. **Single-Ticker Backtest**  
   - 6-month MA/RSI/MACD composite signal  
   - Live price & top-5 news with sentiment icons  
   - “Why this signal?” per-indicator rationale  
   - Key metrics: returns, Sharpe, drawdown, win rate  
   - Charts: price & MA, composite history, equity curves  

2. **Batch Backtest**  
   - Compare a comma-separated list of tickers side-by-side  
   - Downloadable CSV of performance table  

3. **Portfolio Simulator**  
   - Enter your positions (`TICKER,shares,cost_basis`)  
   - Set profit-target & loss-limit override sliders  
   - View market value, P/L %, composite suggestion, and final suggestion  
   - Allocation pie chart & total metrics  

4. **Hyperparameter Grid Search**  
   - Try multiple MA, RSI, MACD settings and rank by strategy returns  

5. **Watchlist Summary**  
   - Quick composite score & signal for any watchlist of tickers  
   - Expanders with per-ticker “Why?” reasoning  

6. **PWA Support**  
   - `public/manifest.json` & service worker pre-cache for offline use  
   - iOS “Add to Home Screen” friendly

---

## 🛠️ Getting Started

### Prerequisites

- Python **3.8+**  
- Git  

### Installation

```bash
# 1. Clone this repo
git clone https://github.com/<your-org>/quantarax-signal-engine.git
cd quantarax-signal-engine

# 2. (Optional) Create & activate virtual environment
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
