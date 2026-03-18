# Confluence Suite Bot 🤖

A Python trading signal bot that converts the **Confluence Suite** Pine Script indicator into a live alerting system.

It monitors **BTC** and **ETH** on a **1-hour timeframe** using the [Delta Exchange](https://www.delta.exchange/) public API, applies a full suite of technical indicators and an AI Neural Network signal classifier, and sends alerts exclusively for **Strong Buy** and **Strong Sell** signals to a Telegram bot.

> ⚠️ This bot sends **signal alerts only** — no order placement of any kind.

---

## Features

| Indicator | Description |
|---|---|
| Smart Trail | HMA + DWMA trend flow line with stdev bands |
| Trend Catcher | Super Smoother filter |
| Trend Tracer | Donchian channel midline |
| Trend Strength | VWMA + ATR metric (±100%) |
| Volatility Metric | ATR/price ratio |
| Squeeze Metric | Bollinger vs Keltner with LinReg |
| Reversal Zones | Nadaraya-Watson kernel regression → SL levels |
| Adaptive Supertrend | K-Means ATR multiplier selection |
| Neural Network Grader | 6 sub-scores → A+/A/B/C/D/F grade |
| Signal Threshold | Fires only when NN score ≥ 0.76 (Grade A) |

### Signal Message includes:
- Entry price, TP1 / TP2 / TP3, Stop Loss
- NN grade, score, and label (e.g. *Strong Buy*)
- Market context: Trend Bias, Trend Strength, Volatility, Squeeze %
- Exit marker warnings (EMA crossover + RSI)

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/ronadasakalesha/Confluence-Suite.git
cd Confluence-Suite
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment
```bash
cp .env.example .env
# Edit .env with your Telegram Bot Token and Chat ID
```

### 4. Run locally
```bash
python bot.py
```

---

## Deploy on Railway.app

1. Push this repo to GitHub
2. Create a new Railway project → **Deploy from GitHub repo**
3. Go to **Variables** and add:
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`
4. Railway will auto-detect the `Procfile` and deploy

---

## Environment Variables

| Variable | Description |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Your bot token from [@BotFather](https://t.me/BotFather) |
| `TELEGRAM_CHAT_ID` | Your chat/channel ID (use [@userinfobot](https://t.me/userinfobot)) |

---

## How It Works

The bot runs on a **60-minute polling loop**. On each tick:
1. Fetches the last 1000 × 1h candles for BTCUSD and ETHUSD from Delta Exchange
2. Computes all indicators on the **last confirmed (closed) candle**
3. Checks if the Adaptive Supertrend flipped direction
4. If so, runs the Neural Network grader — signals only fire when score ≥ 0.76
5. Sends a Telegram alert with full TP/SL levels and market context

---

## Project Structure

```
.
├── bot.py            # Main signal bot
├── requirements.txt  # Python dependencies
├── Procfile          # Railway / Heroku process definition
├── railway.toml      # Railway deployment config
├── .env.example      # Environment variable template
└── README.md
```

---

## Disclaimer

This software is for **educational and informational purposes only**.  
It does **not** place any trades or manage any funds.  
Always do your own research before making financial decisions.
