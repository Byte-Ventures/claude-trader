# Crypto Trading Bot

An automated cryptocurrency trading bot supporting **Coinbase** and **Kraken** exchanges with multi-indicator confluence strategy, comprehensive safety systems, and paper trading mode.

Works with any trading pair (BTC-USD, BTC-EUR, ETH-USD, etc.).

## Features

- **Multi-Exchange**: Supports Coinbase and Kraken with unified interface
- **Multi-Indicator Strategy**: Combines RSI, MACD, Bollinger Bands, EMA crossover, and ATR
- **Multi-Agent AI Review**: 3 reviewers (Pro/Neutral/Opposing) + judge for trade decisions
- **Hourly Market Analysis**: AI-powered analysis during volatile conditions
- **Safety Systems**: Kill switch, circuit breaker, loss limits, order validation
- **Paper Trading**: Test strategies with virtual money using real market data
- **Telegram Notifications**: Real-time alerts for trades, errors, and daily summaries
- **State Persistence**: SQLite database for trade history and recovery

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your credentials
```

### 3. Get Exchange API Keys

#### Coinbase
1. Go to https://portal.cdp.coinbase.com/
2. Create an API key with **View** and **Trade** permissions
3. Download the JSON key file or copy the key and secret to your `.env` file

#### Kraken
1. Go to https://www.kraken.com/u/security/api
2. Create an API key with **Query Funds** and **Create & Modify Orders** permissions
3. Copy the key and secret (base64-encoded) to your `.env` file

### 4. Set Up Telegram (Optional)

1. Message @BotFather on Telegram
2. Create a new bot and copy the token
3. Message @userinfobot to get your chat_id
4. Add both to your `.env` file

### 5. Run in Paper Mode

```bash
python -m src.main
```

The bot starts in paper trading mode by default. Monitor the logs to see how it performs.

## Configuration

Edit `.env` to customize. See `.env.example` for all options with documentation.

### Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `EXCHANGE` | `coinbase` | `coinbase` or `kraken` |
| `TRADING_PAIR` | `BTC-USD` | Trading pair (e.g., BTC-EUR, ETH-USD) |
| `TRADING_MODE` | `paper` | `paper` or `live` |
| `POSITION_SIZE_PERCENT` | `40` | Max position as % of portfolio |
| `SIGNAL_THRESHOLD` | `60` | Minimum score to trade (0-100) |
| `CHECK_INTERVAL_SECONDS` | `60` | Seconds between checks |

### Risk Management

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_DAILY_LOSS_PERCENT` | `10` | Stop trading after this daily loss |
| `MAX_HOURLY_LOSS_PERCENT` | `3` | Pause for 1 hour after this loss |
| `MAX_POSITION_PERCENT` | `80` | Maximum position size allowed |
| `STOP_LOSS_ATR_MULTIPLIER` | `1.5` | Stop loss distance (ATR multiples) |
| `TAKE_PROFIT_ATR_MULTIPLIER` | `2.0` | Take profit distance (ATR multiples) |

### Multi-Agent AI Trade Review (Optional)

Uses 3 reviewer agents with different stances (Pro, Neutral, Opposing) plus a judge for final decision.

| Variable | Default | Description |
|----------|---------|-------------|
| `AI_REVIEW_ENABLED` | `false` | Enable multi-agent AI review via OpenRouter |
| `OPENROUTER_API_KEY` | - | API key from openrouter.ai |
| `REVIEWER_MODEL_1` | `x-ai/grok-4-fast` | First reviewer model |
| `REVIEWER_MODEL_2` | `qwen/qwen3-next-80b-a3b-instruct` | Second reviewer model |
| `REVIEWER_MODEL_3` | `google/gemini-2.5-flash` | Third reviewer model |
| `JUDGE_MODEL` | `deepseek/deepseek-chat-v3.1` | Judge model for final decision |
| `VETO_ACTION` | `info` | `skip`, `reduce`, `delay`, or `info` |
| `AI_REVIEW_ALL` | `false` | Review ALL decisions (debug mode) |

### Hourly Market Analysis (Optional)

AI-powered market analysis with online research. Runs:
- **Hourly** during high/extreme volatility
- **Once** when volatility returns to normal (post-volatility analysis)

Uses the same multi-agent system as trade reviews (3 reviewers with bullish/neutral/bearish stances + judge). Fetches real-time data from CryptoCompare (news) and Blockchain.info (on-chain metrics). AI models can also search the web for additional context.

| Variable | Default | Description |
|----------|---------|-------------|
| `HOURLY_ANALYSIS_ENABLED` | `true` | Enable hourly AI analysis (uses reviewer models) |
| `MARKET_RESEARCH_ENABLED` | `true` | Fetch news and on-chain data from free APIs |
| `AI_WEB_SEARCH_ENABLED` | `true` | Allow AI to search web during analysis |
| `MARKET_RESEARCH_CACHE_MINUTES` | `15` | Cache duration for research data |

## Trading Strategy

The bot uses a **confluence scoring system** that combines multiple indicators with **graduated signals** (v1.7.0+):

| Indicator | Weight | Graduated Signal Range |
|-----------|--------|------------------------|
| RSI (14) | 25% | Dead zone 45-55, scaled ±0.3 to ±1.0 outside |
| MACD (12/26/9) | 25% | Crossover + histogram momentum |
| Bollinger Bands (20, 2σ) | 20% | %B based, dead zone 0.35-0.65 |
| EMA Crossover (9/21) | 15% | Position + momentum, dead zone <0.3% gap |
| Volume | 15% | Confirmation boost/penalty |
| Trend Filter | - | Counter-trend penalty |

**Trade when score ≥ threshold (default 60) or ≤ -threshold for sells**

### Signal Breakdown Example

```
Signal Score: 72/100
  📈 RSI: +18      (RSI at 38, moderate buy zone)
  📈 MACD: +15     (bullish crossover + histogram)
  📈 Bollinger: +12 (%B at 0.25, lower zone)
  📈 EMA: +10      (fast above slow, gap widening)
  📈 Volume: +7    (1.6x average volume boost)
  ➖ Trend Filter: 0
```

## Tuning Guide

### Trade Frequency

Target trades/month depends on market conditions and your risk tolerance:

| Profile | Threshold | Expected Trades | Notes |
|---------|-----------|-----------------|-------|
| Conservative | 70 | 5-15/month | Fewer but higher conviction |
| Moderate | 60 | 20-50/month | Balanced (default) |
| Aggressive | 50 | 50-100/month | More signals, more noise |

### Indicator Tuning

**RSI (Momentum)**
- `RSI_OVERSOLD=35` / `RSI_OVERBOUGHT=65` - Default thresholds
- Tighter (40/60): More signals, earlier entries
- Wider (30/70): Fewer signals, wait for extremes

**EMA (Trend)**
- `EMA_FAST=9` / `EMA_SLOW=21` - Default periods
- Shorter (5/13): More responsive, more whipsaws
- Longer (12/26): Smoother, slower to react

**Bollinger Bands (Volatility)**
- `BOLLINGER_STD=2.0` - Band width (95% of price action)
- Lower (1.5): Narrower bands, more touches
- Higher (2.5): Wider bands, only extreme moves

### Hot-Reload Settings

Update these without restarting:
```bash
nano .env  # Edit values
kill -SIGUSR2 $(pgrep -f "python -m src.main")
# Or with systemd:
sudo systemctl reload claude-trader
```

**Reloadable:** signal_threshold, RSI/MACD/Bollinger/EMA parameters, position_size_percent, loss limits, AI settings

**Requires restart:** exchange, trading_pair, trading_mode, database_path

## Safety Systems

### Kill Switch
- Create `data/KILL_SWITCH` file to halt trading
- Send `SIGUSR1` signal to process
- Requires manual reset to resume

### Circuit Breaker
- **YELLOW**: 5%+ price move → reduced position size
- **RED**: 10%+ price move → trading halted (4h cooldown)
- **BLACK**: 3+ order failures → manual reset required

### Loss Limits
- Daily: 10% max loss → trading stops for the day
- Hourly: 3% max loss → 1 hour pause
- Progressive throttling as limits approach

## Performance Reports

The bot automatically generates performance reports comparing your portfolio against buy-and-hold BTC:

- **Daily**: Sent via Telegram each day with portfolio return vs BTC return
- **Weekly**: Summary every Monday covering the past 7 days
- **Monthly**: Summary on the 1st of each month covering the previous month

Reports show **alpha** (portfolio return minus BTC return) to measure strategy effectiveness.

## Project Structure

```
coinbase-trader/
├── config/
│   ├── settings.py          # Pydantic configuration
│   └── logging_config.py    # Structured logging
├── src/
│   ├── api/
│   │   ├── exchange_protocol.py # Unified exchange interface
│   │   ├── exchange_factory.py  # Exchange client factory
│   │   ├── coinbase_client.py   # Coinbase API
│   │   ├── kraken_client.py     # Kraken API
│   │   └── paper_client.py      # Paper trading
│   ├── indicators/
│   │   ├── rsi.py, macd.py, bollinger.py, ema.py, atr.py
│   ├── strategy/
│   │   ├── signal_scorer.py     # Confluence scoring
│   │   └── position_sizer.py    # ATR-based sizing
│   ├── safety/
│   │   ├── kill_switch.py
│   │   ├── circuit_breaker.py
│   │   ├── loss_limiter.py
│   │   └── validator.py
│   ├── notifications/
│   │   └── telegram.py
│   ├── ai/
│   │   ├── trade_reviewer.py    # Multi-agent AI trade review
│   │   └── market_analyzer.py   # Hourly AI market analysis
│   ├── state/
│   │   └── database.py          # SQLite persistence
│   └── daemon/
│       └── runner.py            # Main loop
├── scripts/
│   ├── coinbase-trader.service  # systemd service file
│   └── install-service.sh       # Ubuntu install script
├── data/                        # SQLite database
├── logs/                        # Log files
└── .env                         # Configuration
```

## Running as a Service

### Ubuntu (Recommended)

Use the included install script:

```bash
# Clone and install
git clone https://github.com/Byte-Ventures/claude-trader.git
cd claude-trader
sudo ./scripts/install-service.sh

# Configure
sudo nano /opt/coinbase-trader/.env

# Start
sudo systemctl start coinbase-trader
sudo journalctl -u coinbase-trader -f
```

The service will:
- Start automatically on boot
- Restart within 10 seconds if it crashes
- Run as a dedicated `trader` user
- Log to systemd journal

**Service commands:**
```bash
sudo systemctl status coinbase-trader   # Check status
sudo systemctl stop coinbase-trader     # Stop
sudo systemctl restart coinbase-trader  # Restart
sudo journalctl -u coinbase-trader -f   # Follow logs
```

### Manual systemd Setup

If you prefer manual installation, copy `scripts/coinbase-trader.service` to `/etc/systemd/system/` and adjust paths.

### launchd (macOS)

Create `~/Library/LaunchAgents/com.btc-bot.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.btc-bot</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>-m</string>
        <string>src.main</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/path/to/coinbase-trader</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
```

```bash
launchctl load ~/Library/LaunchAgents/com.btc-bot.plist
```

## Before Going Live

- [ ] Run paper trading for 1+ week
- [ ] Verify Telegram notifications work
- [ ] Test kill switch (`touch data/KILL_SWITCH`)
- [ ] Test circuit breaker behavior
- [ ] Verify trading pair is valid on your exchange
- [ ] Start with small position size (default 40% is conservative)
- [ ] Monitor closely for first 48 hours

## Risk Warning

⚠️ **Trading cryptocurrency is risky.** This bot:
- Can and will lose money
- Makes no profit guarantees
- Requires active monitoring
- Should start with small amounts
- Includes fees in P&L calculations but slippage can vary

Only trade what you can afford to lose.

## Contributing

1. Fork the repository
2. Create a feature branch from `develop`: `git checkout -b feature/my-feature develop`
3. Make your changes and update `src/version.py`
4. Push and create a PR to `develop`
5. After review, changes will be merged to `develop`, then released to `main`

See `CLAUDE.md` for detailed branching workflow and coding guidelines.

## License

MIT
