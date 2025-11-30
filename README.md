# Crypto Trading Bot

An automated cryptocurrency trading bot supporting **Coinbase** and **Kraken** exchanges with multi-indicator confluence strategy, comprehensive safety systems, and paper trading mode.

Works with any trading pair (BTC-USD, BTC-EUR, ETH-USD, etc.).

## Features

- **Multi-Exchange**: Supports Coinbase and Kraken with unified interface
- **Multi-Indicator Strategy**: Combines RSI, MACD, Bollinger Bands, EMA crossover, and ATR
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

Edit `.env` to customize:

| Variable | Default | Description |
|----------|---------|-------------|
| `EXCHANGE` | `coinbase` | `coinbase` or `kraken` |
| `TRADING_PAIR` | `BTC-USD` | Trading pair (e.g., BTC-EUR, ETH-USD) |
| `TRADING_MODE` | `paper` | `paper` or `live` |
| `POSITION_SIZE_PERCENT` | `40` | Max position as % of portfolio |
| `SIGNAL_THRESHOLD` | `60` | Minimum score to trade (0-100) |
| `CHECK_INTERVAL_SECONDS` | `60` | Seconds between checks |
| `MAX_DAILY_LOSS_PERCENT` | `10` | Stop trading after this loss |
| `PAPER_INITIAL_QUOTE` | `5000` | Starting quote currency for paper trading |
| `PAPER_INITIAL_BASE` | `0.05` | Starting base currency for paper trading (~50/50 split) |

## Trading Strategy

The bot uses a **confluence scoring system** that combines multiple indicators:

| Indicator | Weight | Signal |
|-----------|--------|--------|
| RSI (14) | 25% | Buy < 35, Sell > 65 |
| MACD (12/26/9) | 25% | Crossover signals |
| Bollinger Bands (20, 2σ) | 20% | Band touches |
| EMA Crossover (9/21) | 15% | Trend direction |
| Volume | 15% | Confirmation boost |

**Trade when score ≥ 60 (or ≤ -60 for sells)**

## Safety Systems

### Kill Switch
- Create `data/KILL_SWITCH` file to halt trading
- Send `SIGUSR1` signal to process
- Requires manual reset to resume

### Config Hot-Reload
Update strategy parameters without restarting:

```bash
# Edit .env with new values (thresholds, position sizes, limits)
nano .env

# Trigger reload
kill -SIGUSR2 $(pgrep -f "python -m src.main")
```

**Reloadable settings:** signal_threshold, RSI/MACD/Bollinger/EMA parameters, position_size_percent, stop_loss/take_profit multipliers, max_position_percent, loss limits.

**Requires restart:** exchange, trading_pair, trading_mode, database_path.

### Circuit Breaker
- **YELLOW**: 5%+ price move → reduced position size
- **RED**: 10%+ price move → trading halted (4h cooldown)
- **BLACK**: 3+ order failures → manual reset required

### Loss Limits
- Daily: 10% max loss → trading stops for the day
- Hourly: 3% max loss → 1 hour pause
- Progressive throttling as limits approach

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

## License

MIT
