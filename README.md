# Bitcoin Trading Bot

An automated Bitcoin trading bot for Coinbase with multi-indicator confluence strategy, comprehensive safety systems, and paper trading mode.

## Features

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

### 3. Get Coinbase API Keys

1. Go to https://portal.cdp.coinbase.com/
2. Create an API key with **View** and **Trade** permissions
3. Download the JSON key file
4. Copy the key and secret to your `.env` file

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
| `TRADING_MODE` | `paper` | `paper` or `live` |
| `POSITION_SIZE_PERCENT` | `75` | Max position as % of portfolio |
| `SIGNAL_THRESHOLD` | `60` | Minimum score to trade (0-100) |
| `CHECK_INTERVAL_SECONDS` | `60` | Seconds between checks |
| `MAX_DAILY_LOSS_PERCENT` | `10` | Stop trading after this loss |

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
│   │   ├── coinbase_client.py   # Coinbase API
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
│   └── daemon/
│       └── runner.py            # Main loop
├── data/                        # SQLite database
├── logs/                        # Log files
└── .env                         # Configuration
```

## Running as a Service

### systemd (Linux)

Create `/etc/systemd/system/btc-bot.service`:

```ini
[Unit]
Description=Bitcoin Trading Bot
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/path/to/coinbase-trader
ExecStart=/usr/bin/python3 -m src.main
Restart=always
RestartSec=10
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable btc-bot
sudo systemctl start btc-bot
sudo journalctl -u btc-bot -f
```

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
- [ ] Test kill switch (touch data/KILL_SWITCH)
- [ ] Test circuit breaker behavior
- [ ] Start with 10% position size
- [ ] Monitor closely for first 48 hours

## Risk Warning

⚠️ **Trading cryptocurrency is risky.** This bot:
- Can lose money
- Makes no profit guarantees
- Requires monitoring
- Should start with small amounts

Only trade what you can afford to lose.

## License

MIT
