# Claude Bitcoin Trader

*Get Rich or Vibe Tryin'*

An automated Bitcoin trading bot supporting **Coinbase** and **Kraken** exchanges with multi-indicator confluence strategy, AI-powered trade review, comprehensive safety systems, and paper trading mode.

Works with any trading pair (BTC-USD, BTC-EUR, ETH-USD, etc.).

## Features

- **Multi-Exchange**: Supports Coinbase and Kraken with unified interface
- **Multi-Indicator Strategy**: Combines RSI, MACD, Bollinger Bands, EMA crossover, and ATR
- **Multi-Agent AI Review**: 3 reviewers (Pro/Neutral/Opposing) + judge for trade decisions
- **Hourly Market Analysis**: AI-powered analysis during volatile conditions
- **Multi-Timeframe Confirmation**: Daily + 4-hour trend alignment before trading
- **Market Regime Adaptation**: Fear & Greed Index, volatility, and trend-aware adjustments
- **Whale Detection**: Boosts signal confidence on high-volume institutional activity
- **Trade Cooldown**: Prevents rapid consecutive trades (falling knife protection)
- **Live Dashboard**: Real-time web dashboard with charts, signals, and trade history
- **Safety Systems**: Kill switch, circuit breaker, loss limits, order validation
- **Paper Trading**: Test strategies with virtual money using real market data
- **Anti-Bot Mode**: Run inverse strategy alongside normal trading for performance comparison
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
| `AI_FAILURE_MODE_BUY` | `safe` | AI failure behavior for buys: `safe` (skip) or `open` (proceed) |
| `AI_FAILURE_MODE_SELL` | `open` | AI failure behavior for sells: `safe` (skip) or `open` (proceed) |
| `AI_RECOMMENDATION_TTL_MINUTES` | `20` | How long AI recs influence thresholds |
| `AI_MAX_TOKENS` | `4000` | Max tokens for AI responses |

**AI Recommendations**: When the judge issues "accumulate", "reduce", or "wait" recommendations, they influence signal thresholds for `AI_RECOMMENDATION_TTL_MINUTES`. Effect decays linearly to zero over this period.

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

### Market Protection Layers

The bot uses multiple complementary layers to protect against poor entries during extreme market conditions. These layers work together to provide defense-in-depth without being overly restrictive.

#### Fear & Greed Index Integration

The bot fetches the [Bitcoin Fear & Greed Index](https://alternative.me/crypto/fear-and-greed-index/) every 15 minutes and applies it across five protection layers:

**Categories**: Extreme Fear (0-25), Fear (25-45), Neutral (45-55), Greed (55-75), Extreme Greed (75-100)

#### Protection Layer 1: Regime Threshold Adjustments

Modifies the score threshold required to trigger trades based on market sentiment and trend alignment:

- **Extreme Fear + Bearish Trend + Buy Signal**: `threshold_mult: 0.0` - Neutralizes threshold reduction (fear justified, don't catch falling knife)
- **Extreme Fear + Bullish Trend + Buy Signal**: `threshold_mult: 1.2` - Amplifies threshold reduction (contrarian opportunity)
- **Extreme Greed + Bullish Trend + Sell Signal**: `threshold_mult: 1.5` - Amplifies threshold boost (sell into euphoria)

Default impact at scale=1.0: threshold adjusts ±10-15 points

#### Protection Layer 2: Regime Position Sizing

Adjusts position size based on market conditions:

- **Extreme Fear + Bearish Trend + Buy**: 30% position reduction (don't catch knife)
- **Extreme Fear + Bullish Trend + Buy**: 15% position increase (contrarian opportunity)
- **Extreme Greed + Bullish Trend + Sell**: 20% position increase (sell into strength)

#### Protection Layer 3: Extreme Fear MTF Override

When daily and 4H timeframes disagree during extreme fear, applies full counter-penalty instead of half:

- **Without extreme fear**: Daily disagrees with 4H → half penalty (-10 points)
- **With extreme fear**: Daily disagrees with 4H → full penalty (-20 points)

This prevents 4H neutral signals from neutralizing bearish daily trends during panic conditions.

#### Protection Layer 4: Dual-Extreme Blocking

**Completely blocks** buy orders when BOTH conditions exist:
- Sentiment = Extreme Fear (≤24/100)
- Volatility = Extreme (ATR-based)

These dual-extreme conditions create unfavorable risk/reward scenarios for new entries.

#### Protection Layer 5: Extreme Volatility Stop Widening

Widens stop losses during extreme volatility to prevent whipsaw exits:
- Normal volatility: 1.5x ATR stop distance
- Extreme volatility: 2.0x ATR stop distance

Prevents being stopped out by normal price fluctuations during high volatility.

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

### Adaptive Check Intervals

The bot can adjust check frequency based on market volatility:

| Variable | Default | Description |
|----------|---------|-------------|
| `ADAPTIVE_INTERVAL_ENABLED` | `true` | Enable volatility-based intervals |
| `INTERVAL_LOW_VOLATILITY` | `120` | Seconds between checks (low vol) |
| `INTERVAL_NORMAL` | `60` | Seconds between checks (normal) |
| `INTERVAL_HIGH_VOLATILITY` | `30` | Seconds between checks (high vol) |
| `INTERVAL_EXTREME_VOLATILITY` | `15` | Seconds between checks (extreme) |

### Candle Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `CANDLE_INTERVAL` | `ONE_HOUR` | Candlestick granularity for analysis |
| `CANDLE_LIMIT` | `100` | Number of candles to fetch |

**Reloadable:** signal_threshold, RSI/MACD/Bollinger/EMA parameters, position_size_percent, loss limits, AI settings

**Requires restart:** exchange, trading_pair, trading_mode, database_path

## Advanced Configuration

All settings below are documented in `.env.example` with detailed comments. Most users won't need to change these.

### Paper Trading Initial Balances

| Variable | Default | Description |
|----------|---------|-------------|
| `PAPER_INITIAL_QUOTE` | `5000` | Starting quote currency (USD/EUR) |
| `PAPER_INITIAL_BASE` | `0.05` | Starting base currency (BTC) |

### Anti-Bot Mode (Paper Trading Only)

Run an inverse strategy alongside your normal trading to compare performance. When enabled, every trade the normal bot makes is mirrored with the opposite action by an "anti-bot":

- Normal bot **buys** → Anti-bot **sells**
- Normal bot **sells** (signal or trailing stop) → Anti-bot **buys**

Both bots see identical signals and market conditions, making it a fair comparison. Each maintains separate virtual balances tracked independently in the database.

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_ANTI_BOT` | `false` | Enable anti-bot inverse trading |

**How it works:**
- On first enable, anti-bot copies balance from normal bot's current state
- Anti-bot can go negative on quote currency (USD/EUR) to simulate shorting
- Anti-bot cannot go negative on base currency (BTC) - can't sell what it doesn't have
- Both positions are tracked separately with `bot_mode` column in database
- Compare performance: if anti-bot consistently wins, consider inverting your strategy

**Use case:** Validate whether your trading signals have predictive value. If the anti-bot performs better over time, your signals may be systematically wrong.

### Indicator Parameters

| Indicator | Parameters | Defaults |
|-----------|------------|----------|
| RSI | `RSI_PERIOD`, `RSI_OVERSOLD`, `RSI_OVERBOUGHT` | 14, 35, 65 |
| MACD | `MACD_FAST`, `MACD_SLOW`, `MACD_SIGNAL` | 12, 26, 9 |
| EMA | `EMA_FAST`, `EMA_SLOW` | 9, 21 |
| Bollinger | `BOLLINGER_PERIOD`, `BOLLINGER_STD` | 20, 2.0 |
| ATR | `ATR_PERIOD` | 14 |

### Trade Cooldown

Prevents rapid consecutive trades (averaging into falling knives, excessive fees).

| Variable | Default | Description |
|----------|---------|-------------|
| `TRADE_COOLDOWN_ENABLED` | `true` | Enable cooldown between same-direction trades |
| `BUY_COOLDOWN_MINUTES` | `15` | Minimum minutes between buy trades |
| `SELL_COOLDOWN_MINUTES` | `0` | Minutes between sells (0 = disabled for safety) |
| `BUY_PRICE_CHANGE_PERCENT` | `1.0` | Price must drop X% to buy again |
| `SELL_PRICE_CHANGE_PERCENT` | `0` | Price must rise X% to sell again (0 = disabled) |

### Whale Detection

Detects large volume spikes indicating institutional activity, boosting signal confidence.

| Variable | Default | Description |
|----------|---------|-------------|
| `WHALE_VOLUME_THRESHOLD` | `3.0` | Volume ratio for whale detection (3x average) |
| `WHALE_DIRECTION_THRESHOLD` | `0.003` | Price change % to classify whale direction |
| `WHALE_BOOST_PERCENT` | `0.30` | Signal boost for whale activity (30%) |
| `HIGH_VOLUME_BOOST_PERCENT` | `0.20` | Signal boost for high volume (20%) |

### Order Execution

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_LIMIT_ORDERS` | `true` | Use IOC limit orders (reduces slippage) |
| `LIMIT_ORDER_OFFSET_PERCENT` | `0.1` | Offset from best bid/ask (0.1%) |

**Note**: IOC (Immediate-Or-Cancel) orders fill immediately or cancel unfilled portions.

### Trailing Stop & Breakeven

| Variable | Default | Description |
|----------|---------|-------------|
| `TRAILING_STOP_ATR_MULTIPLIER` | `1.0` | Trailing stop distance (1x ATR) |
| `BREAKEVEN_ATR_MULTIPLIER` | `0.5` | Profit needed to activate trailing (0.5x ATR) |
| `MIN_STOP_LOSS_PERCENT` | `1.5` | Minimum stop loss floor (% of price) |

### Market Regime Adaptation

Dynamically adjusts strategy based on Fear & Greed Index, volatility, and trend. See [Market Protection Layers](#market-protection-layers) for detailed explanation of how these work together.

| Variable | Default | Description |
|----------|---------|-------------|
| `REGIME_ADAPTATION_ENABLED` | `true` | Enable regime-based adjustments (Layers 1-2) |
| `REGIME_SENTIMENT_ENABLED` | `true` | Use Fear & Greed Index |
| `REGIME_VOLATILITY_ENABLED` | `true` | Use ATR-based volatility |
| `REGIME_TREND_ENABLED` | `true` | Use EMA trend direction |
| `REGIME_ADJUSTMENT_SCALE` | `0.5` | Intensity (0=off, 0.5=moderate, 1.0=full, 2.0=aggressive) |
| `REGIME_FLAP_PROTECTION` | `true` | Require 2 consecutive detections before regime change |
| `BLOCK_TRADES_EXTREME_CONDITIONS` | `true` | Block buys during extreme fear + extreme volatility (Layer 4) |
| `STOP_LOSS_ATR_MULTIPLIER_EXTREME` | `2.0` | Wider stops during extreme volatility (Layer 5) |

### Multi-Timeframe Confirmation (MTF)

Checks Daily + 4-hour trends before trading. Both must agree for strong bias. During extreme fear, applies full counter-penalty when daily/4H disagree (Layer 3).

| Variable | Default | Description |
|----------|---------|-------------|
| `MTF_ENABLED` | `true` | Enable higher timeframe confirmation |
| `MTF_4H_ENABLED` | `false` | Include 4H timeframe (default: daily-only is simpler) |
| `MTF_CANDLE_LIMIT` | `50` | Candles to fetch for HTF trend |
| `MTF_DAILY_CACHE_MINUTES` | `60` | Cache duration for daily candles |
| `MTF_4H_CACHE_MINUTES` | `30` | Cache duration for 4H candles |
| `MTF_ALIGNED_BOOST` | `20` | Score boost when aligned with HTF trend |
| `MTF_COUNTER_PENALTY` | `30` | Score penalty when against HTF trend (doubled during extreme fear when daily/4H disagree) |

### AI Weight Profile Selection

AI analyzes market conditions to select optimal indicator weights per candle.

| Variable | Default | Description |
|----------|---------|-------------|
| `AI_WEIGHT_PROFILE_ENABLED` | `false` | Enable AI-driven weight selection |
| `AI_WEIGHT_FALLBACK_PROFILE` | `default` | Fallback when AI unavailable |
| `AI_WEIGHT_PROFILE_MODEL` | `openai/gpt-5.2` | Model for profile selection |

**Profiles**: `trending` (MACD/EMA heavy), `ranging` (RSI/BB heavy), `volatile`, `default`

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

### Signal History Monitoring

The bot tracks signal history in the database. If storage fails 10 consecutive times, a Telegram alert is sent. Additional alerts at every 50 failures thereafter (60, 110, 160...). Trading continues even if history storage fails (non-blocking).

## Performance Reports

The bot automatically generates performance reports comparing your portfolio against buy-and-hold BTC:

- **Daily**: Sent via Telegram each day with portfolio return vs BTC return
- **Weekly**: Summary every Monday covering the past 7 days
- **Monthly**: Summary on the 1st of each month covering the previous month

Reports show **alpha** (portfolio return minus BTC return) to measure strategy effectiveness.

## Live Dashboard

A real-time web dashboard showing trading activity, signals, and portfolio status.

### Starting the Dashboard

```bash
# Terminal 1: Trading daemon
python -m src.main

# Terminal 2: Dashboard server
python -m src.dashboard.server
```

Dashboard accessible at `http://localhost:8081`

### Dashboard Features

- **Candlestick Chart**: Live price action with TradingView Lightweight Charts
- **Signal Display**: Current signal score (-100 to +100) with threshold indicator
- **Signal Breakdown**: Visual breakdown of RSI, MACD, Bollinger, EMA, Volume contributions
- **Portfolio Overview**: Quote/base balances, total value, position percentage
- **Circuit Breaker Status**: Current safety level (GREEN/YELLOW/RED/BLACK)
- **Recent Trades**: Last 20 trades with side, size, price, and P&L
- **Performance Chart**: Portfolio vs BTC buy-and-hold comparison
- **Notifications Feed**: Real-time trade reviews, market analysis, and alerts

### Dashboard Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DASHBOARD_HOST` | `127.0.0.1` | Bind address (use `0.0.0.0` for network access) |
| `DASHBOARD_PORT` | `8081` | Web server port |

The dashboard uses WebSocket for real-time updates (1.5s poll interval) and includes rate limiting on all API endpoints.

## Project Structure

```
claude-trader/
├── config/
│   ├── settings.py          # Pydantic configuration
│   └── logging_config.py    # Structured logging
├── src/
│   ├── main.py              # Entry point
│   ├── version.py           # Version string
│   ├── api/
│   │   ├── exchange_protocol.py # Unified exchange interface
│   │   ├── exchange_factory.py  # Exchange client factory
│   │   ├── coinbase_client.py   # Coinbase API
│   │   ├── coinbase_auth.py     # Coinbase JWT authentication
│   │   ├── kraken_client.py     # Kraken API
│   │   ├── kraken_auth.py       # Kraken request signing
│   │   ├── paper_client.py      # Paper trading
│   │   └── symbol_mapper.py     # Exchange symbol normalization
│   ├── indicators/
│   │   ├── rsi.py, macd.py, bollinger.py, ema.py, atr.py
│   ├── strategy/
│   │   ├── signal_scorer.py     # Confluence scoring
│   │   ├── position_sizer.py    # ATR-based sizing
│   │   ├── regime.py            # Market regime detection
│   │   └── weight_profile_selector.py # AI-driven indicator weights
│   ├── safety/
│   │   ├── kill_switch.py
│   │   ├── circuit_breaker.py
│   │   ├── loss_limiter.py
│   │   ├── trade_cooldown.py    # Consecutive trade prevention
│   │   └── validator.py
│   ├── notifications/
│   │   └── telegram.py
│   ├── ai/
│   │   ├── trade_reviewer.py    # Multi-agent AI trade review
│   │   ├── market_analyzer.py   # Hourly AI market analysis
│   │   ├── market_research.py   # News and on-chain data fetching
│   │   ├── sentiment.py         # Fear & Greed Index integration
│   │   └── web_search.py        # DuckDuckGo search for AI agents
│   ├── state/
│   │   └── database.py          # SQLite persistence
│   ├── daemon/
│   │   └── runner.py            # Main loop
│   ├── dashboard/
│   │   ├── server.py            # FastAPI dashboard server
│   │   ├── routes.py            # REST API endpoints
│   │   ├── websocket.py         # WebSocket manager
│   │   └── models.py            # Pydantic response models
│   ├── static/
│   │   ├── dashboard.css        # Dashboard styles
│   │   └── dashboard.js         # Frontend JavaScript
│   └── templates/
│       └── index.html           # Dashboard HTML
├── scripts/
│   ├── claude-trader.service    # Main bot systemd service
│   ├── claude-trader-dashboard.service # Dashboard systemd service
│   ├── install-service.sh       # Ubuntu install script
│   ├── reload-config.sh         # Hot-reload .env settings
│   └── update.sh                # Git pull + restart deployment
├── data/                        # SQLite database
├── logs/                        # Log files
└── .env                         # Configuration
```

## Testing

The project includes a comprehensive test suite with 300+ tests covering safety systems, strategy logic, and API integrations.

**Run all tests:**
```bash
pytest
```

**Run with coverage:**
```bash
pytest --cov=src --cov-report=html
```

**Live Integration Tests:**

The test suite includes optional integration tests that make real API calls to Coinbase. These tests verify authentication, data retrieval, and order execution against the live API.

⚠️  **WARNING**: Live tests place real orders and cost real money in fees (~$0.30-0.60 per run).

Setup requires a separate test account with $100+ balance. See [tests/README.md](tests/README.md#live-integration-tests) for detailed setup instructions, safety features, and usage.

```bash
# Run live integration tests (requires setup)
pytest tests/test_coinbase_integration_live.py --run-live-tests -v
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
sudo nano /opt/claude-trader/.env

# Start
sudo systemctl start claude-trader
sudo journalctl -u claude-trader -f
```

The service will:
- Start automatically on boot
- Restart within 10 seconds if it crashes
- Run as a dedicated `trader` user
- Log to systemd journal

**Service commands:**
```bash
sudo systemctl status claude-trader   # Check status
sudo systemctl stop claude-trader     # Stop
sudo systemctl restart claude-trader  # Restart
sudo journalctl -u claude-trader -f   # Follow logs
```

### Manual systemd Setup

If you prefer manual installation, copy `scripts/claude-trader.service` to `/etc/systemd/system/` and adjust paths.

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
    <string>/path/to/claude-trader</string>
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

## Disclaimer

⚠️ **WARNING: YOU WILL VERY LIKELY LOSE MONEY** ⚠️

This software is provided for **educational and experimental purposes only**. By using this application, you acknowledge and accept that:

- **You will very likely lose some or all of your money**
- This bot makes no profit guarantees whatsoever
- Past performance (if any) does not indicate future results
- The authors and contributors are not financial advisors
- The authors and contributors accept no liability for any financial losses
- Cryptocurrency markets are extremely volatile and unpredictable
- Automated trading amplifies both gains and losses
- Technical failures, bugs, or exchange issues can cause unexpected losses

**Do not use money you cannot afford to lose completely.**

This is an experimental project. If you choose to use it with real money, you do so entirely at your own risk.

## Contributing

1. Fork the repository
2. Create a feature branch from `develop`: `git checkout -b feature/my-feature develop`
3. Make your changes and update `src/version.py`
4. Push and create a PR to `develop`
5. After review, changes will be merged to `develop`, then released to `main`

See `CLAUDE.md` for detailed branching workflow and coding guidelines.

## License

[AGPL-3.0](LICENSE) - All modifications must be shared under the same license.
