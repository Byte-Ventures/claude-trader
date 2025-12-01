# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the bot
python -m src.main

# Install dependencies
pip install -r requirements.txt

# Deploy to server (from repo root)
sudo ./scripts/update.sh
```

## Architecture

**Trading Bot** supporting Coinbase and Kraken exchanges with multi-indicator confluence strategy.

### Core Flow
`src/main.py` → `src/daemon/runner.py` (main loop) → Exchange client + Strategy + Safety systems

### Exchange Abstraction
All exchange clients implement `ExchangeClient` protocol (`src/api/exchange_protocol.py`):
- `coinbase_client.py`, `kraken_client.py`, `paper_client.py`
- Factory pattern in `exchange_factory.py` selects client based on config
- Trading pairs normalized to `BASE-QUOTE` format (e.g., `BTC-USD`)

### Safety Systems (all in `src/safety/`)
- **KillSwitch**: File-based or signal-based halt, requires manual reset
- **CircuitBreaker**: Multi-level (GREEN→YELLOW→RED→BLACK) with auto-cooldown
- **LossLimiter**: Daily/hourly loss limits with progressive throttling
- **Validator**: Aggregates all safety checks, provides position multiplier

### Strategy
`src/strategy/signal_scorer.py` combines RSI, MACD, Bollinger, EMA, Volume into -100 to +100 score. Trade when |score| ≥ threshold.

## Versioning

Always update `src/version.py` when making commits:

- **MAJOR**: Breaking changes, major refactors
- **MINOR**: New features, significant enhancements
- **PATCH**: Bug fixes, small improvements

Update the version BEFORE committing.

## Branching

- `main`: Production branch (deployed to server)
- `develop`: Development/integration branch
- `feature/*`: Feature branches (e.g., `feature/add-stop-loss`)
- `fix/*`: Bug fix branches (e.g., `fix/telegram-timeout`)

### Workflow

1. Create feature/fix branch from `develop`
2. Make changes and commit with version bump
3. Push and create PR to `develop`
4. After review, merge to `develop`
5. When ready for release, create PR from `develop` to `main`

### Rules

- All merges to `main` must be done via pull requests
- Direct commits to `main` are not allowed
- Tag releases on `main` with version (e.g., `v1.9.3`)

## Pull Request Reviews (CRITICAL)

**This is a financial trading tool. All PR review comments MUST be fetched and thoroughly analyzed before merging.**

### Fetching PR Review Comments

```bash
# Get all comments on a PR (includes bot reviews)
gh api repos/Byte-Ventures/claude-trader/issues/{PR_NUMBER}/comments

# Get code review comments (inline)
gh api repos/Byte-Ventures/claude-trader/pulls/{PR_NUMBER}/comments

# View PR with all details
gh pr view {PR_NUMBER} --comments
```

### Review Process

1. **After creating a PR**, poll for review comments for at least 5 minutes:
   ```bash
   # Poll every 60 seconds for 5 minutes
   for i in {1..5}; do
     echo "Checking for reviews (attempt $i/5)..."
     gh api repos/Byte-Ventures/claude-trader/issues/{PR_NUMBER}/comments
     sleep 60
   done
   ```
2. **Before merging any PR**, fetch and read ALL review comments
3. **Critical issues** (marked 🔴) must be fixed before merge
4. **High priority issues** (marked 🟡) should be addressed or documented why not
5. **Security concerns** require immediate attention
6. After fixes, push new commit and re-request review if needed

### Bot Reviews

The `claude[bot]` automatically reviews PRs. Its comments appear under `/issues/{PR_NUMBER}/comments`. Always read these thoroughly - they may identify:
- Type errors
- Security vulnerabilities
- Missing error handling
- API compatibility issues
