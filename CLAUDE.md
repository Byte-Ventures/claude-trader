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

## Database Operations

All database operations MUST support both PAPER and ACTUAL (live) trading modes. They MUST be kept completely separate:

- Every table with trading data has `is_paper` column
- All queries MUST filter by `is_paper` parameter
- Paper and live data must NEVER mix
- Both modes must be independently functional
- Test with paper trading before enabling live trading

## Versioning

Always update `src/version.py` when making commits:

- **MAJOR**: Breaking changes, major refactors
- **MINOR**: New features, significant enhancements
- **PATCH**: Bug fixes, small improvements

Update the version BEFORE committing.

## Configuration Parameters

When creating new configuration parameters:

1. Add the field to `config/settings.py` with proper Field validation
2. **MUST** update `.env.example` with the recommended default and documentation
3. Use descriptive comments explaining the parameter's purpose and valid range

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
6. **After every PR merge to `main`**, merge `main` back into `develop`:
   ```bash
   git fetch origin main
   git merge origin/main
   git push origin develop
   ```

### Creating PRs to Main

**Before creating a PR from `develop` to `main`, always pull main first:**

```bash
git fetch origin main
git log origin/main..develop --oneline  # See actual new commits
```

This ensures the PR description only includes commits not already in main.

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

1. **After creating a PR**, poll for review comments for at least 10 minutes:
   ```bash
   # Poll every 30 seconds for 10 minutes
   for i in {1..20}; do
     echo "Checking for reviews (attempt $i/20)..."
     gh api repos/Byte-Ventures/claude-trader/issues/{PR_NUMBER}/comments
     sleep 30
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

### Handling Identified Issues

**All issues identified in PR reviews MUST be handled.**

First, **verify the issue** - Check if the issue actually exists in the code (bot reviews can be wrong).

Then, do ONE of the following:

1. **Fix it** - Commit a fix addressing the issue
2. **Document why it's not an issue** - Add a code comment explaining why the concern doesn't apply
3. **Plan to fix later** - Create a TODO comment or issue tracking the fix for a future PR
4. **Explicitly decline** - Document in PR comments why the suggestion won't be implemented

No issue should be left unacknowledged. When summarizing PR reviews, create a checklist showing how each issue was handled.
