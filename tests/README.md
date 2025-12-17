# Claude Trader Tests

Comprehensive test suite for the claude-trader cryptocurrency trading bot.

## Setup

Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

## Running Tests

Run all tests:
```bash
pytest
```

Run specific test file:
```bash
pytest tests/test_circuit_breaker.py
```

Run with coverage report:
```bash
pytest --cov=src --cov-report=html
```

Run specific test:
```bash
pytest tests/test_circuit_breaker.py::test_green_to_yellow_price_drop
```

## Test Structure

### `test_circuit_breaker.py`
Comprehensive tests for the circuit breaker safety system (43 tests):
- **Initialization**: Default/custom config, callback registration
- **State Transitions**: GREEN→YELLOW→RED→BLACK transitions
- **Price Movement**: Flash crashes, spikes, sustained drops
- **Failure Tracking**: API and order failure counters
- **Recovery**: Cooldown auto-recovery, manual reset
- **Properties**: can_trade, position_multiplier, status
- **Edge Cases**: Boundary conditions, error handling

## Test Coverage Goals

- **Safety Systems**: 100% coverage (critical for financial safety)
- **Strategy & Signals**: 95%+ coverage
- **Indicators**: 90%+ coverage (mathematical correctness)
- **Exchange Clients**: 85%+ coverage (with mocked APIs)

## Test Categories

Tests are marked with pytest markers:

- `@pytest.mark.unit` - Fast unit tests
- `@pytest.mark.integration` - Integration tests (may require external services)
- `@pytest.mark.slow` - Tests that take longer to run
- `@pytest.mark.integration_live` - Live API integration tests (requires --run-live-tests flag)

## Live Integration Tests

### Overview

Live integration tests make **REAL API calls** to Coinbase to verify authentication, data retrieval, and order execution work correctly. These tests are disabled by default and require explicit opt-in with credentials and a CLI flag.

⚠️  **WARNING: These tests place REAL orders on Coinbase and cost real money in fees.**

### Setup

1. **Create a separate test account**
   - Go to https://coinbase.com and create a NEW account specifically for testing
   - **NEVER use your production trading account**
   - Fund the account with $100 USD minimum (recommended: $100)

2. **Generate API credentials**
   - Visit https://portal.cdp.coinbase.com/
   - Create new API key with these permissions:
     * Read accounts and balances
     * View market data
     * Place orders
     * Cancel orders
   - Download the JSON key file

3. **Configure test credentials**

   Add to your `.env` file (NOT `.env.example`):

   ```bash
   # Option 1: Key file (recommended for Ed25519 keys)
   COINBASE_TEST_KEY_FILE=/path/to/test_account_key.json

   # Option 2: Direct credentials (if not using key file)
   COINBASE_TEST_API_KEY=organizations/{org_id}/apiKeys/{key_id}
   COINBASE_TEST_API_SECRET="-----BEGIN EC PRIVATE KEY-----\n...\n-----END EC PRIVATE KEY-----"

   # Optional: Override minimum balance requirement (default: 50)
   COINBASE_TEST_MIN_BALANCE_USD=50.00
   ```

### Running Live Tests

**Run all live tests** (read-only + order validation):
```bash
pytest tests/test_coinbase_integration_live.py --run-live-tests -v
```

**Run read-only tests only** (no orders, safe):
```bash
pytest tests/test_coinbase_integration_live.py::TestCoinbaseReadOnly --run-live-tests -v
```

**Run order validation tests only** (places real orders):
```bash
pytest tests/test_coinbase_integration_live.py::TestCoinbaseOrderValidation --run-live-tests -v
```

**Run without flag** (all tests skip):
```bash
pytest tests/test_coinbase_integration_live.py -v
# Expected: All tests skipped with "requires --run-live-tests flag"
```

### What the Tests Do

#### Read-Only Tests (7 tests, safe)
- ✅ Verify authentication works (JWT building)
- ✅ Get accounts and balances (USD, BTC)
- ✅ Fetch current price (validates reasonable range)
- ✅ Get market data (bid/ask/volume, validates ask ≥ bid)
- ✅ Retrieve candles (OHLCV data, validates high ≥ low, etc.)

**Cost**: $0 (no orders)

#### Order Validation Tests (6 tests, real orders)
- 🟡 **Limit buy below market** - Places $10 limit buy slightly below bid, cancels after 100ms
- 🟡 **Market buy** - Buys $10 of BTC via market order (executes immediately)
- 🟡 **Limit sell above market** - Places limit sell slightly above ask, cancels after 100ms
- 🟡 **Market sell** - Sells $10 worth of BTC via market order (executes immediately)
- 🟡 **Ensure BTC balance** - If BTC balance is zero, buys $10 BTC for sell tests
- 🟡 **Invalid parameters** - Verifies orders with bad params are rejected

**Cost**: ~$0.30-0.60 per test run (fees only, capital is reused via buy/sell cycles)

### Safety Features

1. **Tests skip by default** - Must provide `--run-live-tests` flag
2. **Credential check** - Tests skip if no test credentials configured
3. **Balance check** - Order tests skip if balance < $50 USD
4. **Production check** - Warns if production credentials are set
5. **Order cleanup** - Fixture tracks and cancels unfilled orders after each test
6. **Amount validation** - All orders capped at $10 (Coinbase minimum)
7. **Isolated test account** - Documentation emphasizes separate account

### Risk Assessment

| Risk Type | Level | Details |
|-----------|-------|---------|
| Financial | MODERATE | $10 per order × 4 orders = $40 max, ~$0.30-0.60 in fees |
| Security | LOW | Separate test account with minimal funds |
| API | LOW | Manual execution only (not in CI) |

**Worst case scenario**: 3 executed orders + 1 cancelled = ~$30.30-30.60 total

**Best case scenario**: Buy/sell cycles net to ~$0.30-0.60 in fees only

### Minimum Balance Requirements

- **Order validation tests**: $50 USD minimum ($100 recommended)
- **Read-only tests**: No minimum (works with $0 balance)

If balance falls below $50 USD, order validation tests will skip automatically.

### Troubleshooting

**Tests skip with "Test credentials not configured"**
- Check that `COINBASE_TEST_KEY_FILE` or `COINBASE_TEST_API_KEY/SECRET` is set in `.env`
- Verify the key file path is correct and file exists
- Ensure the JSON file is valid (no syntax errors)

**Tests skip with "Insufficient balance"**
- Check balance: Account must have at least $50 USD
- Add more funds to test account
- Or override minimum: Set `COINBASE_TEST_MIN_BALANCE_USD=10.00` in `.env`

**Authentication failed**
- Verify API key has correct permissions (read accounts, place orders, cancel orders)
- Check that API key is not expired
- Ensure JSON key file contains valid Ed25519 private key
- Try regenerating API credentials

**Order cancellation failed**
- Market orders fill instantly and can't be cancelled (expected behavior)
- Check order book depth - limit orders may have filled before cancellation
- This is not a test failure - cleanup fixture handles this gracefully

**Rate limit errors**
- Coinbase has rate limits on API calls
- Wait 60 seconds between test runs
- Avoid running tests in parallel

**Tests pass but balance decreased significantly**
- Check that market orders are executing (expected for 2 tests)
- Verify buy/sell cycle completed (should net to ~fees only)
- Review test logs for order execution details
- Expected: $0.30-0.60 fee per full test run

### Best Practices

1. **Separate test account** - Never use production credentials
2. **Minimal funds** - Keep only $100-200 in test account
3. **Monitor balance** - Check balance before/after test runs
4. **Review logs** - Check structlog output for order details
5. **Run selectively** - Use read-only tests for frequent validation
6. **Manual only** - Never run in CI/CD (costs money, requires credentials)
7. **Document costs** - Track fees to understand test overhead

### CI/CD Configuration

**IMPORTANT**: Live integration tests are excluded from CI/CD pipelines.

These tests require:
- Real API credentials (security risk in CI)
- Real money (cost per run)
- Manual oversight (order validation)

Do NOT add `--run-live-tests` flag to CI workflows.

## Current Coverage

### Batch 1: Critical Safety Systems
| Module | Tests | Coverage | Status |
|--------|-------|----------|--------|
| circuit_breaker.py | 40 | 100% | ✅ Complete |
| loss_limiter.py | 52 | 98% | ✅ Complete |
| kill_switch.py | 35 | 97% | ✅ Complete |
| validator.py | 47 | 98% | ✅ Complete |
| **Batch 1 Total** | **174** | **~99%** | **✅ DONE** |

### Batch 2: Strategy Modules
| Module | Tests | Coverage | Status |
|--------|-------|----------|--------|
| signal_scorer.py | 60 | 87% | ✅ Complete |
| position_sizer.py | 40 | 99% | ✅ Complete |
| regime.py | 35 | 99% | ✅ Complete |
| **Batch 2 Total** | **135** | **~95%** | **✅ DONE** |

**Combined Total**: 309 tests | 100% passing | Overall coverage: 17%

### Batch 3: Technical Indicators - Coming Next
| Module | Tests | Coverage | Status |
|--------|-------|----------|--------|
| rsi.py | TBD | - | Pending |
| macd.py | TBD | - | Pending |
| bollinger.py | TBD | - | Pending |
| ema.py | TBD | - | Pending |
| atr.py | TBD | - | Pending |

## Adding New Tests

1. Create test file: `tests/test_<module>.py`
2. Import fixtures from `conftest.py`
3. Use descriptive test names: `test_<what>_<when>_<expected>`
4. Group related tests with comments
5. Add docstrings explaining test purpose
6. Use `freezegun` for time-based tests
7. Mock external dependencies
