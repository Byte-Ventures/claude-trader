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
