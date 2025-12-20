// Claude Trader Dashboard JavaScript

let chart = null;
let candleSeries = null;
let priceLine = null;
let performanceChart = null;
let portfolioSeries = null;
let btcSeries = null;
let cramerSeries = null;
let ws = null;
let reconnectAttempts = 0;
let seenNotificationIds = new Set();
/**
 * Current candle being built from real-time WebSocket updates.
 * Tracks OHLC state within a single candle period:
 * - open: First price when candle started (immutable within period)
 * - high: Maximum price seen (monotonically increasing)
 * - low: Minimum price seen (monotonically decreasing)
 * - close: Latest price (always updated)
 * - time: Unix timestamp of candle start, aligned to interval boundary
 * @type {{time: number, open: number, high: number, low: number, close: number}|null}
 */
let currentCandle = null;

/** Candle interval in seconds, loaded from backend config */
let candleIntervalSeconds = 60;

/** Flag to defer WebSocket candle updates until initial data loads */
let isInitialized = false;

/** Timestamp of last candle update, for throttling */
let lastCandleUpdate = 0;

/** Pending candle update timeout, ensures final throttled update is applied */
let pendingCandleUpdate = null;

/** Timestamp of last stale candle warning, for throttling console spam */
let lastStaleWarning = 0;

const MAX_RECONNECT_ATTEMPTS = 10;
const CANDLE_UPDATE_THROTTLE_MS = 1000;  // Throttle candle updates to 1/second
const STALE_WARNING_THROTTLE_MS = 60000;  // Throttle stale candle warnings to 1/minute
const BASE_RECONNECT_DELAY = 1000;
const MAX_RECONNECT_DELAY = 30000;
const MAX_SEEN_NOTIFICATIONS = 100;  // Prevent memory leak from unbounded Set

/**
 * Convert candle interval string to seconds.
 *
 * Supports two formats:
 * - Backend format: "ONE_MINUTE", "FIFTEEN_MINUTE", "ONE_HOUR", etc.
 * - Short format: "1m", "15m", "1h", etc.
 *
 * @param {string} interval - Interval string from backend config
 * @returns {number} Interval duration in seconds (defaults to 60 if unknown)
 */
function parseIntervalToSeconds(interval) {
    if (!interval) {
        console.warn('parseIntervalToSeconds: null/undefined interval, defaulting to 60s');
        return 60;
    }
    const intervalMap = {
        'ONE_MINUTE': 60,
        'FIVE_MINUTE': 300,
        'FIFTEEN_MINUTE': 900,
        'THIRTY_MINUTE': 1800,
        'ONE_HOUR': 3600,
        'TWO_HOUR': 7200,
        'SIX_HOUR': 21600,
        'ONE_DAY': 86400,
        '1m': 60,
        '5m': 300,
        '15m': 900,
        '30m': 1800,
        '1h': 3600,
        '2h': 7200,
        '6h': 21600,
        '1d': 86400,
    };
    const seconds = intervalMap[interval];
    if (!seconds) {
        console.warn(`parseIntervalToSeconds: unknown interval "${interval}", defaulting to 60s`);
        return 60;
    }
    return seconds;
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    initChart();
    initPerformanceChart();
    loadInitialData();
    connectWebSocket();
});

// Initialize TradingView Lightweight Chart
function initChart() {
    const container = document.getElementById('chart-container');

    chart = LightweightCharts.createChart(container, {
        width: container.clientWidth,
        height: container.clientHeight,
        layout: {
            background: { color: '#1f2937' },
            textColor: '#9ca3af',
        },
        grid: {
            vertLines: { color: '#374151' },
            horzLines: { color: '#374151' },
        },
        crosshair: {
            mode: LightweightCharts.CrosshairMode.Normal,
        },
        rightPriceScale: {
            borderColor: '#374151',
            scaleMargins: {
                top: 0.05,
                bottom: 0.05,
            },
        },
        timeScale: {
            borderColor: '#374151',
            timeVisible: true,
            secondsVisible: false,
            barSpacing: 12,
        },
        localization: {
            timeFormatter: (timestamp) => {
                const date = new Date(timestamp * 1000);
                return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            },
            dateFormatter: (timestamp) => {
                const date = new Date(timestamp * 1000);
                return date.toLocaleDateString();
            },
        },
    });

    // Add price line series FIRST (renders behind candles)
    priceLine = chart.addLineSeries({
        color: 'rgba(16, 185, 129, 0.6)',  // Default green, updated with data
        lineWidth: 2,
        crosshairMarkerVisible: false,
        lastValueVisible: false,
        priceLineVisible: false,
    });

    // Add candlestick series SECOND (renders in front)
    candleSeries = chart.addCandlestickSeries({
        upColor: '#10b981',
        downColor: '#ef4444',
        borderDownColor: '#ef4444',
        borderUpColor: '#10b981',
        wickDownColor: '#ef4444',
        wickUpColor: '#10b981',
    });

    // Handle window resize
    window.addEventListener('resize', () => {
        chart.applyOptions({
            width: container.clientWidth,
            height: container.clientHeight
        });
        if (performanceChart) {
            const perfContainer = document.getElementById('performance-chart');
            performanceChart.applyOptions({ width: perfContainer.clientWidth });
        }
    });
}

// Get price line color based on price direction
function getPriceLineColor(data) {
    if (!data || data.length < 2) return 'rgba(16, 185, 129, 0.6)';

    const firstPrice = data[0].close;
    const lastPrice = data[data.length - 1].close;

    // Green if price went up or stayed same, red if went down
    return firstPrice <= lastPrice
        ? 'rgba(16, 185, 129, 0.6)'   // Green (up or same)
        : 'rgba(239, 68, 68, 0.6)';   // Red (down)
}

// Initialize Performance Chart
function initPerformanceChart() {
    const container = document.getElementById('performance-chart');
    if (!container) return;

    performanceChart = LightweightCharts.createChart(container, {
        width: container.clientWidth,
        height: 250,
        layout: {
            background: { color: '#1f2937' },
            textColor: '#9ca3af',
        },
        grid: {
            vertLines: { color: '#374151' },
            horzLines: { color: '#374151' },
        },
        rightPriceScale: {
            borderColor: '#374151',
            scaleMargins: { top: 0.1, bottom: 0.1 },
        },
        timeScale: {
            borderColor: '#374151',
        },
        localization: {
            dateFormatter: (timestamp) => {
                const date = new Date(timestamp * 1000);
                return date.toLocaleDateString();
            },
        },
    });

    // Portfolio performance line (green)
    portfolioSeries = performanceChart.addLineSeries({
        color: '#10b981',
        lineWidth: 2,
        title: 'Portfolio',
    });

    // BTC performance line (yellow)
    btcSeries = performanceChart.addLineSeries({
        color: '#f59e0b',
        lineWidth: 2,
        title: 'BTC',
    });

    // Cramer Mode performance line (red) - hidden until data available
    cramerSeries = performanceChart.addLineSeries({
        color: '#ef4444',
        lineWidth: 2,
        title: 'Cramer',
        visible: false,
    });
}

// Load initial data from REST API
async function loadInitialData() {
    // Reset flag during reload (e.g., on WebSocket reconnect)
    isInitialized = false;

    try {
        // Load config FIRST - CRITICAL for candleIntervalSeconds before any candle processing
        const configResponse = await fetch('/api/config');
        if (configResponse.ok) {
            const config = await configResponse.json();
            updateConfig(config);
        } else {
            console.error('Failed to load config - candle bucketing will use default 60s interval');
        }

        // Load candles (now using correct candleIntervalSeconds from config)
        const candlesResponse = await fetch('/api/candles?limit=100');
        if (candlesResponse.ok) {
            const candles = await candlesResponse.json();
            const chartData = candles
                .map(c => {
                    // Parse timestamp - handle both ISO format and empty strings
                    const timestamp = c.timestamp ? new Date(c.timestamp + 'Z').getTime() : NaN;
                    const time = Math.floor(timestamp / 1000);
                    const open = parseFloat(c.open);
                    const high = parseFloat(c.high);
                    const low = parseFloat(c.low);
                    const close = parseFloat(c.close);
                    return { time, open, high, low, close };
                })
                .filter(c => {
                    // Filter out invalid candles (NaN values crash the chart)
                    return !isNaN(c.time) && c.time > 0 &&
                           !isNaN(c.open) && !isNaN(c.high) &&
                           !isNaN(c.low) && !isNaN(c.close) &&
                           c.open !== null && c.high !== null &&
                           c.low !== null && c.close !== null;
                });
            if (chartData.length > 0) {
                candleSeries.setData(chartData);

                // Initialize currentCandle from the last loaded candle
                const lastCandle = chartData[chartData.length - 1];
                currentCandle = { ...lastCandle };

                // Set price line data and color based on price direction
                const lineData = chartData.map(c => ({ time: c.time, value: c.close }));
                priceLine.setData(lineData);
                priceLine.applyOptions({ color: getPriceLineColor(chartData) });
            }
        }

        // Load trades
        const tradesResponse = await fetch('/api/trades?limit=20');
        if (tradesResponse.ok) {
            const trades = await tradesResponse.json();
            updateTradesTable(trades);
            updateLastTradeTime(trades);
        }

        // Load daily stats
        const statsResponse = await fetch('/api/stats');
        if (statsResponse.ok) {
            const stats = await statsResponse.json();
            updateDailyStats(stats);
        }

        // Load current state
        const stateResponse = await fetch('/api/state');
        if (stateResponse.ok) {
            const state = await stateResponse.json();
            if (state) {
                updateDashboard(state);
            }
        }

        // Load notifications
        const notificationsResponse = await fetch('/api/notifications?limit=10');
        if (notificationsResponse.ok) {
            const notifications = await notificationsResponse.json();
            displayNotifications(notifications);
        }

        // Load performance data
        const performanceResponse = await fetch('/api/performance?days=30');
        if (performanceResponse.ok) {
            const performance = await performanceResponse.json();
            updatePerformanceChart(performance);
        }

        // Mark initialization complete - WebSocket updates can now update candles
        isInitialized = true;
    } catch (error) {
        console.error('Failed to load initial data:', error);
    }
}

// WebSocket connection
function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

    ws.onopen = () => {
        console.log('WebSocket connected');
        document.getElementById('connection-status').textContent = 'Connected';
        document.getElementById('connection-status').className = 'status connected';

        // On reconnect, reload all data to catch up on missed updates.
        // Race condition prevention: loadInitialData() sets isInitialized=false at start,
        // which causes updateDashboard() to skip chart updates. This prevents WebSocket
        // messages from corrupting currentCandle while REST APIs are still fetching.
        // Once all data is loaded, isInitialized=true allows chart updates again.
        if (reconnectAttempts > 0) {
            console.log('Reconnected - reloading data to catch up');
            loadInitialData();
        }
        reconnectAttempts = 0;
    };

    ws.onmessage = (event) => {
        const state = JSON.parse(event.data);
        updateDashboard(state);

        // Handle new notifications
        if (state.new_notifications && state.new_notifications.length > 0) {
            addNewNotifications(state.new_notifications);
        }
    };

    ws.onclose = () => {
        console.log('WebSocket disconnected');
        document.getElementById('connection-status').textContent = 'Disconnected';
        document.getElementById('connection-status').className = 'status disconnected';

        // Clean up pending candle update timeout to prevent stale updates
        if (pendingCandleUpdate) {
            clearTimeout(pendingCandleUpdate);
            pendingCandleUpdate = null;
        }

        // Attempt to reconnect with exponential backoff and jitter
        if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
            reconnectAttempts++;
            const backoff = Math.min(BASE_RECONNECT_DELAY * Math.pow(2, reconnectAttempts - 1), MAX_RECONNECT_DELAY);
            const jitter = Math.random() * 1000;  // 0-1s jitter to prevent thundering herd
            const delay = backoff + jitter;
            console.log(`Reconnecting in ${Math.round(delay)}ms... (attempt ${reconnectAttempts})`);
            setTimeout(connectWebSocket, delay);
        }
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
}

// Update dashboard with new state
function updateDashboard(state) {
    // Update signal
    const signalEl = document.getElementById('signal-score');
    const score = state.signal.score;
    signalEl.textContent = score > 0 ? `+${score}` : score;
    signalEl.className = 'metric-value ' + (score > 0 ? 'positive' : score < 0 ? 'negative' : 'neutral');

    document.getElementById('signal-threshold').textContent = `Threshold: |${state.signal.threshold}|`;

    // Update indicators
    const indicators = state.indicators;

    // Update weight profile
    const weightProfile = state.weight_profile;
    if (weightProfile) {
        const profileEmoji = {
            'trending': '📈',
            'ranging': '↔️',
            'volatile': '⚡',
            'default': '⚖️'
        };
        const emoji = profileEmoji[weightProfile.name] || '🔄';
        document.getElementById('weight-profile-value').textContent =
            `${emoji} ${weightProfile.name.charAt(0).toUpperCase() + weightProfile.name.slice(1)}`;
        const confidencePct = Math.round(weightProfile.confidence * 100);
        document.getElementById('weight-profile-confidence').textContent =
            confidencePct > 0 ? `${confidencePct}% confidence` : '--';
    } else {
        document.getElementById('weight-profile-value').textContent = 'Disabled';
        document.getElementById('weight-profile-confidence').textContent = '--';
    }

    // Update HTF bias
    const htfBiasCard = document.getElementById('htf-bias-card');
    const htfBias = state.htf_bias;
    if (htfBias) {
        htfBiasCard.style.display = 'block';

        const trendEmoji = {
            'bullish': '📈',
            'bearish': '📉',
            'neutral': '↔️'
        };

        const biasEmoji = htfBias.combined_bias === 'bullish' ? '✅' :
                         htfBias.combined_bias === 'bearish' ? '❌' :
                         '⚖️';

        // Defensive check: Pydantic guarantees combined_bias is non-null Literal["bullish", "bearish", "neutral"]
        // but we check anyway for runtime safety in JavaScript
        const biasText = htfBias.combined_bias
            ? htfBias.combined_bias.charAt(0).toUpperCase() + htfBias.combined_bias.slice(1)
            : 'Unknown';
        document.getElementById('htf-combined-bias').textContent = `${biasEmoji} ${biasText}`;

        const dailyEmoji = htfBias.daily_trend
            ? (trendEmoji[htfBias.daily_trend] || '↔️')
            : '↔️';

        // Only show 4H trend if available (when mtf_4h_enabled=true)
        if (htfBias.four_hour_trend) {
            const fourHourEmoji = trendEmoji[htfBias.four_hour_trend] || '↔️';
            document.getElementById('htf-trends').textContent =
                `Daily: ${dailyEmoji} | 4H: ${fourHourEmoji}`;
        } else {
            // Daily-only mode: hide 4H label for cleaner display
            document.getElementById('htf-trends').textContent = `Daily: ${dailyEmoji}`;
        }
    } else {
        htfBiasCard.style.display = 'none';
    }

    // Update circuit breaker
    const cbEl = document.getElementById('circuit-breaker');
    const cbLevel = state.safety.circuit_breaker;
    cbEl.textContent = cbLevel;
    cbEl.className = 'metric-value status-' + cbLevel.toLowerCase();
    document.getElementById('can-trade').textContent = state.safety.can_trade ? 'Can Trade' : 'Trading Halted';

    // Update portfolio
    document.getElementById('quote-balance').textContent = formatCurrency(parseFloat(state.portfolio.quote_balance));
    document.getElementById('base-balance').textContent = formatBTC(parseFloat(state.portfolio.base_balance));
    document.getElementById('portfolio-value').textContent = formatCurrency(parseFloat(state.portfolio.portfolio_value));
    document.getElementById('position-percent').textContent = state.portfolio.position_percent.toFixed(1) + '%';

    // Update Cramer portfolio (if available)
    const cramerCard = document.getElementById('cramer-portfolio-card');
    if (state.cramer_portfolio) {
        document.getElementById('cramer-quote-balance').textContent = formatCurrency(parseFloat(state.cramer_portfolio.quote_balance));
        document.getElementById('cramer-base-balance').textContent = formatBTC(parseFloat(state.cramer_portfolio.base_balance));
        document.getElementById('cramer-portfolio-value').textContent = formatCurrency(parseFloat(state.cramer_portfolio.portfolio_value));
        document.getElementById('cramer-position-percent').textContent = state.cramer_portfolio.position_percent.toFixed(1) + '%';
        cramerCard.style.display = 'block';
    } else {
        cramerCard.style.display = 'none';
    }

    // Update signal breakdown bars
    const breakdown = state.signal.breakdown;
    updateBreakdownBar('breakdown-rsi', breakdown.rsi || 0);
    updateBreakdownBar('breakdown-macd', breakdown.macd || 0);
    updateBreakdownBar('breakdown-bollinger', breakdown.bollinger || 0);
    updateBreakdownBar('breakdown-ema', breakdown.ema || 0);
    updateBreakdownBar('breakdown-volume', breakdown.volume || 0);

    // Update chart with new price (if we have valid data)
    //
    // Guards:
    // - Skip until initial data loads (prevents race condition with loadInitialData)
    // - Throttle to 1 update/second (prevents hammering chart library)
    //
    // Candle Bucketing Algorithm:
    // 1. Align timestamp to interval boundary: floor(time / interval) * interval
    //    Example: 23:47:15 with 15-min interval → 23:45:00
    // 2. If same bucket as currentCandle: update H/L/C, preserve O
    // 3. If newer bucket: start new candle with O=H=L=C=price
    // 4. If older bucket: skip (stale data from reconnect/lag)
    //
    // Throttling: Only throttle updates WITHIN the same candle period.
    // New candle periods always update immediately to ensure accurate open price.
    //
    // This matches exchange OHLC bucketing (Unix epoch aligned, 24/7 crypto markets).
    if (isInitialized && state.timestamp && candleSeries && !isNaN(price) && price > 0) {
        // Don't add 'Z' - state timestamps already have timezone info (+00:00)
        const time = Math.floor(new Date(state.timestamp).getTime() / 1000);
        if (!isNaN(time) && time > 0) {
            const candleTime = Math.floor(time / candleIntervalSeconds) * candleIntervalSeconds;
            const now = Date.now();
            const isNewCandlePeriod = !currentCandle || candleTime > currentCandle.time;
            const isThrottled = !isNewCandlePeriod &&
                (now - lastCandleUpdate < CANDLE_UPDATE_THROTTLE_MS);

            // Clear any pending trailing update (new data supersedes it)
            if (pendingCandleUpdate) {
                clearTimeout(pendingCandleUpdate);
                pendingCandleUpdate = null;
            }

            if (isThrottled) {
                // Schedule trailing update to ensure final price is captured
                // This prevents losing the close price if updates stop arriving
                // Capture current timestamp to detect stale updates after reconnect
                const scheduledAt = now;
                pendingCandleUpdate = setTimeout(() => {
                    pendingCandleUpdate = null;
                    // Staleness check: ensure this update is still relevant
                    // If WebSocket reconnected, lastCandleUpdate will have jumped ahead
                    const currentTime = Date.now();
                    const isStale = (currentTime - scheduledAt) > (CANDLE_UPDATE_THROTTLE_MS * 2);
                    if (isStale) {
                        return;  // Discard stale update from before reconnect
                    }
                    if (currentCandle && currentCandle.time === candleTime) {
                        currentCandle.high = Math.max(currentCandle.high, price);
                        currentCandle.low = Math.min(currentCandle.low, price);
                        currentCandle.close = price;
                        candleSeries.update(currentCandle);
                        if (priceLine) {
                            priceLine.update({ time: currentCandle.time, value: price });
                        }
                    }
                }, CANDLE_UPDATE_THROTTLE_MS);
                return;
            }

            lastCandleUpdate = now;

            if (currentCandle && currentCandle.time === candleTime) {
                // Same candle period - update high/low/close (open is preserved as first price)
                // Note: JS Math.max/min on floats may introduce minor precision errors (~1e-15).
                // This is acceptable for chart display; backend uses Decimal for exact values.
                // IMPORTANT: Use backend API for trading decisions, not dashboard display values
                currentCandle.high = Math.max(currentCandle.high, price);
                currentCandle.low = Math.min(currentCandle.low, price);
                currentCandle.close = price;
                candleSeries.update(currentCandle);
            } else if (isNewCandlePeriod) {
                // New candle period (must be newer) - start fresh
                currentCandle = {
                    time: candleTime,
                    open: price,
                    high: price,
                    low: price,
                    close: price,
                };
                candleSeries.update(currentCandle);
            } else {
                // candleTime < currentCandle.time - skip stale update to avoid chart error
                // This can happen due to WebSocket reconnect lag or minor clock skew between server/client
                // Throttle warnings to 1/minute to avoid console spam during network issues
                if (now - lastStaleWarning >= STALE_WARNING_THROTTLE_MS) {
                    console.warn(`Skipping stale candle update (reconnect lag or clock skew): new=${candleTime} (${new Date(candleTime * 1000).toISOString()}), current=${currentCandle.time} (${new Date(currentCandle.time * 1000).toISOString()}), interval=${candleIntervalSeconds}s`);
                    lastStaleWarning = now;
                }
            }

            // Update price line (only if we updated the candle)
            if (priceLine && currentCandle && candleTime >= currentCandle.time) {
                priceLine.update({ time: currentCandle.time, value: price });
            }
        }
    }

    // Update trades table if recent_trades included
    if (state.recent_trades) {
        updateTradesTable(state.recent_trades);
        updateLastTradeTime(state.recent_trades);
    }
}

// Update breakdown bar visualization
function updateBreakdownBar(id, value) {
    const bar = document.getElementById(id);
    const label = document.getElementById(id + '-val');

    // Max value per indicator: RSI/MACD=25, Bollinger=20, EMA=15, Volume=variable
    const maxVal = 25;
    const percentage = Math.min(Math.abs(value) / maxVal, 1) * 50; // 50% = half the bar

    // Update bar position and width
    bar.className = 'breakdown-bar ' + (value > 0 ? 'positive' : value < 0 ? 'negative' : '');
    bar.style.width = percentage + '%';

    // Update label
    if (label) {
        label.textContent = value > 0 ? `+${value}` : value;
        label.className = 'breakdown-value-label ' + (value > 0 ? 'positive' : value < 0 ? 'negative' : '');
    }
}

// Update config display
function updateConfig(config) {
    document.getElementById('signal-threshold').textContent = `Threshold: |${config.signal_threshold}|`;
    if (config.candle_interval) {
        candleIntervalSeconds = parseIntervalToSeconds(config.candle_interval);
    }
}

// Update trades table
function updateTradesTable(trades) {
    const tbody = document.getElementById('trades-body');

    if (!trades || trades.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" class="no-trades">No trades yet</td></tr>';
        return;
    }

    tbody.innerHTML = trades.map(trade => {
        const pnl = trade.realized_pnl ? parseFloat(trade.realized_pnl) : null;
        const pnlClass = pnl ? (pnl >= 0 ? 'pnl-positive' : 'pnl-negative') : '';
        const pnlText = pnl !== null ? formatCurrency(pnl) : '--';
        const isCramer = trade.bot_mode === 'inverted';
        const modeBadge = isCramer ? '<span class="trade-mode cramer">CRAMER</span>' : '';

        return `
            <tr>
                <td>${new Date(trade.executed_at + 'Z').toLocaleString()}</td>
                <td class="trade-${trade.side}">${trade.side.toUpperCase()}${modeBadge}</td>
                <td>${formatBTC(parseFloat(trade.size))}</td>
                <td>${formatCurrency(parseFloat(trade.price))}</td>
                <td class="${pnlClass}">${pnlText}</td>
            </tr>
        `;
    }).join('');
}

// Update daily stats display
function updateDailyStats(stats) {
    const pnlEl = document.getElementById('daily-pnl');
    const pnlPercentEl = document.getElementById('daily-pnl-percent');
    const tradesEl = document.getElementById('total-trades');

    if (!stats) {
        pnlEl.textContent = '$0.00';
        pnlEl.className = 'metric-value';
        pnlPercentEl.textContent = 'No data';
        tradesEl.textContent = '0';
        return;
    }

    const pnl = parseFloat(stats.realized_pnl) || 0;
    const startBalance = parseFloat(stats.starting_balance) || 0;
    const pnlPercent = startBalance > 0 ? (pnl / startBalance) * 100 : 0;

    pnlEl.textContent = formatCurrency(pnl);
    pnlEl.className = 'metric-value ' + (pnl >= 0 ? 'pnl-positive' : 'pnl-negative');
    pnlPercentEl.textContent = (pnlPercent >= 0 ? '+' : '') + pnlPercent.toFixed(2) + '%';
    tradesEl.textContent = stats.total_trades || '0';
}

// Update last trade time display
function updateLastTradeTime(trades) {
    const lastTradeEl = document.getElementById('last-trade-time');

    if (!trades || trades.length === 0) {
        lastTradeEl.textContent = 'No trades yet';
        return;
    }

    const lastTrade = trades[0]; // Most recent trade
    const tradeTime = new Date(lastTrade.executed_at);
    const now = new Date();
    const diffMs = now - tradeTime;
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const diffDays = Math.floor(diffHours / 24);

    if (diffDays > 0) {
        lastTradeEl.textContent = `Last: ${diffDays}d ago`;
    } else if (diffHours > 0) {
        lastTradeEl.textContent = `Last: ${diffHours}h ago`;
    } else {
        const diffMins = Math.floor(diffMs / (1000 * 60));
        lastTradeEl.textContent = `Last: ${diffMins}m ago`;
    }
}

// Display notifications
function displayNotifications(notifications) {
    const container = document.getElementById('notifications-container');

    if (!notifications || notifications.length === 0) {
        container.innerHTML = '<div class="no-notifications">No notifications yet</div>';
        return;
    }

    // Track seen IDs
    notifications.forEach(n => seenNotificationIds.add(n.id));

    container.innerHTML = notifications.map(n => createNotificationHTML(n, false)).join('');
}

// Add new notifications (from WebSocket)
function addNewNotifications(notifications) {
    const container = document.getElementById('notifications-container');
    const noNotifications = container.querySelector('.no-notifications');
    if (noNotifications) {
        noNotifications.remove();
    }

    notifications.forEach(n => {
        if (!seenNotificationIds.has(n.id)) {
            seenNotificationIds.add(n.id);

            // Prevent memory leak: clear oldest entries if Set gets too large
            if (seenNotificationIds.size > MAX_SEEN_NOTIFICATIONS) {
                const idsToKeep = Array.from(seenNotificationIds).slice(-MAX_SEEN_NOTIFICATIONS / 2);
                seenNotificationIds.clear();
                idsToKeep.forEach(id => seenNotificationIds.add(id));
            }

            const html = createNotificationHTML(n, true);
            container.insertAdjacentHTML('afterbegin', html);

            // Remove 'new' class after animation
            setTimeout(() => {
                const newItem = container.querySelector(`.notification-item[data-id="${n.id}"]`);
                if (newItem) {
                    newItem.classList.remove('new');
                }
            }, 5000);
        }
    });
}

// Create notification HTML
function createNotificationHTML(n, isNew) {
    // Strip HTML tags for display
    const fullMessage = n.message.replace(/<[^>]*>/g, '');
    const truncated = fullMessage.length > 200;
    const displayMessage = truncated ? fullMessage.substring(0, 200) + '...' : fullMessage;
    const expandClass = truncated ? 'expandable' : '';

    return `
        <div class="notification-item ${isNew ? 'new' : ''} ${expandClass}" data-id="${n.id}" data-full="${escapeHtml(fullMessage)}" data-truncated="${escapeHtml(displayMessage)}" onclick="toggleNotification(this)">
            <div class="notification-header">
                <div>
                    <span class="notification-type ${n.type}">${n.type.replace('_', ' ')}</span>
                    <span class="notification-title">${escapeHtml(n.title)}</span>
                </div>
                <span class="notification-time">${new Date(n.created_at + 'Z').toLocaleString()}</span>
            </div>
            <div class="notification-message">${escapeHtml(displayMessage)}</div>
            ${truncated ? '<div class="notification-expand-hint">Click to expand</div>' : ''}
        </div>
    `;
}

// Toggle notification expand/collapse
function toggleNotification(el) {
    if (!el.classList.contains('expandable')) return;

    const messageEl = el.querySelector('.notification-message');
    const hintEl = el.querySelector('.notification-expand-hint');
    const isExpanded = el.classList.contains('expanded');

    if (isExpanded) {
        messageEl.textContent = el.dataset.truncated;
        hintEl.textContent = 'Click to expand';
        el.classList.remove('expanded');
    } else {
        messageEl.textContent = el.dataset.full;
        hintEl.textContent = 'Click to collapse';
        el.classList.add('expanded');
    }
}

// Update performance chart
function updatePerformanceChart(data) {
    if (!performanceChart || !data) return;

    // Handle both old format (array) and new format (object with normal/cramer)
    const normalData = Array.isArray(data) ? data : (data.normal || []);
    const cramerData = Array.isArray(data) ? null : data.cramer;

    if (!normalData || normalData.length === 0) return;

    // Calculate cumulative returns (use ending values consistently)
    const firstDay = normalData[0];
    if (!firstDay) return;
    const startBalance = parseFloat(firstDay.ending_balance) || parseFloat(firstDay.starting_balance);
    const startPrice = parseFloat(firstDay.ending_price) || parseFloat(firstDay.starting_price);

    if (!startBalance || !startPrice) return;

    const portfolioDataPoints = [];
    const btcDataPoints = [];

    normalData.forEach(d => {
        // Use date string format for daily charts (YYYY-MM-DD)
        const time = d.date;
        const balance = parseFloat(d.ending_balance) || parseFloat(d.starting_balance);
        const price = parseFloat(d.ending_price) || parseFloat(d.starting_price);

        if (balance && price) {
            // Calculate percentage return from start
            const portfolioReturn = ((balance - startBalance) / startBalance) * 100;
            const btcReturn = ((price - startPrice) / startPrice) * 100;

            portfolioDataPoints.push({ time, value: portfolioReturn });
            btcDataPoints.push({ time, value: btcReturn });
        }
    });

    if (portfolioDataPoints.length > 0) {
        portfolioSeries.setData(portfolioDataPoints);
        btcSeries.setData(btcDataPoints);
    }

    // Handle Cramer data if available
    if (cramerData && cramerData.length > 0 && cramerSeries) {
        const cramerFirstDay = cramerData[0];
        const cramerStartBalance = parseFloat(cramerFirstDay.ending_balance) || parseFloat(cramerFirstDay.starting_balance);

        if (cramerStartBalance) {
            const cramerDataPoints = [];
            cramerData.forEach(d => {
                const time = d.date;
                const balance = parseFloat(d.ending_balance) || parseFloat(d.starting_balance);
                if (balance) {
                    const cramerReturn = ((balance - cramerStartBalance) / cramerStartBalance) * 100;
                    cramerDataPoints.push({ time, value: cramerReturn });
                }
            });

            if (cramerDataPoints.length > 0) {
                cramerSeries.setData(cramerDataPoints);
                cramerSeries.applyOptions({ visible: true });
                // Show Cramer legend
                const cramerLegend = document.getElementById('cramer-legend');
                if (cramerLegend) cramerLegend.style.display = 'flex';
            }
        }
    } else if (cramerSeries) {
        // Hide Cramer series and legend if no data
        cramerSeries.applyOptions({ visible: false });
        const cramerLegend = document.getElementById('cramer-legend');
        if (cramerLegend) cramerLegend.style.display = 'none';
    }

    // Fit chart to show all data points properly
    performanceChart.timeScale().fitContent();
}

// Escape HTML for safe display
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Format currency (USD/EUR)
function formatCurrency(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    }).format(value);
}

// Format BTC amount
function formatBTC(value) {
    return value.toFixed(8) + ' BTC';
}

// FAQ Modal
(function() {
    const modal = document.getElementById('faq-modal');
    const btn = document.getElementById('faq-btn');
    const close = modal.querySelector('.modal-close');

    btn.onclick = () => modal.classList.add('active');
    close.onclick = () => modal.classList.remove('active');
    modal.onclick = (e) => {
        if (e.target === modal) modal.classList.remove('active');
    };
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') modal.classList.remove('active');
    });
})();
