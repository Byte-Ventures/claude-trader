// Claude Trader Dashboard JavaScript

let chart = null;
let candleSeries = null;
let performanceChart = null;
let portfolioSeries = null;
let btcSeries = null;
let ws = null;
let reconnectAttempts = 0;
let seenNotificationIds = new Set();
const MAX_RECONNECT_ATTEMPTS = 10;
const RECONNECT_DELAY = 3000;
const MAX_SEEN_NOTIFICATIONS = 100;  // Prevent memory leak from unbounded Set

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
}

// Load initial data from REST API
async function loadInitialData() {
    try {
        // Load candles
        const candlesResponse = await fetch('/api/candles?limit=100');
        if (candlesResponse.ok) {
            const candles = await candlesResponse.json();
            const chartData = candles.map(c => ({
                time: Math.floor(new Date(c.timestamp).getTime() / 1000),
                open: parseFloat(c.open),
                high: parseFloat(c.high),
                low: parseFloat(c.low),
                close: parseFloat(c.close),
            }));
            candleSeries.setData(chartData);
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

        // Load config
        const configResponse = await fetch('/api/config');
        if (configResponse.ok) {
            const config = await configResponse.json();
            updateConfig(config);
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

        // Attempt to reconnect
        if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
            reconnectAttempts++;
            console.log(`Reconnecting... (attempt ${reconnectAttempts})`);
            setTimeout(connectWebSocket, RECONNECT_DELAY);
        }
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
}

// Update dashboard with new state
function updateDashboard(state) {
    // Update price
    const priceEl = document.getElementById('current-price');
    const price = parseFloat(state.price);
    priceEl.textContent = formatCurrency(price);

    // Update trading pair
    document.getElementById('trading-pair').textContent = state.trading_pair;

    // Update signal
    const signalEl = document.getElementById('signal-score');
    const score = state.signal.score;
    signalEl.textContent = score > 0 ? `+${score}` : score;
    signalEl.className = 'metric-value ' + (score > 0 ? 'positive' : score < 0 ? 'negative' : 'neutral');

    document.getElementById('signal-threshold').textContent = `Threshold: ${state.signal.threshold}`;

    // Update indicators
    const indicators = state.indicators;
    document.getElementById('volatility-value').textContent = indicators.volatility ?
        indicators.volatility.toUpperCase() : '--';

    // Update regime
    document.getElementById('regime-value').textContent = state.regime || '--';

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

    // Update signal breakdown bars
    const breakdown = state.signal.breakdown;
    updateBreakdownBar('breakdown-rsi', breakdown.rsi || 0);
    updateBreakdownBar('breakdown-macd', breakdown.macd || 0);
    updateBreakdownBar('breakdown-bollinger', breakdown.bollinger || 0);
    updateBreakdownBar('breakdown-ema', breakdown.ema || 0);
    updateBreakdownBar('breakdown-volume', breakdown.volume || 0);

    // Update last update time
    document.getElementById('last-update').textContent = `Last update: ${new Date(state.timestamp).toLocaleTimeString()}`;

    // Update chart with new price (if we have a timestamp)
    if (state.timestamp && candleSeries) {
        const time = Math.floor(new Date(state.timestamp).getTime() / 1000);
        candleSeries.update({
            time: time,
            open: price,
            high: price,
            low: price,
            close: price,
        });
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
    document.getElementById('trading-pair').textContent = config.trading_pair;
    document.getElementById('signal-threshold').textContent = `Threshold: ${config.signal_threshold}`;
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

        return `
            <tr>
                <td>${new Date(trade.executed_at).toLocaleString()}</td>
                <td class="trade-${trade.side}">${trade.side.toUpperCase()}</td>
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
                <span class="notification-time">${new Date(n.created_at).toLocaleString()}</span>
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
    if (!performanceChart || !data || data.length === 0) return;

    // Calculate cumulative returns (use ending values consistently)
    const firstDay = data[0];
    const startBalance = parseFloat(firstDay.ending_balance) || parseFloat(firstDay.starting_balance);
    const startPrice = parseFloat(firstDay.ending_price) || parseFloat(firstDay.starting_price);

    if (!startBalance || !startPrice) return;

    const portfolioData = [];
    const btcData = [];

    data.forEach(d => {
        // Use date string format for daily charts (YYYY-MM-DD)
        const time = d.date;
        const balance = parseFloat(d.ending_balance) || parseFloat(d.starting_balance);
        const price = parseFloat(d.ending_price) || parseFloat(d.starting_price);

        if (balance && price) {
            // Calculate percentage return from start
            const portfolioReturn = ((balance - startBalance) / startBalance) * 100;
            const btcReturn = ((price - startPrice) / startPrice) * 100;

            portfolioData.push({ time, value: portfolioReturn });
            btcData.push({ time, value: btcReturn });
        }
    });

    if (portfolioData.length > 0) {
        portfolioSeries.setData(portfolioData);
        btcSeries.setData(btcData);
    }
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
