"""
Tests for the main entry point.

Tests cover:
- Live trading confirmation requirement
- Startup validation
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from config.settings import TradingMode


class TestLiveTradingConfirmation:
    """Tests for the live trading confirmation requirement."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for testing."""
        settings = Mock()
        settings.log_level = "INFO"
        settings.log_file = None
        settings.trading_pair = "BTC-USD"
        settings.check_interval_seconds = 60
        settings.position_size_percent = 25
        settings.signal_threshold = 50
        return settings

    def test_live_trading_rejected_without_confirmation(self, mock_settings):
        """CRITICAL: Live trading must be rejected if confirmation is not set."""
        mock_settings.is_paper_trading = False
        mock_settings.is_live_trading = True
        mock_settings.i_understand_that_i_will_lose_all_my_money = False

        with patch('src.main.get_settings', return_value=mock_settings):
            with patch('src.main.setup_logging'):
                with patch('src.main.get_logger') as mock_get_logger:
                    mock_logger = Mock()
                    mock_get_logger.return_value = mock_logger

                    from src.main import main
                    result = main()

                    # Must return 1 (error) when confirmation is missing
                    assert result == 1
                    # Must log critical message
                    mock_logger.critical.assert_called_once()

    def test_live_trading_allowed_with_confirmation(self, mock_settings):
        """Live trading proceeds when confirmation is explicitly set."""
        mock_settings.is_paper_trading = False
        mock_settings.is_live_trading = True
        mock_settings.i_understand_that_i_will_lose_all_my_money = True

        with patch('src.main.get_settings', return_value=mock_settings):
            with patch('src.main.setup_logging'):
                with patch('src.main.get_logger') as mock_get_logger:
                    mock_logger = Mock()
                    mock_get_logger.return_value = mock_logger
                    with patch('src.main.TradingDaemon') as mock_daemon:
                        # Make daemon.run() exit cleanly
                        mock_daemon_instance = Mock()
                        mock_daemon.return_value = mock_daemon_instance
                        with patch('time.sleep'):  # Skip the 5 second warning delay

                            from src.main import main
                            result = main()

                            # Should not return 1 (error)
                            assert result == 0
                            # Should not log critical about confirmation
                            for call in mock_logger.critical.call_args_list:
                                assert 'live_trading_not_confirmed' not in str(call)

    def test_paper_trading_does_not_require_confirmation(self, mock_settings):
        """Paper trading does not require the live trading confirmation."""
        mock_settings.is_paper_trading = True
        mock_settings.is_live_trading = False
        mock_settings.i_understand_that_i_will_lose_all_my_money = False

        with patch('src.main.get_settings', return_value=mock_settings):
            with patch('src.main.setup_logging'):
                with patch('src.main.get_logger') as mock_get_logger:
                    mock_logger = Mock()
                    mock_get_logger.return_value = mock_logger
                    with patch('src.main.TradingDaemon') as mock_daemon:
                        mock_daemon_instance = Mock()
                        mock_daemon.return_value = mock_daemon_instance

                        from src.main import main
                        result = main()

                        # Should succeed (return 0)
                        assert result == 0
                        # Should not log critical about confirmation
                        for call in mock_logger.critical.call_args_list:
                            assert 'live_trading_not_confirmed' not in str(call)
