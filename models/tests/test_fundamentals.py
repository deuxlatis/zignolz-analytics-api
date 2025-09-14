import json
import os
import tempfile
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from models.sp500.fundamentals import FundamentalStockAnalyzer


@pytest.fixture
def temp_cache_file():
    """Create a temporary cache file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_path = f.name
    yield temp_path
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def sample_tickers():
    """Sample S&P 500 tickers for testing."""
    return ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]


@pytest.fixture
def mock_wikipedia_response():
    """Mock HTML response from Wikipedia."""
    html_content = """
    <table>
        <thead>
            <tr><th>Symbol</th><th>Security</th></tr>
        </thead>
        <tbody>
            <tr><td>AAPL</td><td>Apple Inc.</td></tr>
            <tr><td>MSFT</td><td>Microsoft Corporation</td></tr>
            <tr><td>GOOGL</td><td>Alphabet Inc. Class A</td></tr>
            <tr><td>AMZN</td><td>Amazon.com Inc.</td></tr>
            <tr><td>META</td><td>Meta Platforms Inc.</td></tr>
        </tbody>
    </table>
    """
    mock_response = Mock()
    mock_response.text = html_content
    return mock_response


class TestFundamentalStockAnalyzer:
    """Test suite for FundamentalStockAnalyzer class."""

    def test_init_creates_analyzer_with_cache_file(
        self, temp_cache_file, sample_tickers, mock_wikipedia_response
    ):
        """Test that analyzer initializes with custom cache file."""
        # Ensure cache file doesn't exist (since temp_cache_file creates an empty file)
        if os.path.exists(temp_cache_file):
            os.unlink(temp_cache_file)

        with patch("requests.get") as mock_get:
            mock_get.return_value = mock_wikipedia_response

            with patch("pandas.read_html") as mock_read_html:
                mock_df = pd.DataFrame({"Symbol": sample_tickers})
                mock_read_html.return_value = [mock_df]

                analyzer = FundamentalStockAnalyzer(cache_file=temp_cache_file)
                assert analyzer.cache_file == temp_cache_file
                assert isinstance(analyzer.tickers, list)
                assert analyzer.tickers == sample_tickers

    def test_get_sp500_tickers_with_fresh_cache(self, temp_cache_file, sample_tickers):
        """Test loading tickers from fresh cache file."""
        # Create a fresh cache file
        with open(temp_cache_file, "w") as f:
            json.dump(sample_tickers, f)

        analyzer = FundamentalStockAnalyzer(cache_file=temp_cache_file)

        assert analyzer.tickers == sample_tickers
        assert len(analyzer.tickers) == 5

    def test_get_sp500_tickers_with_stale_cache(
        self, temp_cache_file, sample_tickers, mock_wikipedia_response
    ):
        """Test fetching new tickers when cache is stale."""
        # Create a stale cache file (31 days old)
        with open(temp_cache_file, "w") as f:
            json.dump(["OLD_TICKER"], f)

        # Set file modification time to 31 days ago
        old_time = (datetime.now() - timedelta(days=31)).timestamp()
        os.utime(temp_cache_file, (old_time, old_time))

        with patch("requests.get") as mock_get:
            mock_get.return_value = mock_wikipedia_response

            with patch("pandas.read_html") as mock_read_html:
                mock_df = pd.DataFrame({"Symbol": sample_tickers})
                mock_read_html.return_value = [mock_df]

                analyzer = FundamentalStockAnalyzer(cache_file=temp_cache_file)

                assert analyzer.tickers == sample_tickers
                assert len(analyzer.tickers) == 5
                mock_get.assert_called_once()

                # Verify cache was updated
                with open(temp_cache_file, "r") as f:
                    cached_data = json.load(f)
                assert cached_data == sample_tickers

    def test_get_sp500_tickers_without_cache(
        self, temp_cache_file, sample_tickers, mock_wikipedia_response
    ):
        """Test fetching tickers when cache file doesn't exist."""
        # Ensure cache file doesn't exist
        if os.path.exists(temp_cache_file):
            os.unlink(temp_cache_file)

        with patch("requests.get") as mock_get:
            mock_get.return_value = mock_wikipedia_response

            with patch("pandas.read_html") as mock_read_html:
                mock_df = pd.DataFrame({"Symbol": sample_tickers})
                mock_read_html.return_value = [mock_df]

                analyzer = FundamentalStockAnalyzer(cache_file=temp_cache_file)

                assert analyzer.tickers == sample_tickers
                assert len(analyzer.tickers) == 5
                mock_get.assert_called_once()

                # Verify cache file was created
                assert os.path.exists(temp_cache_file)
                with open(temp_cache_file, "r") as f:
                    cached_data = json.load(f)
                assert cached_data == sample_tickers

    def test_get_sp500_tickers_cache_within_threshold(
        self, temp_cache_file, sample_tickers
    ):
        """Test that cache is used when file is within the cache threshold."""
        # Create a cache file that's 15 days old (within 30-day threshold)
        with open(temp_cache_file, "w") as f:
            json.dump(sample_tickers, f)

        # Set file modification time to 15 days ago
        recent_time = (datetime.now() - timedelta(days=15)).timestamp()
        os.utime(temp_cache_file, (recent_time, recent_time))

        with patch("requests.get") as mock_get:
            analyzer = FundamentalStockAnalyzer(cache_file=temp_cache_file)

            assert analyzer.tickers == sample_tickers
            # Should not make any HTTP requests
            mock_get.assert_not_called()

    def test_get_sp500_tickers_handles_request_error(self, temp_cache_file):
        """Test error handling when Wikipedia request fails."""
        # Ensure cache file doesn't exist
        if os.path.exists(temp_cache_file):
            os.unlink(temp_cache_file)

        with patch("requests.get") as mock_get:
            mock_get.side_effect = Exception("Network error")

            with pytest.raises(Exception) as exc_info:
                FundamentalStockAnalyzer(cache_file=temp_cache_file)

            assert "Network error" in str(exc_info.value)

    def test_get_sp500_tickers_handles_parsing_error(
        self, temp_cache_file, mock_wikipedia_response
    ):
        """Test error handling when HTML parsing fails."""
        # Ensure cache file doesn't exist
        if os.path.exists(temp_cache_file):
            os.unlink(temp_cache_file)

        with patch("requests.get") as mock_get:
            mock_get.return_value = mock_wikipedia_response

            with patch("pandas.read_html") as mock_read_html:
                mock_read_html.side_effect = ValueError("Parsing error")

                with pytest.raises(ValueError) as exc_info:
                    FundamentalStockAnalyzer(cache_file=temp_cache_file)

                assert "Parsing error" in str(exc_info.value)

    def test_cache_file_permissions(self, temp_cache_file, sample_tickers):
        """Test that cache file is created with proper permissions."""
        # Ensure cache file doesn't exist
        if os.path.exists(temp_cache_file):
            os.unlink(temp_cache_file)

        with patch("requests.get") as mock_get:
            mock_get.return_value = Mock(text="<table></table>")

            with patch("pandas.read_html") as mock_read_html:
                mock_df = pd.DataFrame({"Symbol": sample_tickers})
                mock_read_html.return_value = [mock_df]

                FundamentalStockAnalyzer(cache_file=temp_cache_file)

                # Check file was created and is readable
                assert os.path.exists(temp_cache_file)
                assert os.access(temp_cache_file, os.R_OK)
                assert os.access(temp_cache_file, os.W_OK)

    @patch("models.sp500.fundamentals.logger")
    def test_logging_fresh_cache(self, mock_logger, temp_cache_file, sample_tickers):
        """Test that appropriate log messages are generated for fresh cache."""
        with open(temp_cache_file, "w") as f:
            json.dump(sample_tickers, f)

        FundamentalStockAnalyzer(cache_file=temp_cache_file)

        # Check that debug and info messages were logged
        mock_logger.debug.assert_called()
        mock_logger.info.assert_called()

    @patch("models.sp500.fundamentals.logger")
    def test_logging_stale_cache(
        self, mock_logger, temp_cache_file, sample_tickers, mock_wikipedia_response
    ):
        """Test that warning is logged for stale cache."""
        with open(temp_cache_file, "w") as f:
            json.dump(["OLD"], f)

        # Make cache stale
        old_time = (datetime.now() - timedelta(days=31)).timestamp()
        os.utime(temp_cache_file, (old_time, old_time))

        with patch("requests.get") as mock_get:
            mock_get.return_value = mock_wikipedia_response

            with patch("pandas.read_html") as mock_read_html:
                mock_df = pd.DataFrame({"Symbol": sample_tickers})
                mock_read_html.return_value = [mock_df]

                FundamentalStockAnalyzer(cache_file=temp_cache_file)

                # Check that warning was logged
                mock_logger.warning.assert_called()
