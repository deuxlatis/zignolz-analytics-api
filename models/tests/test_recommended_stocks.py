"""Tests for the recommended_stocks method."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from models.sp500.fundamentals import FundamentalStockAnalyzer


class TestRecommendedStocks:
    """Test cases for the recommended_stocks method."""

    def test_meets_financial_criteria(self):
        """Test the _meets_financial_criteria helper method."""
        analyzer = FundamentalStockAnalyzer()

        # Test data that meets all criteria
        good_data = {
            "annualChangePercent": -30,
            "operatingCashflow": 500000000,
            "freeCashflow": 400000000,
            "ebitda": 1000000000,
            "profitMargins": 0.15,
            "revenueGrowth": -5,
            "ebitdaMargins": 0.20,
            "operatingMargins": 0.18,
            "returnOnAssets": 0.08,
        }

        assert analyzer._meets_financial_criteria(good_data) is True

        # Test data that fails annual change criteria
        bad_annual = good_data.copy()
        bad_annual["annualChangePercent"] = -20  # Not <= -25
        assert analyzer._meets_financial_criteria(bad_annual) is False

        # Test data that fails cash flow criteria
        bad_cash_flow = good_data.copy()
        bad_cash_flow["operatingCashflow"] = 200000000  # Not > 300M
        assert analyzer._meets_financial_criteria(bad_cash_flow) is False

        # Test data with missing field
        missing_field = good_data.copy()
        del missing_field["ebitda"]
        assert analyzer._meets_financial_criteria(missing_field) is False

    @pytest.mark.asyncio
    async def test_recommended_stocks_empty_list(self):
        """Test recommended_stocks with no qualifying stocks."""
        analyzer = await FundamentalStockAnalyzer.create()

        # Test with tickers that likely won't meet criteria
        stable_tickers = ["AAPL", "MSFT"]
        recommendations = await analyzer.recommended_stocks(tickers=stable_tickers)

        # Should return a list (might be empty)
        assert isinstance(recommendations, list)

    @pytest.mark.asyncio
    async def test_recommended_stocks_return_type(self):
        """Test that recommended_stocks returns a list of strings."""
        analyzer = await FundamentalStockAnalyzer.create()

        # Use a small subset for faster testing
        test_tickers = ["AAPL", "GOOGL", "TSLA"]
        recommendations = await analyzer.recommended_stocks(
            tickers=test_tickers, max_concurrent=2
        )

        assert isinstance(recommendations, list)
        # All items should be strings (tickers)
        for ticker in recommendations:
            assert isinstance(ticker, str)
            assert len(ticker) <= 5  # Valid ticker format

    def test_meets_financial_criteria_edge_cases(self):
        """Test edge cases for financial criteria."""
        analyzer = FundamentalStockAnalyzer()

        # Test with exactly boundary values
        boundary_data = {
            "annualChangePercent": -25,  # Exactly at boundary
            "operatingCashflow": 300000001,  # Just above boundary
            "freeCashflow": 300000001,  # Just above boundary
            "ebitda": 0.1,  # Just above 0
            "profitMargins": 0.001,  # Just above 0
            "revenueGrowth": -9.99,  # Just above -10
            "ebitdaMargins": 0.001,  # Just above 0
            "operatingMargins": 0.001,  # Just above 0
            "returnOnAssets": 0.001,  # Just above 0
        }

        assert analyzer._meets_financial_criteria(boundary_data) is True

        # Test with None values
        none_data = {key: None for key in boundary_data.keys()}
        assert analyzer._meets_financial_criteria(none_data) is False

    def test_meets_financial_criteria_all_required_fields(self):
        """Test that all required fields are checked in financial criteria."""
        analyzer = FundamentalStockAnalyzer()

        required_fields = [
            "annualChangePercent",
            "operatingCashflow",
            "freeCashflow",
            "ebitda",
            "profitMargins",
            "revenueGrowth",
            "ebitdaMargins",
            "operatingMargins",
            "returnOnAssets",
        ]

        base_data = {
            "annualChangePercent": -30,
            "operatingCashflow": 500000000,
            "freeCashflow": 400000000,
            "ebitda": 1000000000,
            "profitMargins": 0.15,
            "revenueGrowth": -5,
            "ebitdaMargins": 0.20,
            "operatingMargins": 0.18,
            "returnOnAssets": 0.08,
        }

        # Test that removing any required field causes failure
        for field in required_fields:
            test_data = base_data.copy()
            del test_data[field]
            assert analyzer._meets_financial_criteria(test_data) is False, (
                f"Should fail when {field} is missing"
            )

    def test_short_term_change_logic(self):
        """Test the logic for evaluating short-term changes."""
        # Test data for "signs of recovery" scenario
        signs_of_recovery_values = [
            -5.0,
            -2.0,
            1.5,
            -1.0,
        ]  # Exactly one positive (1.5%)
        in_range = [0 < v <= 3 for v in signs_of_recovery_values]
        others_nonpos = [
            v <= 0 for i, v in enumerate(signs_of_recovery_values) if not in_range[i]
        ]

        assert sum(in_range) == 1  # Exactly one in range
        assert all(others_nonpos)  # Others are non-positive

        # Test data for "full decline" scenario
        full_decline_values = [-5.0, -2.0, -1.5, -1.0]  # All negative
        all_nonpositive = all(v <= 0 for v in full_decline_values)

        assert all_nonpositive is True

        # Test data that should be rejected (too many positive)
        too_many_positive = [2.0, 1.5, -1.0, -2.0]  # Two positive values
        in_range_reject = [0 < v <= 3 for v in too_many_positive]
        assert sum(in_range_reject) == 2  # Should be rejected

        # Test data that should be rejected (positive value too high)
        too_high_positive = [5.0, -1.0, -2.0, -3.0]  # One positive but > 3%
        in_range_high = [0 < v <= 3 for v in too_high_positive]
        assert sum(in_range_high) == 0  # 5.0 is not in (0, 3] range

    @pytest.fixture
    def sample_cache_file(self, tmp_path):
        """Create a temporary cache file for testing."""
        cache_file = tmp_path / "test_cache.json"
        sample_tickers = ["TEST1", "TEST2", "TEST3"]
        with open(cache_file, "w") as f:
            json.dump(sample_tickers, f)
        return str(cache_file)

    @pytest.mark.asyncio
    async def test_recommended_stocks_integration_mock(self, sample_cache_file):
        """Test recommended_stocks with realistic mock data."""
        sample_tickers = ["RECOVERY1", "DECLINE1", "STABLE1", "ERROR1"]

        with patch("aiofiles.open", create=True) as mock_aiofiles:
            mock_file = AsyncMock()
            mock_file.read.return_value = json.dumps(sample_tickers)
            mock_aiofiles.return_value.__aenter__.return_value = mock_file
            mock_aiofiles.return_value.__aexit__.return_value = None

            analyzer = await FundamentalStockAnalyzer.create(
                cache_file=sample_cache_file
            )

            # Mock fundamental data for different scenarios
            mock_fundamentals = [
                {
                    "ticker": "RECOVERY1",
                    "data": {
                        # Meets financial criteria + signs of recovery
                        "annualChangePercent": -30.0,
                        "sixMonthChangePercent": -5.0,
                        "threeMonthChangePercent": 2.0,  # Only one positive (0-3%)
                        "oneMonthChangePercent": -1.0,
                        "oneWeekChangePercent": -0.5,
                        "operatingCashflow": 500000000,
                        "freeCashflow": 400000000,
                        "ebitda": 1000000000,
                        "profitMargins": 0.15,
                        "revenueGrowth": -5.0,
                        "ebitdaMargins": 0.20,
                        "operatingMargins": 0.18,
                        "returnOnAssets": 0.08,
                    },
                },
                {
                    "ticker": "DECLINE1",
                    "data": {
                        # Meets financial criteria + full decline
                        "annualChangePercent": -35.0,
                        "sixMonthChangePercent": -10.0,
                        "threeMonthChangePercent": -5.0,
                        "oneMonthChangePercent": -2.0,
                        "oneWeekChangePercent": -1.0,  # All negative
                        "operatingCashflow": 600000000,
                        "freeCashflow": 500000000,
                        "ebitda": 1200000000,
                        "profitMargins": 0.18,
                        "revenueGrowth": -3.0,
                        "ebitdaMargins": 0.22,
                        "operatingMargins": 0.20,
                        "returnOnAssets": 0.10,
                    },
                },
                {
                    "ticker": "STABLE1",
                    "data": {
                        # Doesn't meet criteria (not enough decline)
                        "annualChangePercent": -15.0,  # Not <= -25
                        "sixMonthChangePercent": -2.0,
                        "threeMonthChangePercent": -1.0,
                        "oneMonthChangePercent": 0.5,
                        "oneWeekChangePercent": -0.2,
                        "operatingCashflow": 500000000,
                        "freeCashflow": 400000000,
                        "ebitda": 1000000000,
                        "profitMargins": 0.15,
                        "revenueGrowth": -5.0,
                        "ebitdaMargins": 0.20,
                        "operatingMargins": 0.18,
                        "returnOnAssets": 0.08,
                    },
                },
                {
                    "ticker": "ERROR1",
                    "error": "Failed to fetch data",
                },
            ]

            with patch.object(
                analyzer, "get_fundamentals_concurrent", return_value=mock_fundamentals
            ):
                recommendations = await analyzer.recommended_stocks(
                    tickers=sample_tickers
                )

                # Should recommend RECOVERY1 and DECLINE1, but not STABLE1 or ERROR1
                assert len(recommendations) == 2
                assert "RECOVERY1" in recommendations
                assert "DECLINE1" in recommendations
                assert "STABLE1" not in recommendations
                assert "ERROR1" not in recommendations

    @pytest.mark.asyncio
    async def test_recommended_stocks_missing_change_data(self, sample_cache_file):
        """Test recommended_stocks when change data is missing."""
        sample_tickers = ["MISSING1"]

        with patch("aiofiles.open", create=True) as mock_aiofiles:
            mock_file = AsyncMock()
            mock_file.read.return_value = json.dumps(sample_tickers)
            mock_aiofiles.return_value.__aenter__.return_value = mock_file
            mock_aiofiles.return_value.__aexit__.return_value = None

            analyzer = await FundamentalStockAnalyzer.create(
                cache_file=sample_cache_file
            )

            # Mock data with missing change fields
            mock_fundamentals = [
                {
                    "ticker": "MISSING1",
                    "data": {
                        "annualChangePercent": -30.0,
                        "sixMonthChangePercent": None,  # Missing data
                        "threeMonthChangePercent": -2.0,
                        "oneMonthChangePercent": -1.0,
                        "oneWeekChangePercent": -0.5,
                        "operatingCashflow": 500000000,
                        "freeCashflow": 400000000,
                        "ebitda": 1000000000,
                        "profitMargins": 0.15,
                        "revenueGrowth": -5.0,
                        "ebitdaMargins": 0.20,
                        "operatingMargins": 0.18,
                        "returnOnAssets": 0.08,
                    },
                },
            ]

            with patch.object(
                analyzer, "get_fundamentals_concurrent", return_value=mock_fundamentals
            ):
                recommendations = await analyzer.recommended_stocks(
                    tickers=sample_tickers
                )

                # Should not recommend stocks with missing change data
                assert recommendations == []

    @pytest.mark.asyncio
    async def test_recommended_stocks_concurrency_parameter(self, sample_cache_file):
        """Test that max_concurrent parameter is passed correctly."""
        sample_tickers = ["TEST1", "TEST2"]

        with patch("aiofiles.open", create=True) as mock_aiofiles:
            mock_file = AsyncMock()
            mock_file.read.return_value = json.dumps(sample_tickers)
            mock_aiofiles.return_value.__aenter__.return_value = mock_file
            mock_aiofiles.return_value.__aexit__.return_value = None

            analyzer = await FundamentalStockAnalyzer.create(
                cache_file=sample_cache_file
            )

            with patch.object(
                analyzer, "get_fundamentals_concurrent", return_value=[]
            ) as mock_get_fundamentals:
                await analyzer.recommended_stocks(
                    tickers=sample_tickers, max_concurrent=5
                )

                # Verify get_fundamentals_concurrent was called with correct parameters
                mock_get_fundamentals.assert_called_once()
                call_kwargs = mock_get_fundamentals.call_args.kwargs
                assert call_kwargs["max_concurrent"] == 5
                assert call_kwargs["field_set"] == "all"
