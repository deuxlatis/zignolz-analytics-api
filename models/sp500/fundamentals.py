import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles  # type: ignore
import aiohttp
import pandas as pd
import yfinance as yf  # type: ignore

from .config_loader import FundamentalsConfig

logger = logging.getLogger(__name__)


class FundamentalStockAnalyzer:
    def __init__(
        self,
        config_path: Optional[str] = None,
        cache_file: Optional[str] = None,
        field_set: str = "all",
        environment: Optional[str] = None,
    ):
        """Initialize the S&P 500 fundamentals analyzer.

        Parameters
        ----------
        config_path : str, optional
            Path to configuration file. If not provided, uses default config.
        cache_file : str, optional
            Path to cache file. If not provided, uses config setting.
        field_set : str, default "all"
            Field set to use for analysis (e.g., "basic_info", "value_analysis", "all").
        environment : str, optional
            Environment name (dev, staging, prod). If not provided, uses ENV variable.
        """
        env = environment if environment is not None else "default"
        if config_path is not None:
            self.config = FundamentalsConfig(config_path=config_path, environment=env)
        else:
            self.config = FundamentalsConfig(environment=env)
        self.cache_file = cache_file or self._get_cache_file_path()
        self.field_set = field_set
        self._tickers: Optional[List[str]] = None

    @classmethod
    async def create(
        cls,
        config_path: Optional[str] = None,
        cache_file: Optional[str] = None,
        field_set: str = "all",
        environment: Optional[str] = None,
    ) -> "FundamentalStockAnalyzer":
        """Async factory method to create and initialize the analyzer.

        Parameters
        ----------
        config_path : str, optional
            Path to configuration file. If not provided, uses default config.
        cache_file : str, optional
            Path to cache file. If not provided, uses config setting.
        field_set : str, default "all"
            Field set to use for analysis (e.g., "basic_info", "value_analysis", "all").
        environment : str, optional
            Environment name (dev, staging, prod). If not provided, uses ENV variable.

        Returns
        -------
        FundamentalStockAnalyzer
            Fully initialized analyzer with tickers loaded.
        """
        analyzer = cls(config_path, cache_file, field_set, environment)
        analyzer._tickers = await analyzer.get_sp500_tickers()
        return analyzer

    @property
    async def tickers(self) -> List[str]:
        """Get S&P 500 tickers, loading them if not already cached."""
        if self._tickers is None:
            self._tickers = await self.get_sp500_tickers()
        return self._tickers

    def _get_cache_file_path(self) -> str:
        """Get the full path to the cache file."""
        current_dir = Path(__file__).parent
        cache_filename = self.config.cache_file
        return str(current_dir / cache_filename)

    async def get_sp500_tickers(self) -> list[str]:
        """Fetch S&P 500 tickers from Wikipedia or cache.

        Returns
        -------
            list[str]
                A list of S&P 500 tickers.
        """
        # If cache file exist and is fresh, load from it
        logger.debug("Checking cache file status...")
        if os.path.exists(self.cache_file):
            file_mod_time = datetime.fromtimestamp(os.path.getmtime(self.cache_file))
            if datetime.now() - file_mod_time < timedelta(days=self.config.cache_days):
                logger.info(
                    f"Loading tickers from cache (last updated: {file_mod_time})"
                )
                async with aiofiles.open(self.cache_file, "r") as f:
                    content = await f.read()
                    tickers = json.loads(content)
                logger.info(f"Loaded {len(tickers)} tickers from cache")
                return tickers
            else:
                logger.warning(
                    f"Cache file is stale (last updated: {file_mod_time}), fetching new data..."
                )
                return await self._fetch_and_cache_tickers()
        else:
            logger.warning("Cache file does not exist, fetching new data...")
            return await self._fetch_and_cache_tickers()

    async def _fetch_and_cache_tickers(self) -> List[str]:
        """Fetch tickers from data source and cache them."""
        headers = {"User-Agent": self.config.user_agent}

        async with aiohttp.ClientSession() as session:
            async with session.get(self.config.data_url, headers=headers) as response:
                html_content = await response.text()

        tables = pd.read_html(StringIO(html_content))
        tickers = tables[0]["Symbol"].tolist()

        async with aiofiles.open(self.cache_file, "w") as f:
            await f.write(json.dumps(tickers))

        logger.info(f"Fetched and cached {len(tickers)} tickers")
        return tickers

    def get_fields(self, field_set: Optional[str] = None) -> List[str]:
        """Get list of fields for analysis.

        Parameters
        ----------
        field_set : str, optional
            Field set to use. If not provided, uses the instance field_set.

        Returns
        -------
        List[str]
            List of field names.
        """
        return self.config.get_fields(field_set or self.field_set)

    def get_available_field_sets(self) -> List[str]:
        """Get list of available field sets.

        Returns
        -------
        List[str]
            List of available field set names.
        """
        field_sets = list(self.config._config.get("field_sets", {}).keys())
        field_groups = list(self.config._config.get("fields", {}).keys())
        return field_sets + field_groups + ["all"]

    async def _calculate_percentage_changes(
        self, ticker: str
    ) -> Dict[str, Optional[float]]:
        """Calculate percentage changes for different time periods.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol.

        Returns
        -------
        Dict[str, Optional[float]]
            Dictionary containing percentage changes for different periods.
        """
        try:
            # Run yfinance in a thread pool to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            stock = yf.Ticker(ticker)
            hist = await loop.run_in_executor(
                None, lambda: stock.history(period="1y", interval="1wk")
            )

            # Handle timezone localization if needed
            if isinstance(hist.index, pd.DatetimeIndex) and hist.index.tz is not None:
                hist.index = hist.index.tz_localize(None)

            if hist.empty or len(hist) < 2:
                logger.warning(f"Insufficient historical data for {ticker}")
                return {
                    "annualChangePercent": None,
                    "sixMonthChangePercent": None,
                    "threeMonthChangePercent": None,
                    "oneMonthChangePercent": None,
                    "oneWeekChangePercent": None,
                }

            current_price = hist["Close"].iloc[-1]

            # Calculate percentage changes with proper weekly intervals
            # 1 year ≈ 52 weeks, 6 months ≈ 26 weeks, 3 months ≈ 13 weeks, 1 month ≈ 4 weeks
            changes = {}
            periods = {
                "annualChangePercent": min(52, len(hist) - 1),
                "sixMonthChangePercent": min(26, len(hist) - 1),
                "threeMonthChangePercent": min(13, len(hist) - 1),
                "oneMonthChangePercent": min(4, len(hist) - 1),
                "oneWeekChangePercent": min(1, len(hist) - 1),
            }

            for field_name, weeks_back in periods.items():
                try:
                    if weeks_back > 0 and len(hist) > weeks_back:
                        past_price = hist["Close"].iloc[-(weeks_back + 1)]
                        if past_price and past_price != 0:
                            change_percent = (
                                (current_price - past_price) / past_price
                            ) * 100
                            changes[field_name] = round(change_percent, 2)
                        else:
                            changes[field_name] = None
                    else:
                        changes[field_name] = None
                except (IndexError, ZeroDivisionError):
                    changes[field_name] = None

            logger.debug(f"Calculated percentage changes for {ticker}")
            return changes

        except Exception as e:
            logger.error(f"Error calculating percentage changes for {ticker}: {e}")
            return {
                "annualChangePercent": None,
                "sixMonthChangePercent": None,
                "threeMonthChangePercent": None,
                "oneMonthChangePercent": None,
                "oneWeekChangePercent": None,
            }

    async def get_fundamentals(
        self, ticker: str, field_set: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get fundamental data for a ticker.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol.
        field_set : str, optional
            Field set to use for data retrieval. If not provided, uses instance field_set.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing fundamental data for the specified fields.
        """
        fields = self.get_fields(field_set)
        logger.info(f"Fetching {len(fields)} fields for ticker {ticker}")

        try:
            # Run yfinance in a thread pool to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            stock = yf.Ticker(ticker)
            info = await loop.run_in_executor(None, lambda: stock.info)

            # Get basic fundamental data
            fundamental_data = {field: info.get(field, None) for field in fields}

            # Check if any percentage change fields are requested
            percentage_fields = {
                "annualChangePercent",
                "sixMonthChangePercent",
                "threeMonthChangePercent",
                "oneMonthChangePercent",
                "oneWeekChangePercent",
            }

            if any(field in percentage_fields for field in fields):
                # Calculate percentage changes
                percentage_changes = await self._calculate_percentage_changes(ticker)
                # Add requested percentage change fields
                for field in fields:
                    if field in percentage_fields:
                        fundamental_data[field] = percentage_changes.get(field)

            result = {
                "ticker": ticker,
                "data": fundamental_data,
            }
            logger.debug(f"Successfully fetched data for {ticker}")
            return result
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return {
                "ticker": ticker,
                "data": {field: None for field in fields},
                "error": str(e),
            }

    async def get_multiple_fundamentals(
        self,
        tickers: Optional[List[str]] = None,
        field_set: Optional[str] = None,
        max_concurrent: int = 10,
    ) -> list[dict[str, Any]]:
        """Get fundamental data for multiple tickers concurrently.

        Parameters
        ----------
        tickers : List[str], optional
            List of ticker symbols. If not provided, uses all S&P 500 tickers.
        field_set : str, optional
            Field set to use for data retrieval. If not provided, uses instance field_set.
        max_concurrent : int, default 10
            Maximum number of concurrent requests to limit API rate.

        Returns
        -------
        List[Dict[str, Any]]
            List of dictionaries containing fundamental data for each ticker.
        """
        if tickers is None:
            tickers = await self.tickers

        fields = self.get_fields(field_set)
        logger.info(
            f"Fetching {len(fields)} fields for {len(tickers)} tickers with max {max_concurrent} concurrent requests"
        )

        # Create a semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)

        async def fetch_single_ticker(ticker: str, index: int) -> Dict[str, Any]:
            """Fetch data for a single ticker with semaphore protection."""
            async with semaphore:
                logger.info(f"Processing ticker {ticker} ({index + 1}/{len(tickers)})")
                return await self.get_fundamentals(ticker, field_set)

        # Create tasks for all tickers
        tasks = [fetch_single_ticker(ticker, i) for i, ticker in enumerate(tickers)]

        # Execute all tasks concurrently and gather results
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions that occurred
        processed_results: list[dict[str, Any]] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing ticker {tickers[i]}: {result}")
                processed_results.append(
                    {
                        "ticker": tickers[i],
                        "data": {field: None for field in fields},
                        "error": str(result),
                    }
                )
            else:
                # result is guaranteed to be Dict[str, Any] here due to isinstance check
                assert isinstance(result, dict)  # Type narrowing for mypy
                processed_results.append(result)

        logger.info(f"Completed processing {len(processed_results)} tickers")
        return processed_results
