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

    async def get_fundamentals_concurrent(
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

    async def recommended_stocks(
        self,
        tickers: Optional[List[str]] = None,
        max_concurrent: int = 10,
    ) -> List[str]:
        """Get recommended stock tickers based on specific criteria.

        This method identifies stocks with potential for recovery based on two criteria:
        1. Signs of recovery: Exactly one short-term period (6M, 3M, 1M, 1W) showing
           modest gains (0-3%) while others are <= 0%
        2. Full decline: All short-term periods <= 0%

        Both criteria require:
        - Annual decline >= 25%
        - Strong financial fundamentals (positive cash flows, margins, etc.)

        Parameters
        ----------
        tickers : List[str], optional
            List of ticker symbols to analyze. If not provided, uses all S&P 500 tickers.
        max_concurrent : int, default 10
            Maximum number of concurrent requests to limit API rate.

        Returns
        -------
        List[str]
            List of recommended ticker symbols.
        """
        if tickers is None:
            tickers = await self.tickers

        logger.info(f"Analyzing {len(tickers)} tickers for stock recommendations")

        # Get comprehensive fundamental data for all tickers
        fundamentals = await self.get_fundamentals_concurrent(
            tickers=tickers,
            field_set="all",  # Need all fields for comprehensive analysis
            max_concurrent=max_concurrent,
        )

        signs_of_recovery = []
        full_decline = []

        for stock in fundamentals:
            data = stock.get("data", {})

            # Skip if there's an error or missing critical data
            if "error" in stock:
                continue

            # Check core financial criteria - same for both scenarios
            if not self._meets_financial_criteria(data):
                continue

            # Check short-term change criteria
            change_fields = [
                "sixMonthChangePercent",
                "threeMonthChangePercent",
                "oneMonthChangePercent",
                "oneWeekChangePercent",
            ]

            values = []
            valid = True
            for field in change_fields:
                val = data.get(field)
                if val is None:
                    valid = False
                    break
                values.append(val)

            if not valid:
                continue

            # Criterion 1: Exactly one short-term period in (0, 3%] and others <= 0
            in_range = [0 < v <= 3 for v in values]
            others_nonpos = [v <= 0 for i, v in enumerate(values) if not in_range[i]]

            # Criterion 2: All short-term periods <= 0
            all_nonpositive = all(v <= 0 for v in values)

            if sum(in_range) == 1 and all(others_nonpos):
                signs_of_recovery.append(stock["ticker"])
            elif all_nonpositive:
                full_decline.append(stock["ticker"])

        # Combine both lists for final recommendations
        recommended = signs_of_recovery + full_decline

        logger.info(
            f"Found {len(recommended)} recommended stocks: "
            f"{len(signs_of_recovery)} with signs of recovery, "
            f"{len(full_decline)} in full decline"
        )

        return recommended

    def _meets_financial_criteria(self, data: Dict[str, Any]) -> bool:
        """Check if stock meets the core financial criteria.

        Parameters
        ----------
        data : Dict[str, Any]
            Stock fundamental data dictionary.

        Returns
        -------
        bool
            True if stock meets all financial criteria, False otherwise.
        """
        return (
            data.get("annualChangePercent") is not None
            and data["annualChangePercent"] <= -25
            and data.get("operatingCashflow") is not None
            and data["operatingCashflow"] > 300000000
            and data.get("freeCashflow") is not None
            and data["freeCashflow"] > 300000000
            and data.get("ebitda") is not None
            and data["ebitda"] > 0
            and data.get("profitMargins") is not None
            and data["profitMargins"] > 0
            and data.get("revenueGrowth") is not None
            and data["revenueGrowth"] > -10
            and data.get("ebitdaMargins") is not None
            and data["ebitdaMargins"] > 0
            and data.get("operatingMargins") is not None
            and data["operatingMargins"] > 0
            and data.get("returnOnAssets") is not None
            and data["returnOnAssets"] > 0
        )

    async def score_recommended_stocks(
        self,
        tickers: Optional[List[str]] = None,
        max_concurrent: int = 10,
    ) -> List[Dict[str, Any]]:
        """Score recommended stocks using a ranking-based approach across multiple financial metrics.

        This method fetches comprehensive fundamental data for recommended stocks,
        ranks them across 9 key financial metrics, and returns sorted results.

        Parameters
        ----------
        tickers : List[str], optional
            List of ticker symbols to analyze. If not provided, gets recommendations
            from recommended_stocks() method.
        max_concurrent : int, default 10
            Maximum number of concurrent requests to limit API rate.

        Returns
        -------
        List[Dict[str, Any]]
            List of dictionaries sorted by total score (best to worst), each containing:
            - ticker: Stock ticker symbol
            - shortName: Company name
            - total_score: Aggregate ranking score (lower is better)
            - individual scores for each scoring metric
            - performance metrics (price changes)
            - financial metrics used in scoring
        """
        # Get recommended tickers if not provided
        if tickers is None:
            logger.info("Getting recommended stocks for scoring...")
            tickers = await self.recommended_stocks(max_concurrent=max_concurrent)

        if not tickers:
            logger.warning("No recommended stocks found for scoring")
            return []

        logger.info(f"Scoring {len(tickers)} recommended stocks")

        # Get comprehensive fundamental data for recommended stocks
        fundamentals = await self.get_fundamentals_concurrent(
            tickers=tickers,
            field_set="all",
            max_concurrent=max_concurrent,
        )

        # Filter out stocks with errors and create DataFrame
        valid_stocks = [stock for stock in fundamentals if "error" not in stock]

        if not valid_stocks:
            logger.warning("No valid stock data found for scoring")
            return []

        # Create DataFrame for scoring analysis
        df_data = []
        ticker_list = []

        for stock in valid_stocks:
            ticker_list.append(stock["ticker"])
            df_data.append(stock["data"])

        df = pd.DataFrame(df_data, index=ticker_list)

        # Define columns needed for scoring and display
        required_columns = [
            "shortName",
            "annualChangePercent",
            "sixMonthChangePercent",
            "threeMonthChangePercent",
            "oneMonthChangePercent",
            "oneWeekChangePercent",
            "operatingCashflow",
            "freeCashflow",
            "ebitda",
            "profitMargins",
            "revenueGrowth",
            "ebitdaMargins",
            "operatingMargins",
            "returnOnAssets",
            "returnOnEquity",
            "enterpriseValue",
        ]

        # Filter DataFrame to include only required columns that exist
        available_columns = [col for col in required_columns if col in df.columns]
        df = df[available_columns]

        # Define columns to score (exclude display-only columns)
        columns_to_score = [
            "operatingCashflow",
            "freeCashflow",
            "ebitda",
            "profitMargins",
            "revenueGrowth",
            "ebitdaMargins",
            "operatingMargins",
            "returnOnAssets",
            "returnOnEquity",
        ]

        # Filter scoring columns to only those that exist in the data
        available_scoring_columns = [
            col for col in columns_to_score if col in df.columns
        ]

        if not available_scoring_columns:
            logger.error("No scoring columns available in the data")
            return []

        logger.info(f"Using {len(available_scoring_columns)} metrics for scoring")

        # Rank and score each column (higher rank = better performance)
        for column in available_scoring_columns:
            # Skip columns with all null values
            if df[column].isna().all():
                logger.warning(f"Column {column} has all null values, skipping")
                continue

            df.loc[:, f"{column}_score"] = (
                df[column]
                .rank(ascending=False, method="min", na_option="bottom")
                .astype(int)
            )

        # Calculate total score (sum of individual ranking scores)
        score_columns = [
            f"{col}_score"
            for col in available_scoring_columns
            if f"{col}_score" in df.columns
        ]

        if score_columns:
            df.loc[:, "total_score"] = df[score_columns].sum(axis=1)
        else:
            logger.error("No score columns were created")
            return []

        # Sort by total score (lower total score = better overall performance)
        df = df.sort_values("total_score", ascending=True)

        # Convert to list of dictionaries for return
        results = []
        for ticker in df.index:
            total_score_value = df.loc[ticker, "total_score"]

            # Safely convert total_score to int
            try:
                if pd.notna(total_score_value):
                    total_score = int(float(total_score_value))  # type: ignore
                else:
                    total_score = 0
            except (ValueError, TypeError):
                total_score = 0

            stock_data = {
                "ticker": str(ticker),
                "total_score": total_score,
            }

            # Add all available data
            for col in df.columns:
                value = df.loc[ticker, col]

                # Handle pandas types for JSON serialization
                if pd.isna(value):
                    stock_data[col] = None
                else:
                    # Convert pandas/numpy types to Python native types
                    try:
                        if isinstance(value, (int, float, str, bool)):
                            stock_data[col] = value
                        elif hasattr(value, "item") and callable(
                            getattr(value, "item")
                        ):
                            # Handle numpy/pandas scalar types that have .item() method
                            stock_data[col] = value.item()  # type: ignore
                        else:
                            # Fallback to string conversion for other types
                            stock_data[col] = str(value)
                    except (ValueError, TypeError, AttributeError):
                        # If all else fails, convert to string
                        stock_data[col] = str(value) if value is not None else None

            results.append(stock_data)

        logger.info(
            f"Successfully scored {len(results)} stocks. "
            f"Best score: {results[0]['total_score']}, "
            f"Worst score: {results[-1]['total_score']}"
        )

        return results

    async def get_recommended_stocks(
        self,
        tickers: Optional[List[str]] = None,
        max_concurrent: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get recommended stocks along with their detailed scores and metrics.

        This method combines the functionality of recommended_stocks() and
        score_recommended_stocks() to provide a comprehensive view of
        recommended stocks with their scoring details. Returns a maximum of 10 stocks,
        prioritizing those with the highest enterprise value when more than 10 are available.

        Parameters
        ----------
        tickers : List[str], optional
            List of ticker symbols to analyze. If not provided, gets recommendations
            from recommended_stocks() method.
        max_concurrent : int, default 10
            Maximum number of concurrent requests to limit API rate.

        Returns
        -------
        List[Dict[str, Any]]
            List of dictionaries (max 10) sorted by total score (best to worst), each containing:
            - ticker: Stock ticker symbol
            - shortName: Company name
            - total_score: Aggregate ranking score (lower is better)
            - individual scores for each scoring metric
            - performance metrics (price changes)
            - financial metrics used in scoring
        """
        # Get recommended tickers if not provided
        if tickers is None:
            logger.info("Getting recommended stocks for scoring...")
            tickers = await self.recommended_stocks(max_concurrent=max_concurrent)

        if not tickers:
            logger.warning("No recommended stocks found for scoring")
            return []

        # Score the recommended stocks
        scored_stocks = await self.score_recommended_stocks(
            tickers=tickers,
            max_concurrent=max_concurrent,
        )

        # If we have more than 10 recommendations, keep the 10 with highest enterprise value
        if len(scored_stocks) > 10:
            logger.info(
                f"Filtering {len(scored_stocks)} recommendations to top 10 by enterprise value"
            )

            # Filter out stocks without enterprise value data
            stocks_with_ev = [
                stock
                for stock in scored_stocks
                if stock.get("enterpriseValue") is not None
            ]

            # Sort by enterprise value (descending) and take top 10
            if stocks_with_ev:
                stocks_with_ev.sort(key=lambda x: x["enterpriseValue"], reverse=True)
                scored_stocks = stocks_with_ev[:10]
                logger.info("Selected top 10 stocks by enterprise value")
            else:
                # Fallback: if no enterprise value data, keep first 10 by total score
                scored_stocks = scored_stocks[:10]
                logger.warning(
                    "No enterprise value data available, keeping first 10 by total score"
                )

        # Sort the scored stocks by total_score (lower is better)
        scored_stocks.sort(key=lambda x: x.get("total_score", float("inf")))
        return scored_stocks
