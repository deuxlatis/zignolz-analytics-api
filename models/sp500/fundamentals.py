import json
import logging
import os
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Get the directory where this file is located
CURRENT_DIR = Path(__file__).parent
CACHE_FILE = str(CURRENT_DIR / "sp500_tickers.json")
CACHE_DAYS = 30
SP500_TICKERS_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


class FundamentalStockAnalyzer:
    def __init__(self, cache_file: str = CACHE_FILE):
        self.cache_file = cache_file
        self.tickers = self.get_sp500_tickers()

    def get_sp500_tickers(self):
        # If cache file exist and is fresh, load from it
        logger.debug("Checking cache file status...")
        if os.path.exists(self.cache_file):
            file_mod_time = datetime.fromtimestamp(os.path.getmtime(self.cache_file))
            if datetime.now() - file_mod_time < timedelta(days=CACHE_DAYS):
                logger.info(
                    f"Loading tickers from cache (last updated: {file_mod_time})"
                )
                with open(self.cache_file, "r") as f:
                    tickers = json.load(f)
                logger.info(f"Loaded {len(tickers)} tickers from cache")
                return tickers
            else:
                logger.warning(
                    f"Cache file is stale (last updated: {file_mod_time}), fetching new data..."
                )
                headers = {"User-Agent": "Mozilla/5.0"}
                response = requests.get(SP500_TICKERS_URL, headers=headers)
                tables = pd.read_html(StringIO(response.text))
                tickers = tables[0]["Symbol"].tolist()
                with open(self.cache_file, "w") as f:
                    json.dump(tickers, f)
                logger.info(f"Fetched and cached {len(tickers)} tickers")
                return tickers
        else:
            logger.warning("Cache file does not exist, fetching new data...")
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(SP500_TICKERS_URL, headers=headers)
            tables = pd.read_html(StringIO(response.text))
            tickers = tables[0]["Symbol"].tolist()
            with open(self.cache_file, "w") as f:
                json.dump(tickers, f)
            logger.info(f"Fetched and cached {len(tickers)} tickers")
            return tickers
