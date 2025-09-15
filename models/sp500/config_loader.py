import os
from pathlib import Path
from typing import Any, Dict, List

import yaml


class FundamentalsConfig:
    """Configuration loader for S&P 500 fundamentals analysis."""

    def __init__(self, config_path: str = None, environment: str = None):
        """Initialize configuration loader.

        Parameters
        ----------
        config_path : str, optional
            Path to configuration file. If not provided, uses default.
        environment : str, optional
            Environment name (dev, staging, prod). If not provided, uses ENV variable.
        """
        self.environment = environment or os.getenv("ENVIRONMENT", "dev")

        if config_path is None:
            current_dir = Path(__file__).parent
            config_path = current_dir / "config.yaml"

            # Look for environment-specific config first
            env_config_path = current_dir / f"config.{self.environment}.yaml"
            if env_config_path.exists():
                config_path = env_config_path

        self.config_path = Path(config_path)
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            # Return default configuration if file doesn't exist
            return self._get_default_config()

        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)

        # Override with environment variables if they exist
        config = self._apply_env_overrides(config)
        return config

    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "cache": {"days": 30, "file": "sp500_tickers.json"},
            "data_source": {
                "url": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                "user_agent": "Mozilla/5.0",
            },
            "fields": self._get_default_fields(),
        }

    def _get_default_fields(self) -> Dict[str, List[str]]:
        """Return default fields configuration."""
        return {
            "basic_info": [
                "shortName",
                "sector",
                "industry",
                "marketCap",
                "currentPrice",
            ],
            "valuation_metrics": ["trailingPE", "forwardPE", "priceToBook"],
            "profitability_metrics": [
                "profitMargins",
                "returnOnEquity",
                "returnOnAssets",
            ],
            "financial_health": ["debtToEquity", "quickRatio"],
        }

    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration.

        Supports:
        - SP500_CACHE_DAYS: Override cache duration
        - SP500_CACHE_FILE: Override cache file path
        - SP500_DATA_URL: Override data source URL
        """
        if cache_days := os.getenv("SP500_CACHE_DAYS"):
            config.setdefault("cache", {})["days"] = int(cache_days)

        if cache_file := os.getenv("SP500_CACHE_FILE"):
            config.setdefault("cache", {})["file"] = cache_file

        if data_url := os.getenv("SP500_DATA_URL"):
            config.setdefault("data_source", {})["url"] = data_url

        return config

    def get_fields(self, field_set: str = "all") -> List[str]:
        """Get list of fields for a specific field set.

        Parameters
        ----------
        field_set : str
            Name of the field set to retrieve. Use "all" for all fields.

        Returns
        -------
        List[str]
            List of field names.
        """
        if field_set == "all":
            # Flatten all field groups
            all_fields = []
            for group in self._config.get("fields", {}).values():
                if isinstance(group, list):
                    all_fields.extend(group)
            return list(set(all_fields))  # Remove duplicates

        # Check if it's a predefined field set
        field_sets = self._config.get("field_sets", {})
        if field_set in field_sets:
            if field_sets[field_set] == "all":
                return self.get_fields("all")
            return field_sets[field_set]

        # Check if it's a field group
        fields = self._config.get("fields", {})
        if field_set in fields:
            return fields[field_set]

        raise ValueError(f"Unknown field set: {field_set}")

    def get_cache_config(self) -> Dict[str, Any]:
        """Get cache configuration."""
        return self._config.get("cache", {})

    def get_data_source_config(self) -> Dict[str, Any]:
        """Get data source configuration."""
        return self._config.get("data_source", {})

    @property
    def cache_days(self) -> int:
        """Get cache duration in days."""
        return self.get_cache_config().get("days", 30)

    @property
    def cache_file(self) -> str:
        """Get cache file path."""
        return self.get_cache_config().get("file", "sp500_tickers.json")

    @property
    def data_url(self) -> str:
        """Get data source URL."""
        return self.get_data_source_config().get(
            "url", "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        )

    @property
    def user_agent(self) -> str:
        """Get user agent string."""
        return self.get_data_source_config().get("user_agent", "Mozilla/5.0")
