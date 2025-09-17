"""
Example usage of the async FundamentalStockAnalyzer.

This demonstrates how to use the refactored async version for better performance.
"""

import asyncio
import logging
import time

from models.sp500.fundamentals import FundamentalStockAnalyzer

# Configure logging to see progress
logging.basicConfig(level=logging.INFO)


async def main():
    """Main example function showing async usage."""
    print("Creating async analyzer...")

    # Use the async factory method to create and initialize the analyzer
    analyzer = await FundamentalStockAnalyzer.create(
        field_set="basic_info"  # Use a smaller field set for faster testing
    )

    print(f"Loaded {len(await analyzer.tickers)} S&P 500 tickers")

    # Test with a small subset first
    test_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

    print(f"\nFetching data for {len(test_tickers)} tickers...")
    start_time = time.time()

    # Get fundamentals for multiple tickers concurrently
    results = await analyzer.get_fundamentals_concurrent(
        tickers=test_tickers,
        max_concurrent=3,  # Limit concurrent requests
    )

    end_time = time.time()
    print(f"Completed in {end_time - start_time:.2f} seconds")

    # Display results
    for result in results:
        ticker = result["ticker"]
        if "error" in result:
            print(f"{ticker}: Error - {result['error']}")
        else:
            data = result["data"]
            market_cap = data.get("marketCap", "N/A")
            sector = data.get("sector", "N/A")
            print(f"{ticker}: Market Cap: {market_cap}, Sector: {sector}")


async def benchmark_comparison():
    """Compare performance between concurrent vs sequential processing."""
    analyzer = await FundamentalStockAnalyzer.create(field_set="basic_info")
    test_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NFLX", "NVDA"]

    print(f"\nBenchmarking with {len(test_tickers)} tickers...")

    # Test concurrent version
    print("Running concurrent version...")
    start_time = time.time()
    await analyzer.get_fundamentals_concurrent(tickers=test_tickers, max_concurrent=5)
    concurrent_time = time.time() - start_time
    print(f"Concurrent version: {concurrent_time:.2f} seconds")

    # Simulate sequential version by setting max_concurrent=1
    print("Running sequential version...")
    start_time = time.time()
    await analyzer.get_fundamentals_concurrent(tickers=test_tickers, max_concurrent=1)
    sequential_time = time.time() - start_time
    print(f"Sequential version: {sequential_time:.2f} seconds")

    speedup = sequential_time / concurrent_time
    print(f"Speedup: {speedup:.2f}x faster with concurrency")


if __name__ == "__main__":
    # Run the main example
    asyncio.run(main())

    # Run the benchmark comparison
    asyncio.run(benchmark_comparison())
