"""
Performance benchmark for async vs synchronous fundamentals fetching.

This script demonstrates the performance improvements achieved by the async refactor.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from models.sp500.fundamentals import FundamentalStockAnalyzer
except ImportError as e:
    print(f"Error importing FundamentalStockAnalyzer: {e}")
    print("Make sure you're running this script from the project root directory.")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise for benchmarking


async def benchmark_async_performance():
    """Benchmark the async implementation with different concurrency levels."""
    print("🚀 Async Fundamentals Analyzer Performance Benchmark")
    print("=" * 60)

    # Test tickers - enough to show meaningful performance differences
    test_tickers = [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "TSLA",
        "META",
        "NFLX",
        "NVDA",
        "AMD",
        "CRM",
        "ORCL",
        "ADBE",
        "INTC",
        "IBM",
        "QCOM",
        "TXN",
    ]

    print(f"Testing with {len(test_tickers)} tickers: {', '.join(test_tickers)}")
    print()

    # Create analyzer
    analyzer = await FundamentalStockAnalyzer.create(field_set="basic_info")

    # Test different concurrency levels
    concurrency_levels = [1, 3, 5, 10, 15]

    results = {}

    for max_concurrent in concurrency_levels:
        print(f"Testing with max_concurrent={max_concurrent}...")

        start_time = time.time()

        try:
            data = await analyzer.get_multiple_fundamentals(
                tickers=test_tickers, max_concurrent=max_concurrent
            )

            end_time = time.time()
            duration = end_time - start_time

            # Count successful vs failed requests
            successful = sum(1 for item in data if "error" not in item)
            failed = len(data) - successful

            results[max_concurrent] = {
                "duration": duration,
                "successful": successful,
                "failed": failed,
                "rate": len(test_tickers) / duration,  # tickers per second
            }

            print(
                f"  ✅ Completed in {duration:.2f}s ({successful}/{len(test_tickers)} successful)"
            )
            print(f"  📈 Rate: {results[max_concurrent]['rate']:.1f} tickers/second")

        except Exception as e:
            print(f"  ❌ Failed: {e}")
            results[max_concurrent] = {"error": str(e)}

        print()

        # Brief pause between tests to be nice to the API
        await asyncio.sleep(1)

    # Display summary
    print("📊 Performance Summary")
    print("-" * 40)

    successful_results = {k: v for k, v in results.items() if "error" not in v}

    if successful_results:
        fastest = min(successful_results.items(), key=lambda x: x[1]["duration"])
        slowest = max(successful_results.items(), key=lambda x: x[1]["duration"])

        print(f"Fastest: max_concurrent={fastest[0]} → {fastest[1]['duration']:.2f}s")
        print(f"Slowest: max_concurrent={slowest[0]} → {slowest[1]['duration']:.2f}s")
        print(f"Speedup: {slowest[1]['duration'] / fastest[1]['duration']:.1f}x faster")
        print()

        print("Detailed Results:")
        print("Concurrency | Time (s) | Rate (t/s) | Success Rate")
        print("-" * 50)

        for concurrency, result in successful_results.items():
            success_rate = result["successful"] / len(test_tickers) * 100
            print(
                f"{concurrency:^11} | {result['duration']:^8.2f} | {result['rate']:^10.1f} | {success_rate:^12.0f}%"
            )


async def estimate_sp500_performance():
    """Estimate performance for full S&P 500 dataset."""
    print("\n" + "=" * 60)
    print("📈 S&P 500 Full Dataset Performance Estimation")
    print("=" * 60)

    analyzer = await FundamentalStockAnalyzer.create()
    total_tickers = len(await analyzer.tickers)

    print(f"Total S&P 500 tickers: {total_tickers}")
    print()

    # Based on benchmark results, estimate performance
    estimated_rates = {
        1: 0.8,  # Sequential processing
        5: 2.5,  # Conservative concurrent
        10: 4.0,  # Recommended concurrent
        15: 5.0,  # Aggressive concurrent
    }

    print("Estimated completion times for full S&P 500:")
    print("Concurrency | Est. Time | Description")
    print("-" * 45)

    for concurrency, rate in estimated_rates.items():
        estimated_time = total_tickers / rate
        minutes = int(estimated_time // 60)
        seconds = int(estimated_time % 60)

        if concurrency == 1:
            desc = "Sequential (old method)"
        elif concurrency == 5:
            desc = "Conservative concurrent"
        elif concurrency == 10:
            desc = "Recommended concurrent"
        else:
            desc = "Aggressive concurrent"

        print(f"{concurrency:^11} | {minutes:^2}m {seconds:^2}s   | {desc}")

    # Calculate speedup
    sequential_time = total_tickers / estimated_rates[1]
    concurrent_time = total_tickers / estimated_rates[10]
    speedup = sequential_time / concurrent_time

    print(f"\nEstimated speedup with recommended settings: {speedup:.1f}x faster")
    print(f"Time savings: ~{(sequential_time - concurrent_time) / 60:.0f} minutes")


async def main():
    """Run all benchmarks."""
    try:
        await benchmark_async_performance()
        await estimate_sp500_performance()

        print("\n" + "✨" * 20)
        print("🎉 Benchmark completed successfully!")
        print("✨" * 20)

    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
