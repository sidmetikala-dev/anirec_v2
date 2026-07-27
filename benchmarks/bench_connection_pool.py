"""Compare steady-state query latency with and without a connection pool.

The "fresh" case opens and closes a PostgreSQL connection for every query.
The "pooled" case borrows an already-open connection and returns it to the
pool after every query.
"""

import argparse
import csv
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from psycopg_pool import ConnectionPool


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_FILE = Path(__file__).resolve().parent / "results" / "pool_results.csv"
QUERY = "SELECT 1"


def run_query(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(QUERY)
        cur.fetchone()


def time_fresh_connection(db_url: str) -> float:
    """Time connection setup, one query, and connection teardown."""
    start = time.perf_counter()
    with psycopg.connect(db_url, prepare_threshold=None) as conn:
        run_query(conn)
    return time.perf_counter() - start


def time_pooled_connection(pool: ConnectionPool) -> float:
    """Time borrowing a connection, one query, and returning it."""
    start = time.perf_counter()
    with pool.connection() as conn:
        run_query(conn)
    return time.perf_counter() - start


def percentile(values: list[float], percentage: float) -> float:
    """Return a linearly interpolated percentile without extra dependencies."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * percentage
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def benchmark(db_url: str, iterations: int) -> tuple[list[float], list[float]]:
    fresh_times = []
    pooled_times = []

    # Use a pre-opened pool so this benchmark measures normal repeated usage,
    # not one-time pool creation. One unrecorded query warms up each path.
    pool = ConnectionPool(
        db_url,
        min_size=1,
        max_size=4,
        open=False,
        kwargs={"prepare_threshold": None},
    )
    with pool:
        pool.wait()
        time_fresh_connection(db_url)
        time_pooled_connection(pool)

        # Alternate the order to reduce bias from network or database drift.
        for iteration in range(iterations):
            if iteration % 2 == 0:
                fresh_times.append(time_fresh_connection(db_url))
                pooled_times.append(time_pooled_connection(pool))
            else:
                pooled_times.append(time_pooled_connection(pool))
                fresh_times.append(time_fresh_connection(db_url))

    return fresh_times, pooled_times


def milliseconds(seconds: float) -> float:
    return seconds * 1_000


def summarize(fresh: list[float], pooled: list[float]) -> dict[str, str | int | float]:
    fresh_median = statistics.median(fresh)
    pooled_median = statistics.median(pooled)

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "iterations": len(fresh),
        "fresh_mean_ms": round(milliseconds(statistics.mean(fresh)), 3),
        "fresh_median_ms": round(milliseconds(fresh_median), 3),
        "fresh_p95_ms": round(milliseconds(percentile(fresh, 0.95)), 3),
        "pooled_mean_ms": round(milliseconds(statistics.mean(pooled)), 3),
        "pooled_median_ms": round(milliseconds(pooled_median), 3),
        "pooled_p95_ms": round(milliseconds(percentile(pooled, 0.95)), 3),
        "median_speedup": round(fresh_median / pooled_median, 2),
    }


def append_result(result: dict[str, str | int | float], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not output_file.exists() or output_file.stat().st_size == 0

    with output_file.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=result.keys())
        if needs_header:
            writer.writeheader()
        writer.writerow(result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare fresh PostgreSQL connections with pooled connections."
    )
    parser.add_argument(
        "-n",
        "--iterations",
        type=int,
        default=30,
        help="number of measured queries per method (default: 30)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_RESULTS_FILE,
        help=f"CSV output path (default: {DEFAULT_RESULTS_FILE})",
    )
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    db_url = os.getenv("DATABASE_URL_PROD")
    if not db_url:
        raise RuntimeError("DATABASE_URL_PROD is not set in the environment or .env file")

    fresh, pooled = benchmark(db_url, args.iterations)
    result = summarize(fresh, pooled)
    append_result(result, args.output)

    print(f"Fresh median: {result['fresh_median_ms']:.3f} ms")
    print(f"Pooled median: {result['pooled_median_ms']:.3f} ms")
    print(f"Median speedup: {result['median_speedup']:.2f}x")
    print(f"Saved result to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
