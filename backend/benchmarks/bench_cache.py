"""Compare uncached and cached recommendation-request latency.

For each benchmark username, this script removes that user's saved
recommendation runs, makes one request that must compute and save fresh
recommendations, and then repeats the identical request to exercise the cache.

The Flask application must already be running.
"""

import argparse
import csv
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_FILE = Path(__file__).resolve().parent / "results" / "cache_results.csv"
DEFAULT_URL = "http://127.0.0.1:5000/recs"

USERNAMES = [
    "LoyJoy",
    "DanteNani2004",
    "Kutu_Mastor",
    "Dadalt",
    "yorkie_pro",
    "Opelo_Stradyon",
    "Captn_Cook",
    "Dinterdos123",
    "Ebelha",
    "Moonshyla",
    "LocNguyen",
    "IM_C4",
    "HumanRat",
    "KSTTK",
    "DaiShi57",
    "dareggon",
    "pensa89",
    "MarnikBe",
    "Edison-great",
    "TheRealist68",
    "MeloniaShaby",
    "Irseus",
    "AkiAki_Akira",
    "GrumbleDango",
    "Vurnox",
    "xF0x",
    "alfalfy",
    "Siipn",
    "Haileytokar",
]

def clear_cached_recommendations(db_url: str, username: str) -> None:
    """Delete only the selected benchmark user's cached recommendation runs."""
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM recommendation_runs AS rr
                USING users AS u
                WHERE rr.user_id = u.user_id
                  AND u.username = %s
                """,
                (username,),
            )

def check_server(session: requests.Session, url: str, timeout: float) -> None:
    """Verify the app is reachable and warm the HTTP path before timing."""
    health_url = f"{url}/health"
    try:
        response = session.get(health_url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as error:
        raise RuntimeError(
            f"Health check failed. Is the Flask app running at {health_url}?"
        ) from error

def timed_request(
    session: requests.Session,
    url: str,
    username: str,
    top_k: int,
    timeout: float,
) -> tuple[float, dict[str, Any]]:
    start = time.perf_counter()
    try:
        response = session.post(
            url,
            json={"username": username, "top_k": top_k},
            timeout=timeout,
        )
    except requests.RequestException as error:
        raise RuntimeError(
            f"Request for {username!r} failed. Is the Flask app running at {url}?"
        ) from error

    try:
        payload = response.json()
    except ValueError as error:
        raise RuntimeError(
            f"Request for {username!r} returned a non-JSON response "
            f"(HTTP {response.status_code})."
        ) from error
    elapsed = time.perf_counter() - start

    if not response.ok:
        message = payload.get("error", payload)
        raise RuntimeError(
            f"Request for {username!r} failed with HTTP "
            f"{response.status_code}: {message}"
        )

    return elapsed, payload

def cache_pipeline_seconds(payload: dict[str, Any], username: str) -> float:
    """Read the server-side timing added by the recommendation route."""
    value = payload.get("cache_pipeline_ms")
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or value < 0
    ):
        raise RuntimeError(
            f"Response for {username!r} has no valid cache_pipeline_ms value. "
            "Restart the Flask app so it uses the instrumented route."
        )
    return value / 1_000

def benchmark_user(
    session: requests.Session,
    db_url: str,
    url: str,
    username: str,
    top_k: int,
    timeout: float,
    reset_cache: bool,
) -> tuple[float, float, float, float]:
    if reset_cache:
        clear_cached_recommendations(db_url, username)

    miss_time, miss_payload = timed_request(
        session, url, username, top_k, timeout
    )
    if miss_payload.get("cached") is True:
        raise RuntimeError(
            f"Expected an uncached response for {username!r}, but received a cache hit. "
            "Run without --no-reset-cache or clear this user's saved runs."
        )

    hit_time, hit_payload = timed_request(
        session, url, username, top_k, timeout
    )
    if hit_payload.get("cached") is not True:
        raise RuntimeError(
            f"Expected a cache hit for the second {username!r} request, "
            "but the response was not marked as cached."
        )

    if miss_payload.get("recommendations") != hit_payload.get("recommendations"):
        raise RuntimeError(
            f"Cached recommendations for {username!r} do not match "
            "the freshly generated recommendations."
        )

    miss_pipeline_time = cache_pipeline_seconds(miss_payload, username)
    hit_pipeline_time = cache_pipeline_seconds(hit_payload, username)
    return miss_time, hit_time, miss_pipeline_time, hit_pipeline_time

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

def milliseconds(seconds: float) -> float:
    return seconds * 1_000

def summarize(
    miss_pipeline_times: list[float],
    hit_pipeline_times: list[float],
    miss_request_times: list[float],
    hit_request_times: list[float],
    url: str,
    top_k: int,
) -> dict[str, str | int | float]:
    miss_median = statistics.median(miss_pipeline_times)
    hit_median = statistics.median(hit_pipeline_times)
    request_miss_median = statistics.median(miss_request_times)
    request_hit_median = statistics.median(hit_request_times)

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "endpoint": url,
        "user_count": len(miss_pipeline_times),
        "top_k": top_k,
        "server_uncached_mean_ms": round(
            milliseconds(statistics.mean(miss_pipeline_times)), 3
        ),
        "server_uncached_median_ms": round(milliseconds(miss_median), 3),
        "server_uncached_p95_ms": round(
            milliseconds(percentile(miss_pipeline_times, 0.95)), 3
        ),
        "server_cached_mean_ms": round(
            milliseconds(statistics.mean(hit_pipeline_times)), 3
        ),
        "server_cached_median_ms": round(milliseconds(hit_median), 3),
        "server_cached_p95_ms": round(
            milliseconds(percentile(hit_pipeline_times, 0.95)), 3
        ),
        "server_median_time_saved_ms": round(
            milliseconds(miss_median - hit_median), 3
        ),
        "server_median_speedup": round(miss_median / hit_median, 2),
        "end_to_end_uncached_mean_ms": round(
            milliseconds(statistics.mean(miss_request_times)), 3
        ),
        "end_to_end_uncached_median_ms": round(
            milliseconds(request_miss_median), 3
        ),
        "end_to_end_uncached_p95_ms": round(
            milliseconds(percentile(miss_request_times, 0.95)), 3
        ),
        "end_to_end_cached_mean_ms": round(
            milliseconds(statistics.mean(hit_request_times)), 3
        ),
        "end_to_end_cached_median_ms": round(
            milliseconds(request_hit_median), 3
        ),
        "end_to_end_cached_p95_ms": round(
            milliseconds(percentile(hit_request_times, 0.95)), 3
        ),
        "end_to_end_median_time_saved_ms": round(
            milliseconds(request_miss_median - request_hit_median), 3
        ),
        "end_to_end_median_speedup": round(
            request_miss_median / request_hit_median, 2
        ),
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
        description="Compare uncached and cached recommendation-request latency."
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"recommendation endpoint (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="number of recommendations requested (default: 5)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=len(USERNAMES),
        help=f"number of benchmark users to test (default: {len(USERNAMES)})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="timeout for each HTTP request in seconds (default: 120)",
    )
    parser.add_argument(
        "--no-reset-cache",
        action="store_true",
        help="do not delete the selected users' cached runs before testing",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_RESULTS_FILE,
        help=f"CSV output path (default: {DEFAULT_RESULTS_FILE})",
    )
    args = parser.parse_args()

    if not 1 <= args.top_k <= 50:
        parser.error("--top-k must be between 1 and 50")
    if not 1 <= args.limit <= len(USERNAMES):
        parser.error(f"--limit must be between 1 and {len(USERNAMES)}")
    if args.timeout <= 0:
        parser.error("--timeout must be greater than 0")

    args.url = args.url.rstrip("/")
    return args

def main() -> None:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    db_url = os.getenv("DATABASE_URL_PROD")
    if not args.no_reset_cache and not db_url:
        raise RuntimeError("DATABASE_URL_PROD is not set in the environment or .env file")

    miss_request_times = []
    hit_request_times = []
    miss_pipeline_times = []
    hit_pipeline_times = []
    selected_users = USERNAMES[: args.limit]

    with requests.Session() as session:
        check_server(session, args.url, args.timeout)
        for index, username in enumerate(selected_users, start=1):
            (
                miss_request_time,
                hit_request_time,
                miss_pipeline_time,
                hit_pipeline_time,
            ) = benchmark_user(
                session=session,
                db_url=db_url or "",
                url=args.url,
                username=username,
                top_k=args.top_k,
                timeout=args.timeout,
                reset_cache=not args.no_reset_cache,
            )
            miss_request_times.append(miss_request_time)
            hit_request_times.append(hit_request_time)
            miss_pipeline_times.append(miss_pipeline_time)
            hit_pipeline_times.append(hit_pipeline_time)
            print(
                f"[{index}/{len(selected_users)}] {username}: "
                f"server uncached={milliseconds(miss_pipeline_time):.3f} ms, "
                f"server cached={milliseconds(hit_pipeline_time):.3f} ms"
            )

    result = summarize(
        miss_pipeline_times,
        hit_pipeline_times,
        miss_request_times,
        hit_request_times,
        args.url,
        args.top_k,
    )
    append_result(result, args.output)

    print(f"Server uncached median: {result['server_uncached_median_ms']:.3f} ms")
    print(f"Server cached median: {result['server_cached_median_ms']:.3f} ms")
    print(
        f"Server median time saved: "
        f"{result['server_median_time_saved_ms']:.3f} ms"
    )
    print(f"Server median speedup: {result['server_median_speedup']:.2f}x")
    print(f"Saved result to: {args.output.resolve()}")

if __name__ == "__main__":
    main()
