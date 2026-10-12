"""Live model evaluations against deterministic backend fixtures.

Run from backend: python -m evals.run_agent_evals --runs 3
This uses the configured model API, but no MAL requests, database, or model training.
"""
import argparse
import contextlib
import io
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace

from dotenv import load_dotenv
from anirec.llm_client import create_client, provider_config, redact_api_keys
from unittest.mock import patch

from anirec.agent_service import run_agent
from anirec.agent_tools import AgentTools
from anirec.mal_client import MALProfileRestrictedError, MALUserNotFoundError
from anirec.recommendation_service import NoRatedAnimeError

CASES_PATH = Path(__file__).with_name("cases.json")


def fixtures():
    rows = [
        (20, "Naruto", 220), (1735, "Naruto: Shippuuden", 500),
        (34572, "Black Clover", 170), (1535, "Death Note", 37),
        (11061, "Hunter x Hunter (2011)", 148), (5114, "Fullmetal Alchemist: Brotherhood", 64),
        (40748, "Jujutsu Kaisen", 24), (38000, "Kimetsu no Yaiba", 26),
        (44511, "Chainsaw Man", 12), (9253, "Steins;Gate", 24),
        (37999, "Kaguya-sama wa Kokurasetai", 12), (37450, "Seishun Buta Yarou", 13),
        (5940, "Seiken no Blacksmith", 12), (21, "One Piece", 0),
        (235, "Meitantei Conan", 0), (19, "Monster", 74), (5040, "One Outs", 25),
    ]
    metadata = {str(i): {"title": title, "num_episodes": episodes,
                       "genres": [{"name": "Drama"}], "synopsis": "Synthetic evaluation synopsis.",
                       "main_picture": None} for i, title, episodes in rows}

    def ranking(positive_anime_ids, negative_anime_ids=None, negative_weight=0.5, limit=100):
        excluded = set(positive_anime_ids) | set(negative_anime_ids or [])
        return dict([(i, round(1 - index * .02, 4)) for index, (i, _, _) in enumerate(rows)
                     if i not in excluded][:limit])

    def provider(username, top_k=50):
        if username == "anirec_missing_fixture_user":
            raise MALUserNotFoundError("MyAnimeList user not found. Check the username and try again.")
        if username == "anirec_private_fixture_user":
            raise MALProfileRestrictedError("This MyAnimeList profile is private or restricted.")
        if username == "anirec_empty_fixture_user":
            raise NoRatedAnimeError("No completed, scored TV anime found for this user")
        return {"recommendations": [{"anime_id": i, "predicted_score": score}
                                    for i, score in ranking([]).items()][:top_k]}

    def search(names, limit=10):
        names = [names] if isinstance(names, str) else names
        return [{"anime_id": i, "anime_name": title} for i, title, _ in rows
                if any(name.casefold() in title.casefold() for name in names)][:limit]

    return AgentTools(SimpleNamespace(find_similar=ranking), metadata, provider), search


class RecordedTools:
    def __init__(self, tools):
        self.tools = tools
        self.anime_data = tools.anime_data
        self.calls = []

    def tool_schemas(self, format):
        return self.tools.tool_schemas(format)

    def execute_tool(self, name, arguments):
        record = {"name": name, "arguments": arguments}
        self.calls.append(record)
        try:
            result = self.tools.execute_tool(name, arguments)
            record["result"] = json.loads(result)
            return result
        except Exception as exc:
            record["error"] = str(exc)
            raise

    def get_metadata(self, *args, **kwargs):
        return self.tools.get_metadata(*args, **kwargs)


class RecordedClient:
    def __init__(self, client):
        self.client = client
        self.provider = getattr(client, "provider", "groq")
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        record = {"input_chars": len(json.dumps(kwargs["messages"]))}
        self.requests.append(record)
        start = time.perf_counter()
        try:
            response = self.client.chat.completions.create(**kwargs)
            record["usage"] = response.usage.model_dump() if response.usage else {}
            record["finish_reason"] = response.choices[0].finish_reason if response.choices else None
            return response
        except Exception as exc:
            # Avoid recording provider bodies or headers containing credentials.
            record["error_type"] = type(exc).__name__
            record["status"] = getattr(exc, "status_code", None)
            body = getattr(exc, "body", None)
            error = body.get("error", body) if isinstance(body, dict) else None
            if isinstance(error, dict) and isinstance(error.get("message"), str):
                message = redact_api_keys(error["message"])
                record["provider_message"] = message[:1500]
            raise
        finally:
            record["latency_seconds"] = round(time.perf_counter() - start, 3)


def check_case(case, answer, calls, metadata):
    failures = []
    names = [call["name"] for call in calls]
    for name in case.get("required", []):
        if name not in names:
            failures.append(f"Required tool not used: {name}")
    for name in case.get("forbidden", []):
        if name in names:
            failures.append(f"Unexpected tool: {name}")
    if answer is None:
        return failures + ["No validated final answer"]
    ids = [r.anime_id for r in answer.recommendations]
    if case.get("expect_error"):
        if not answer.error or case.get("error_contains", "").casefold() not in answer.error.casefold():
            failures.append("Expected user-facing error missing")
        if ids:
            failures.append("Recommendations returned despite user error")
        return failures
    if answer.error:
        failures.append(f"Unexpected answer error: {answer.error}")
    if len(ids) != len(set(ids)) or any(str(i) not in metadata for i in ids):
        failures.append("Duplicate or unknown final IDs")
    if set(ids) & set(case.get("excluded_ids", [])):
        failures.append("Excluded anime returned")
    maximum = case.get("max_episodes")
    if maximum is not None:
        filters = [c for c in calls if c["name"] == "filter_anime"]
        if not filters or any(c["arguments"].get("max_episodes") != maximum for c in filters):
            failures.append(f"Expected episode limit {maximum}")
        if any(not 0 < (metadata.get(str(i), {}).get("num_episodes") or 0) <= maximum for i in ids):
            failures.append("Final episode constraint violated")
    reference = case.get("reference_id")
    if reference is not None and not any(c["name"] == "get_metadata" and
            str(reference) in map(str, c["arguments"].get("anime_ids", [])) and
            "num_episodes" in (c["arguments"].get("columns_to_get") or []) for c in calls):
        failures.append("Reference episode count was not fetched")
    rankings = [c["result"] for c in calls if c["name"] in
                {"find_similar_tool", "get_recs_with_username_tool", "filter_anime"}
                and isinstance(c.get("result"), dict) and "error" not in c["result"]]
    if rankings:
        expected = [int(i) for i in list(rankings[-1])[:5]]
        if ids != expected:
            failures.append(f"Expected backend top five {expected}, got {ids}")
    elif not ids:
        failures.append("Unexpected empty recommendations")
    if case.get("unsupported_constraint"):
        failures.append("Niche/popularity filtering is not implemented")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--case", action="append", help="Case ID; repeat to select several")
    parser.add_argument("--case-interval", type=float, default=0,
                        help="Minimum seconds between case starts, including repeated runs")
    parser.add_argument("--list", action="store_true", help="List cases without API calls")
    parser.add_argument("--output", type=Path, default=Path("evals/reports/latest.json"))
    args = parser.parse_args()
    cases = json.loads(CASES_PATH.read_text())
    if args.case:
        unknown = set(args.case) - {c["id"] for c in cases}
        if unknown:
            parser.error(f"Unknown cases: {sorted(unknown)}")
        cases = [c for c in cases if c["id"] in args.case]
    if args.list:
        for case in cases:
            print(case["id"], "-", case["prompt"])
        return 0
    if args.runs < 1:
        parser.error("--runs must be positive")
    if not math.isfinite(args.case_interval) or args.case_interval < 0:
        parser.error("--case-interval must be a finite nonnegative number")
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    provider, model = provider_config()
    reports = []
    previous_start = None
    with create_client(provider) as client:
        for case in cases:
            for repeat in range(args.runs):
                if previous_start is not None:
                    remaining = args.case_interval - (time.perf_counter() - previous_start)
                    if remaining > 0:
                        print(f"Waiting {remaining:.1f}s before next case", flush=True)
                        time.sleep(remaining)
                previous_start = time.perf_counter()
                tools, search = fixtures()
                recorded_tools, recorded_client = RecordedTools(tools), RecordedClient(client)
                start = time.perf_counter()
                answer = None
                failure = None
                trace = []
                with patch("anirec.agent_tools.search_anime", side_effect=search), contextlib.redirect_stdout(io.StringIO()):
                    try:
                        answer = run_agent(recorded_client, recorded_tools, case["prompt"], model=model, trace=trace).parsed
                    except Exception as exc:
                        failure = type(exc).__name__
                failures = check_case(case, answer, recorded_tools.calls, tools.anime_data)
                if failure:
                    failures.append(f"Execution failed: {failure}")
                status = "FAIL" if failures else "PASS"
                if case.get("known_limitation") and failures == ["Niche/popularity filtering is not implemented"]:
                    status = "KNOWN_LIMITATION"
                report = {"case":case["id"], "prompt":case["prompt"], "run":repeat+1,
                          "status":status, "failures":failures, "latency_seconds":round(time.perf_counter()-start,3),
                          "answer":answer.model_dump() if answer else None,
                          "trace":trace, "tools":recorded_tools.calls, "requests":recorded_client.requests}
                reports.append(report)
                print(f'{status}: {case["id"]} run {repeat+1} ({report["latency_seconds"]}s)')
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps({"provider":provider,"model":model,"reasoning_effort":os.getenv("GROQ_REASONING_EFFORT", "medium") if provider == "groq" else None,"backend":"synthetic fixtures","runs":reports}, indent=2))
    evaluated = [r for r in reports if r["status"] != "KNOWN_LIMITATION"]
    passed = sum(r["status"] == "PASS" for r in evaluated)
    print(f"Passed {passed}/{len(evaluated)} evaluated runs; report: {args.output}")
    return 1 if any(r["status"] == "FAIL" for r in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
