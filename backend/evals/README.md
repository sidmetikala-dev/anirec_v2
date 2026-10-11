# Agent evaluations

These opt-in evaluations call the real Groq model with deterministic backend
fixtures. They test model behavior and orchestration, not real MAL data,
recommendation relevance, PostgreSQL, or the frontend. They consume API quota.
Normal unit tests remain offline.

From `backend` with its virtual environment activated:

```bash
# Inspect cases without making API calls.
python -m evals.run_agent_evals --list

# Quick regression check.
python -m evals.run_agent_evals --case bare_username --runs 1

# Run each of the 15 prompts three times.
python -m evals.run_agent_evals --runs 3

# Save a model comparison separately.
python -m evals.run_agent_evals --runs 3 --output evals/reports/model-comparison.json
```

The runner reads `GROQ_API_KEY` and `GROQ_MODEL` from `backend/.env`.
GPT-OSS reasoning effort is controlled by `GROQ_REASONING_EFFORT` (`low`,
`medium`, or `high`, default `medium`) and recorded in the report. For a
comparison in Git Bash:

```bash
GROQ_REASONING_EFFORT=medium python -m evals.run_agent_evals --case exclude_naruto --runs 3 --output evals/reports/medium.json
```

Higher effort may consume more tokens and quota.

The runner executes sequentially and writes a JSON report after every case. Reports include tool
arguments and results, validated answers, request latency, token usage,
finish reasons, and provider error types/status codes. API calls and retry
attempts are counted separately in `requests`.

Assertions check required/forbidden tools, explicit and relative episode
limits, reference metadata lookups, unknown/duplicate IDs, user-facing errors,
exclusions, and final backend ranking order. Different valid tool sequences
are allowed. Nonzero exit status means a regression or execution failure.

Missing/private/empty usernames are simulated fixture users; no real accounts
are queried. Synthetic similarity rankings are stable and are not a quality
benchmark. The niche case is reported as a known limitation when that is its
only failure; unrelated errors still fail the case.

Run the offline checker tests with the normal suite:

```bash
python -m unittest discover -s tests
```
