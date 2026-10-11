The Flask agent and evaluation runner use the same provider factory.

For local Ollama, add these settings to backend/.env:

```dotenv
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:7b
OLLAMA_BASE_URL=http://127.0.0.1:11434/v1
```

Download the model first with `ollama pull qwen2.5:7b`. Keep Ollama running,
restart Flask, and run from backend:

```sh
python -m evals.run_agent_evals --case bare_username --runs 1 --output evals/reports/ollama.json
```

For Cloud Run, configure its environment with:

```dotenv
LLM_PROVIDER=groq
LLM_MODEL=openai/gpt-oss-20b
GROQ_API_KEY=<your secret>
```

Without LLM_PROVIDER, Groq remains the default. Without LLM_MODEL, Groq uses
GROQ_MODEL (or llama-3.1-8b-instant), and Ollama uses qwen2.5:7b. Unknown
providers fail during initialization. Providers do not automatically fall back.
Cloud Run uses Groq; its localhost cannot reach the Ollama server on your PC.

Ollama uses the OpenAI-compatible endpoint, translates the output token setting,
and omits Groq-only reasoning and tool-choice controls. Its request timeout is
120 seconds to allow local inference and model loading. Groq keeps its 30-second
timeout and bounded Retry-After handling. Both use the same tools, backend
ranking, final metadata selection, answer validation, and frontend response.
Eval reports identify the provider and model. Groq records individual retry
attempts, while Ollama has automatic SDK retries disabled.

Reports include a trace of all attempted tool calls, including calls rejected before
execution, their errors, and corrective messages for answers without a backend
ranking. The tools list continues to show calls that reached tool execution.

For Gemini, set LLM_PROVIDER=gemini, LLM_MODEL to a model available to your
project, and GEMINI_API_KEY. GEMINI_MODEL is the fallback when LLM_MODEL is absent.
The client uses Google's OpenAI-compatible endpoint and the existing shared tool
schemas. Thought-signature extensions on tool calls are retained in history.

To space evaluations about two case starts per minute:

```sh
python -m evals.run_agent_evals --runs 1 --case-interval 30 --output evals/reports/gemini.json
```

This interval applies to evals only, including repetitions, and does not count as
case latency. It is not an API quota guarantee: cases make multiple requests;
RPM, TPM, and daily quotas still apply. Increase the interval if needed.
