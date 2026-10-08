import atexit
import os

import httpx
from flask import Blueprint, current_app, jsonify, request
from google import genai
from google.genai import errors, types

from ..agent_service import AgentLoopError, build_gemini_tools, run_agent

agent_py = Blueprint("agent", __name__)


def init_agent_tools(app):
    app.config["GEMINI_TOOLS"] = build_gemini_tools(app.config["AGENT_TOOLS"])
    app.config.setdefault("GEMINI_MODEL", os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"))
    app.config.setdefault("AGENT_MAX_TURNS", 10)
    if "GEMINI_CLIENT" not in app.config:
        client = genai.Client(
            api_key=os.environ["GEMINI_API_KEY"],
            http_options=types.HttpOptions(
                timeout=30_000, retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
        app.config["GEMINI_CLIENT"] = client
        atexit.register(client.close)


@agent_py.route("/", methods=["GET"])
def agent():
    prompt = request.args.get("prompt", "Find me 5 anime similar to Naruto").strip()
    if not prompt or len(prompt) > 4000:
        return jsonify(error="Prompt must contain between 1 and 4000 characters."), 400
    try:
        response = run_agent(
            current_app.config["GEMINI_CLIENT"], current_app.config["AGENT_TOOLS"], prompt,
            model=current_app.config["GEMINI_MODEL"], tools=current_app.config["GEMINI_TOOLS"],
            max_turns=current_app.config["AGENT_MAX_TURNS"],
        )
    except errors.APIError as exc:
        message = str(exc.message)
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            message = message.replace(api_key, "[REDACTED]")
        current_app.logger.warning("Gemini request failed with status %s: %s", exc.code, message)
        status = 503 if exc.code == 429 or (exc.code and exc.code >= 500) else 502
        return jsonify(error="Gemini is temporarily unavailable. Try again later."
                       if status == 503 else "Gemini could not process the request."), status
    except httpx.TimeoutException:
        return jsonify(error="Gemini timed out. Try again later."), 504
    except httpx.RequestError:
        return jsonify(error="Could not connect to Gemini."), 502
    except AgentLoopError as exc:
        return jsonify(error=str(exc)), 502
    return jsonify(response.model_dump(mode="json", exclude_none=True))
