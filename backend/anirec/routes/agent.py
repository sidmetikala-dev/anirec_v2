import atexit
import os

import httpx
from flask import Blueprint, current_app, jsonify, request
import groq
import openai
from ..llm_client import create_client, provider_config

from ..agent_service import AgentAnswer, AgentLoopError, run_agent

agent_py = Blueprint("agent", __name__)


def init_agent_tools(app):
    provider, model = provider_config()
    app.config["LLM_PROVIDER"] = provider
    app.config.setdefault("LLM_MODEL", app.config.get("GROQ_MODEL", model) if provider == "groq" else model)
    app.config["LLM_TOOLS"] = app.config["AGENT_TOOLS"].tool_schemas("groq" if provider == "gemini" else provider)
    app.config.setdefault("AGENT_MAX_TURNS", 10)
    if "LLM_CLIENT" not in app.config:
        client = app.config.get("GROQ_CLIENT") if provider == "groq" else None
        if client is None:
            client = create_client(provider)
            atexit.register(client.close)
        app.config["LLM_CLIENT"] = client
    app.logger.info("Agent provider=%s model=%s", provider, app.config["LLM_MODEL"])


@agent_py.route("/", methods=["GET"])
def agent():
    prompt = request.args.get("prompt", "Find me 5 anime similar to Naruto").strip()
    if not prompt or len(prompt) > 4000:
        return jsonify(recommendations=[], error="Prompt must contain between 1 and 4000 characters."), 400
    try:
        response = run_agent(
            current_app.config["LLM_CLIENT"], current_app.config["AGENT_TOOLS"], prompt,
            model=current_app.config["LLM_MODEL"], tools=current_app.config["LLM_TOOLS"],
            max_turns=current_app.config["AGENT_MAX_TURNS"],
        )
        answer = response.parsed
        if not isinstance(answer, AgentAnswer):
            raise AgentLoopError("The model returned an invalid structured answer.")
        if answer.error:
            return jsonify(recommendations=[], error=answer.error), 400
        anime_data = current_app.config["AGENT_TOOLS"].anime_data
        recommendations = []
        for recommendation in answer.recommendations:
            anime = anime_data.get(str(recommendation.anime_id))
            if not anime or not isinstance(anime.get("title"), str) or not anime["title"]:
                raise AgentLoopError("Recommendation metadata is unavailable.")
            recommendations.append({
                "anime_id": recommendation.anime_id,
                "title": anime["title"],
                "picture_link": (anime.get("main_picture") or {}).get("medium"),
            })
    except (groq.APIStatusError, openai.APIStatusError) as exc:
        message = str(exc)
        for key_name in ("GROQ_API_KEY", "GEMINI_API_KEY"):
            api_key = os.getenv(key_name)
            if api_key:
                message = message.replace(api_key, "[REDACTED]")
        current_app.logger.warning("LLM request failed with status %s: %s", exc.status_code, message)
        if exc.status_code == 404:
            return jsonify(recommendations=[], error="The configured model is unavailable. Check LLM_MODEL and provider access."), 502
        status = 503 if exc.status_code == 429 or (exc.status_code and exc.status_code >= 500) else 502
        return jsonify(recommendations=[], error="The model provider is temporarily unavailable. Try again later."
                       if status == 503 else "The model provider could not process the request."), status
    except (groq.APITimeoutError, openai.APITimeoutError, httpx.TimeoutException):
        return jsonify(recommendations=[], error="The model provider timed out. Try again later."), 504
    except (groq.APIConnectionError, openai.APIConnectionError, httpx.RequestError):
        return jsonify(recommendations=[], error="Could not connect to the model provider."), 502
    except AgentLoopError as exc:
        return jsonify(recommendations=[], error=str(exc)), 502
    return jsonify(recommendations=recommendations)
