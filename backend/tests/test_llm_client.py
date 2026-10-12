import json
import os
import unittest
from unittest.mock import Mock, patch

import httpx
from openai import OpenAI

from anirec.agent_tools import AgentTools
from anirec.llm_client import OllamaClient, GeminiClient, create_client, create_completion, provider_config, provider_api_key, redact_api_keys


class ProviderTests(unittest.TestCase):
    def test_api_key_modes_select_provider_key_and_model(self):
        keys = {"GEMINI_API_KEY_BILLED": "paid-test", "GEMINI_API_KEY_UNBILLED": "free-test",
                "GROQ_API_KEY_UNBILLED": "groq-test", "GEMINI_MODEL": "gemini-model",
                "GROQ_MODEL": "groq-model", "LLM_PROVIDER": "gemini", "LLM_MODEL": ""}
        for mode, provider, key in [("gemini-unbilled", "gemini", "free-test"),
                                    ("gemini-billed", "gemini", "paid-test"),
                                    ("groq-unbilled", "groq", "groq-test")]:
            with self.subTest(mode=mode), patch.dict(os.environ, dict(keys, API_KEY_MODE=mode), clear=True):
                self.assertEqual(provider_config(), (provider, f"{provider}-model"))
                self.assertEqual(provider_api_key(provider), key)
                with patch("openai.OpenAI") as gemini, patch("anirec.llm_client.Groq") as groq:
                    create_client(provider)
                    self.assertEqual((gemini if provider == "gemini" else groq).call_args.kwargs["api_key"], key)

    def test_invalid_mode_missing_key_and_provider_mismatch(self):
        with patch.dict(os.environ, {"API_KEY_MODE": "typo"}, clear=True):
            with self.assertRaises(ValueError):
                provider_config()
        with patch.dict(os.environ, {"API_KEY_MODE": "gemini-unbilled",
                                    "GEMINI_API_KEY_BILLED": "paid", "GEMINI_API_KEY": "legacy"}, clear=True):
            with self.assertRaisesRegex(ValueError, "GEMINI_API_KEY_UNBILLED"):
                provider_api_key("gemini")
            with self.assertRaises(ValueError):
                create_client("groq")
        with patch.dict(os.environ, {"GEMINI_API_KEY": "legacy-test"}, clear=True):
            self.assertEqual(provider_api_key("gemini"), "legacy-test")

    def test_redacts_all_api_keys(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY_BILLED": "paid-secret",
                                    "GROQ_API_KEY_UNBILLED": "groq-secret",
                                    "FUTURE_API_KEY": "future-secret", "EMPTY_API_KEY": ""}, clear=True):
            self.assertEqual(redact_api_keys("paid-secret groq-secret future-secret"),
                             "[REDACTED] [REDACTED] [REDACTED]")

    def test_gemini_config_and_supported_request_arguments(self):
        with patch.dict(os.environ, {"LLM_PROVIDER":"gemini", "GEMINI_MODEL":"configured"}, clear=True):
            self.assertEqual(provider_config(), ("gemini", "configured"))
        sdk = Mock()
        client = GeminiClient(sdk)
        create_completion(client, model="configured", messages=[], max_completion_tokens=1024,
                          tool_choice="auto", parallel_tool_calls=False)
        sent = sdk.chat.completions.create.call_args.kwargs
        self.assertEqual(sent["max_tokens"], 1024)
        self.assertEqual(sent["tool_choice"], "auto")
        self.assertNotIn("parallel_tool_calls", sent)
        self.assertNotIn("reasoning_effort", sent)

    def test_legacy_groq_config(self):
        with patch.dict(os.environ, {"GROQ_MODEL": "legacy-model"}, clear=True):
            self.assertEqual(provider_config(), ("groq", "legacy-model"))

    def test_ollama_ignores_groq_model(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "ollama", "GROQ_MODEL": "legacy"}, clear=True):
            self.assertEqual(provider_config(), ("ollama", "qwen2.5:7b"))

    def test_explicit_model_and_invalid_provider(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "ollama", "LLM_MODEL": "custom"}, clear=True):
            self.assertEqual(provider_config(), ("ollama", "custom"))
        with patch.dict(os.environ, {"LLM_PROVIDER": "typo"}, clear=True):
            with self.assertRaises(ValueError):
                provider_config()
        with self.assertRaises(ValueError):
            create_client("typo")

    def test_ollama_wire_format_and_tool_response(self):
        captured = []
        def handle(request):
            captured.append(json.loads(request.content))
            self.assertEqual(str(request.url), "http://localhost:11434/v1/chat/completions")
            return httpx.Response(200, json={"id":"test", "object":"chat.completion", "created":0,
                "model":"qwen2.5:7b", "choices":[{"index":0, "finish_reason":"tool_calls",
                "message":{"role":"assistant", "content":None, "tool_calls":[{"id":"call1",
                "type":"function", "function":{"name":"lookup_anime_ids",
                "arguments":'{"anime_names":["Naruto"]}'}}]}}]})
        sdk = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1",
                     http_client=httpx.Client(transport=httpx.MockTransport(handle)))
        with OllamaClient(sdk) as client:
            response = create_completion(client, model="qwen2.5:7b", messages=[],
                tools=AgentTools.tool_schemas("ollama"), tool_choice="auto", parallel_tool_calls=False,
                max_completion_tokens=1024, temperature=0, response_format={"type":"json_object"})
        self.assertEqual(response.choices[0].message.tool_calls[0].function.name, "lookup_anime_ids")
        payload = captured[0]
        self.assertEqual(payload["max_tokens"], 1024)
        for field in ("max_completion_tokens", "reasoning_effort", "parallel_tool_calls", "tool_choice"):
            self.assertNotIn(field, payload)
        filter_schema = next(t["function"]["parameters"] for t in payload["tools"]
                             if t["function"]["name"] == "filter_anime")
        self.assertNotIn("sim_ids_scores", filter_schema["properties"])


if __name__ == "__main__":
    unittest.main()
