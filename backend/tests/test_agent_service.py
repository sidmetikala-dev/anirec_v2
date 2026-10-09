"""Exercise real SDK response objects without API or database calls."""

import json
import unittest
from unittest.mock import Mock

from pydantic import ValidationError
from flask import Flask
from google import genai
from google.genai import errors, models, types
import httpx

from anirec.agent_service import AgentAnswer, AgentLoopError, run_agent
from anirec.mal_client import MALUserNotFoundError, MALProfileRestrictedError
from anirec.recommendation_service import NoRatedAnimeError
from anirec.routes.agent import agent_py, init_agent_tools


FINAL_ANSWER = {"recommendations": []}


def answer(text=None):
    if text is None:
        text = json.dumps(FINAL_ANSWER)
    # Mock the SDK's response_schema parsing, including invalid-output behavior.
    try:
        parsed = AgentAnswer.model_validate_json(text)
    except ValidationError:
        parsed = None
    return types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(role="model", parts=[types.Part.from_text(text=text)]),
    )], parsed=parsed)


def tool_turn(*names):
    return types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(role="model", parts=[
            types.Part(function_call=types.FunctionCall(name=name, args={}, id=str(i)),
                       thought_signature=b"signature")
            for i, name in enumerate(names)
        ]),
    )])


class AgentServiceTests(unittest.TestCase):
    def setUp(self):
        self.client = Mock()
        self.tools = Mock()

    def run_agent(self, **kwargs):
        return run_agent(self.client, self.tools, "Find anime", tools=[], **kwargs)

    def test_final_schema_serializes_as_json_schema(self):
        self.client.models.generate_content.return_value = answer()
        self.run_agent()
        config = self.client.models.generate_content.call_args.kwargs["config"]
        # Check the actual outgoing field names, not just local schema validation.
        with genai.Client(api_key="schema-check-placeholder") as client:
            payload = models._GenerateContentConfig_to_mldev(client._api_client, config)
        self.assertNotIn("responseSchema", payload)
        schema = payload["responseJsonSchema"]
        self.assertFalse(schema["additionalProperties"])
        self.assertNotIn("additional_properties", json.dumps(payload))
        self.assertEqual(schema["$defs"]["Recommendation"]["properties"]["anime_id"]["minimum"], 1)

    def test_sequential_tools_keep_prompt_history_signatures_and_model(self):
        first, second, final = tool_turn("search_anime_tool"), tool_turn("find_similar_tool"), answer()
        self.client.models.generate_content.side_effect = [first, second, final, answer(json.dumps({
            "recommendations": [{"anime_id": 30, "explanation": "Similar themes"}],
        }))]
        self.tools.execute_tool.side_effect = ['[{"anime_id": 20}]', '{"30": 0.9}']
        self.assertEqual([r.anime_id for r in self.run_agent().parsed.recommendations], [30])
        calls = self.client.models.generate_content.call_args_list
        self.assertEqual([len(c.kwargs["contents"]) for c in calls], [1, 3, 5, 7])
        self.assertEqual({c.kwargs["model"] for c in calls}, {"gemini-3.8-flash"})
        history = calls[2].kwargs["contents"]
        self.assertEqual(history[0].parts[0].text, "Find anime")
        self.assertEqual(history[1].parts[0].thought_signature, b"signature")
        self.assertEqual(history[2].parts[0].function_response.response, {"result": [{"anime_id": 20}]})
        self.assertEqual(history[2].parts[0].function_response.id, "0")
        self.assertEqual([turn.role for turn in history], ["user", "model", "user", "model", "user"])
        self.assertEqual(history[4].parts[0].function_response.response, {"30": 0.9})
        self.assertTrue(calls[0].kwargs["config"].automatic_function_calling.disable)

    def test_backend_controls_filter_input_metadata_selection_and_final_order(self):
        source = {"10": 1.0, "20": 0.9, "30": 0.8, "40": 0.7, "50": 0.6, "60": 0.5}
        filtered = {"20": 0.9, "30": 0.8, "40": 0.7, "50": 0.6, "60": 0.5}
        filter_turn = tool_turn("filter_anime")
        filter_turn.candidates[0].content.parts[0].function_call.args = {
            "max_episodes": 219, "sim_ids_scores": {"60": 99},
        }
        self.client.models.generate_content.side_effect = [
            tool_turn("find_similar_tool"), filter_turn, tool_turn("get_metadata"),
            answer("Ready"), answer(json.dumps({"recommendations": [
                {"anime_id": i, "explanation": str(i)} for i in [60, 50, 40, 30, 20]
            ]})),
        ]
        self.tools.execute_tool.side_effect = [json.dumps(source), json.dumps(filtered), '[]']
        response = self.run_agent()
        calls = self.tools.execute_tool.call_args_list
        self.assertEqual(list(calls[1].args[1]["sim_ids_scores"]), [10, 20, 30, 40, 50, 60])
        self.assertEqual(calls[1].args[1]["max_episodes"], 219)
        self.assertEqual(list(calls[2].args[1]["recommendations"]), [20, 30, 40, 50, 60])
        self.assertEqual([r.anime_id for r in response.parsed.recommendations], [20, 30, 40, 50, 60])

    def test_final_selection_rejects_missing_extra_and_duplicate_ids(self):
        for ids in ([20], [20, 30, 99], [20, 30, 30]):
            with self.subTest(ids=ids):
                self.client.models.generate_content.side_effect = [
                    tool_turn("find_similar_tool"), answer("Ready"),
                    answer(json.dumps({"recommendations": [
                        {"anime_id": i, "explanation": "Similar"} for i in ids
                    ]})),
                ]
                self.tools.execute_tool.return_value = '{"20": 1, "30": 0.5}'
                with self.assertRaisesRegex(AgentLoopError, "backend-selected"):
                    self.run_agent()

    def test_text_answer_requires_no_tools(self):
        final = answer()
        self.client.models.generate_content.return_value = final
        self.assertIs(self.run_agent(), final)
        self.assertEqual(final.parsed.model_dump(exclude_defaults=True), FINAL_ANSWER)
        self.tools.execute_tool.assert_not_called()

    def test_multiple_calls_are_rejected_before_execution(self):
        self.client.models.generate_content.side_effect = [
            tool_turn("search_anime_tool", "find_similar_tool"),
            tool_turn("search_anime_tool"), answer(), answer(),
        ]
        self.tools.execute_tool.return_value = '[]'
        self.run_agent()
        self.tools.execute_tool.assert_called_once_with("search_anime_tool", {})
        parts = self.client.models.generate_content.call_args_list[1].kwargs["contents"][2].parts
        self.assertEqual(len(parts), 2)
        self.assertTrue(all(p.function_response.response["error"]["code"] == "one_tool_per_turn" for p in parts))

    def test_tool_failures_are_sanitized_and_returned_to_model(self):
        self.client.models.generate_content.side_effect = [tool_turn("search_anime_tool"), answer(), answer()]
        self.tools.execute_tool.side_effect = RuntimeError("private credentials")
        with self.assertLogs("anirec.agent_service", level="ERROR"):
            self.run_agent()
        content = self.client.models.generate_content.call_args_list[1].kwargs["contents"][2]
        self.assertEqual(content.parts[0].function_response.response["error"]["code"], "tool_failed")
        self.assertNotIn("private credentials", str(content))

    def test_user_errors_reach_model_and_are_preserved_in_final_answer(self):
        for exc, code in (
            (MALUserNotFoundError("MyAnimeList user not found. Check the username and try again."), "user_not_found"),
            (MALProfileRestrictedError("This MyAnimeList list is private."), "profile_restricted"),
            (NoRatedAnimeError("No completed, scored TV anime found for this user"), "no_rated_anime"),
        ):
            with self.subTest(code=code):
                self.client.models.generate_content.side_effect = [
                    tool_turn("get_recs_with_username_tool"), answer("Cannot recommend"), answer(),
                ]
                self.tools.execute_tool.side_effect = exc
                response = self.run_agent()
                self.assertEqual(response.parsed.error, str(exc))
                self.assertEqual(response.parsed.recommendations, [])
                calls = self.client.models.generate_content.call_args_list
                tool_error = calls[-2].kwargs["contents"][2].parts[0].function_response.response["error"]
                self.assertEqual(tool_error, {"code": code, "message": str(exc)})

    def test_turn_limit_and_empty_responses(self):
        self.client.models.generate_content.return_value = tool_turn("search_anime_tool")
        self.tools.execute_tool.return_value = '[]'
        with self.assertRaisesRegex(AgentLoopError, "within 2"):
            self.run_agent(max_turns=2)
        self.assertEqual(self.client.models.generate_content.call_count, 2)
        for response in (types.GenerateContentResponse(), answer("")):
            self.client.models.generate_content.return_value = response
            with self.assertRaises(AgentLoopError):
                self.run_agent()

    def test_conversations_do_not_share_history(self):
        self.client.models.generate_content.return_value = answer()
        self.run_agent()
        self.run_agent()
        for call in self.client.models.generate_content.call_args_list[::2]:
            self.assertEqual(len(call.kwargs["contents"]), 1)

    def test_final_schema_is_tool_free_and_returns_ranked_recommendations(self):
        payload = {"recommendations": [
            {"anime_id": 30, "explanation": "Similar themes"},
            {"anime_id": 20, "explanation": "Similar setting"},
        ]}
        self.client.models.generate_content.side_effect = [answer("Try these"), answer(json.dumps(payload))]
        self.assertEqual(self.run_agent().parsed.model_dump(exclude_defaults=True), payload)
        first, final = self.client.models.generate_content.call_args_list
        self.assertIsNone(first.kwargs["config"].response_schema)
        config = final.kwargs["config"]
        self.assertIsNone(config.response_schema)
        self.assertEqual(config.response_json_schema, AgentAnswer.model_json_schema())
        self.assertEqual(config.response_mime_type, "application/json")
        self.assertIsNone(config.tools)
        self.assertEqual(final.kwargs["contents"][-2].parts[0].text, "Try these")

    def test_service_returns_sdk_response_when_parsing_fails(self):
        for text in ("not JSON", "", '{}',
                     '{"recommendations":[{"anime_id":0,"explanation":"x"}]}',
                     '{"recommendations":[{"anime_id":"20","explanation":"x"}]}'):
            with self.subTest(text=text):
                self.client.models.generate_content.side_effect = [answer(), answer(text)]
                self.assertIsNone(self.run_agent().parsed)


class AgentRouteTests(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        self.app.testing = True
        self.model_client = Mock()
        self.agent_tools = Mock()
        self.agent_tools.tool_schemas.return_value = []
        self.agent_tools.anime_data = {
            "30": {"title": "Anime Thirty", "main_picture": {"medium": "https://example.com/30.jpg"}},
            "20": {"title": "Anime Twenty", "main_picture": None},
        }
        self.app.config.update(GEMINI_CLIENT=self.model_client, AGENT_TOOLS=self.agent_tools)
        init_agent_tools(self.app)
        self.app.register_blueprint(agent_py, url_prefix="/agent")
        self.http = self.app.test_client()

    def test_route_returns_final_response_and_accepts_prompt(self):
        self.model_client.models.generate_content.return_value = answer()
        result = self.http.get("/agent/?prompt=Suggest+anime")
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json, FINAL_ANSWER)
        contents = self.model_client.models.generate_content.call_args.kwargs["contents"]
        self.assertEqual(contents[0].parts[0].text, "Suggest anime")

    def test_route_enriches_ids_and_omits_explanations(self):
        payload = {"recommendations": [
            {"anime_id": 30, "explanation": "Similar themes"},
            {"anime_id": 20, "explanation": "Similar setting"},
        ]}
        self.model_client.models.generate_content.side_effect = [answer("Ready"), answer(json.dumps(payload))]
        result = self.http.get("/agent/")
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json, {"recommendations": [
            {"anime_id": 30, "title": "Anime Thirty", "picture_link": "https://example.com/30.jpg"},
            {"anime_id": 20, "title": "Anime Twenty", "picture_link": None},
        ]})

    def test_missing_metadata_returns_contract_error(self):
        payload = {"recommendations": [{"anime_id": 999, "explanation": "Similar"}]}
        self.model_client.models.generate_content.side_effect = [answer("Ready"), answer(json.dumps(payload))]
        result = self.http.get("/agent/")
        self.assertEqual(result.status_code, 502)
        self.assertEqual(result.json["recommendations"], [])
        self.assertIn("error", result.json)

    def test_missing_username_returns_user_facing_error(self):
        self.agent_tools.execute_tool.side_effect = MALUserNotFoundError("MyAnimeList user not found. Check the username and try again.")
        self.model_client.models.generate_content.side_effect = [
            tool_turn("get_recs_with_username_tool"), answer("User not found"), answer(),
        ]
        result = self.http.get("/agent/?prompt=missinguser")
        self.assertEqual(result.status_code, 400)
        self.assertEqual(result.json, {
            "recommendations": [],
            "error": "MyAnimeList user not found. Check the username and try again.",
        })

    def test_invalid_prompt_is_rejected(self):
        self.assertEqual(self.http.get("/agent/?prompt=%20").status_code, 400)
        self.model_client.models.generate_content.assert_not_called()

    def test_invalid_final_answer_returns_json_error(self):
        self.model_client.models.generate_content.side_effect = [answer(), answer("not JSON")]
        result = self.http.get("/agent/")
        self.assertEqual(result.status_code, 502)
        self.assertIn("error", result.json)

    def test_provider_errors_return_json(self):
        for exc, status in (
            (errors.ServerError(503, {"error": {"message": "high demand"}}), 503),
            (errors.ClientError(400, {"error": {"message": "bad request"}}), 502),
            (httpx.ReadTimeout("timeout"), 504),
            (httpx.ConnectError("connection failed"), 502),
        ):
            with self.subTest(status=status):
                self.model_client.models.generate_content.side_effect = exc
                result = self.http.get("/agent/")
                self.assertEqual(result.status_code, status)
                self.assertIn("error", result.json)


if __name__ == "__main__":
    unittest.main()
