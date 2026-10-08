"""Exercise real SDK response objects without API or database calls."""

import unittest
from unittest.mock import Mock

from flask import Flask
from google.genai import errors, types
import httpx

from anirec.agent_service import AgentLoopError, run_agent
from anirec.routes.agent import agent_py, init_agent_tools


def answer(text="Recommendations ready"):
    return types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(role="model", parts=[types.Part.from_text(text=text)]),
    )])


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

    def test_sequential_tools_keep_prompt_history_signatures_and_model(self):
        first, second, final = tool_turn("search_anime_tool"), tool_turn("find_similar_tool"), answer()
        self.client.models.generate_content.side_effect = [first, second, final]
        self.tools.execute_tool.side_effect = ['[{"anime_id": 20}]', '{"30": 0.9}']
        self.assertIs(self.run_agent(), final)
        calls = self.client.models.generate_content.call_args_list
        self.assertEqual([len(c.kwargs["contents"]) for c in calls], [1, 3, 5])
        self.assertEqual({c.kwargs["model"] for c in calls}, {"gemini-3.8-flash"})
        history = calls[2].kwargs["contents"]
        self.assertEqual(history[0].parts[0].text, "Find anime")
        self.assertEqual(history[1].parts[0].thought_signature, b"signature")
        self.assertEqual(history[2].parts[0].function_response.response, {"result": [{"anime_id": 20}]})
        self.assertEqual(history[2].parts[0].function_response.id, "0")
        self.assertEqual([turn.role for turn in history], ["user", "model", "user", "model", "user"])
        self.assertEqual(history[4].parts[0].function_response.response, {"30": 0.9})
        self.assertTrue(calls[0].kwargs["config"].automatic_function_calling.disable)

    def test_text_answer_requires_no_tools(self):
        final = answer()
        self.client.models.generate_content.return_value = final
        self.assertIs(self.run_agent(), final)
        self.tools.execute_tool.assert_not_called()

    def test_multiple_calls_are_rejected_before_execution(self):
        self.client.models.generate_content.side_effect = [
            tool_turn("search_anime_tool", "find_similar_tool"),
            tool_turn("search_anime_tool"), answer(),
        ]
        self.tools.execute_tool.return_value = '[]'
        self.run_agent()
        self.tools.execute_tool.assert_called_once_with("search_anime_tool", {})
        parts = self.client.models.generate_content.call_args_list[1].kwargs["contents"][2].parts
        self.assertEqual(len(parts), 2)
        self.assertTrue(all(p.function_response.response["error"]["code"] == "one_tool_per_turn" for p in parts))

    def test_tool_failures_are_sanitized_and_returned_to_model(self):
        self.client.models.generate_content.side_effect = [tool_turn("search_anime_tool"), answer()]
        self.tools.execute_tool.side_effect = RuntimeError("private credentials")
        with self.assertLogs("anirec.agent_service", level="ERROR"):
            self.run_agent()
        content = self.client.models.generate_content.call_args_list[1].kwargs["contents"][2]
        self.assertEqual(content.parts[0].function_response.response["error"]["code"], "tool_failed")
        self.assertNotIn("private credentials", str(content))

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
        for call in self.client.models.generate_content.call_args_list:
            self.assertEqual(len(call.kwargs["contents"]), 1)


class AgentRouteTests(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        self.app.testing = True
        self.model_client = Mock()
        self.agent_tools = Mock()
        self.agent_tools.tool_schemas.return_value = []
        self.app.config.update(GEMINI_CLIENT=self.model_client, AGENT_TOOLS=self.agent_tools)
        init_agent_tools(self.app)
        self.app.register_blueprint(agent_py, url_prefix="/agent")
        self.http = self.app.test_client()

    def test_route_returns_final_response_and_accepts_prompt(self):
        self.model_client.models.generate_content.return_value = answer()
        result = self.http.get("/agent/?prompt=Suggest+anime")
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json["candidates"][0]["content"]["parts"][0]["text"], "Recommendations ready")
        contents = self.model_client.models.generate_content.call_args.kwargs["contents"]
        self.assertEqual(contents[0].parts[0].text, "Suggest anime")

    def test_invalid_prompt_is_rejected(self):
        self.assertEqual(self.http.get("/agent/?prompt=%20").status_code, 400)
        self.model_client.models.generate_content.assert_not_called()

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
