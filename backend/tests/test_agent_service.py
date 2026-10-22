"""Exercise real SDK response objects without API or database calls."""

import json
import unittest
from unittest.mock import Mock, patch

from pydantic import ValidationError
from flask import Flask
import groq
from groq.types.chat import ChatCompletion
import httpx

from anirec.agent_service import AgentAnswer, AgentLoopError, run_agent, _create_completion, _format_answer
from anirec.mal_client import MALUserNotFoundError, MALProfileRestrictedError
from anirec.recommendation_service import NoRatedAnimeError
from anirec.routes.agent import agent_py, init_agent_tools


FINAL_ANSWER = {"recommendations": []}


def completion(message):
    return ChatCompletion(id="test", object="chat.completion", created=0,
                          model="llama-3.1-8b-instant", choices=[{
                              "index": 0, "finish_reason": "stop", "message": message,
                          }])


def answer(text=None):
    return completion({"role": "assistant", "content": json.dumps(FINAL_ANSWER) if text is None else text})


def tool_turn(*names):
    return completion({"role": "assistant", "content": None, "tool_calls": [
        {"type": "function", "id": str(i), "function": {"name": name, "arguments": "{}"}}
        for i, name in enumerate(names)
    ]})


class AgentServiceTests(unittest.TestCase):
    def setUp(self):
        self.client = Mock()
        self.tools = Mock()
        self.tools.get_metadata.return_value = []
        self.tools.anime_data = {"30": {"title": "Test Anime Thirty"}}

    def run_agent(self, **kwargs):
        return run_agent(self.client, self.tools, "Find anime", tools=[], **kwargs)

    def test_similarity_rejects_guessed_ids_then_recovers_after_search(self):
        def similar(ids, negatives=None):
            turn = tool_turn("find_similar_tool")
            turn.choices[0].message.tool_calls[0].function.arguments = json.dumps({
                "positive_anime_ids": ids, "negative_anime_ids": negatives or []})
            return turn
        self.client.chat.completions.create.side_effect = [
            similar([785]), similar(["20"]), tool_turn("lookup_anime_ids"),
            similar([20], [1735]), similar([20]), answer(), answer()]
        self.tools.execute_tool.side_effect = [
            '{"result":[{"anime_id":20}]}', '{}']
        self.run_agent()
        self.assertEqual([c.args[0] for c in self.tools.execute_tool.call_args_list],
                         ["lookup_anime_ids", "find_similar_tool"])
        self.assertEqual(self.tools.execute_tool.call_args.args[1]["positive_anime_ids"], [20])
        history = self.client.chat.completions.create.call_args_list[4].kwargs["messages"]
        errors = [json.loads(m["content"])["error"] for m in history
                  if m["role"] == "tool" and "error" in json.loads(m["content"])]
        self.assertEqual(len(errors), 3)
        self.assertTrue(all(e["code"] == "unresolved_anime_ids" for e in errors))

    def test_recovery_uses_current_search_matches_without_hardcoded_ids(self):
        self.tools.execute_tool.side_effect = [
            '{"result":[{"anime_id":321,"anime_name":"Example Title"}]}', '{}']
        similar = tool_turn("find_similar_tool")
        similar.choices[0].message.tool_calls[0].function.arguments = '{"positive_anime_ids":[321]}'
        self.client.chat.completions.create.side_effect = [tool_turn("lookup_anime_ids"),
            answer("Done"), similar, answer("Ready"), answer()]
        trace = []
        self.run_agent(trace=trace)
        correction = next(e["message"] for e in trace if e["type"] == "premature_answer")
        self.assertIn('"anime_id": 321', correction)
        self.assertIn("Example Title", correction)
        self.assertIn("Do not repeat successful searches", correction)
        self.assertIn("call find_similar_tool", correction)
        self.assertNotIn("Naruto", correction)
        self.assertEqual(self.tools.execute_tool.call_args.args[1], {"positive_anime_ids": [321]})

    def test_null_negative_ids_do_not_bypass_positive_id_guard(self):
        turn = tool_turn("find_similar_tool")
        turn.choices[0].message.tool_calls[0].function.arguments = '{"positive_anime_ids":[1],"negative_anime_ids":null}'
        self.client.chat.completions.create.return_value = turn
        trace = []
        with self.assertRaises(AgentLoopError):
            self.run_agent(max_turns=1, trace=trace)
        self.tools.execute_tool.assert_not_called()
        self.assertEqual(trace[0]["result"]["error"]["code"], "unresolved_anime_ids")

    def test_search_ids_do_not_leak_between_requests(self):
        turn = tool_turn("find_similar_tool")
        turn.choices[0].message.tool_calls[0].function.arguments = '{"positive_anime_ids":[20]}'
        self.client.chat.completions.create.side_effect = [
            tool_turn("lookup_anime_ids"), answer(), turn, answer()]
        self.tools.execute_tool.return_value = '{"result":[{"anime_id":20}]}'
        with self.assertRaises(AgentLoopError):
            self.run_agent(max_turns=2)
        with self.assertRaises(AgentLoopError):
            self.run_agent(max_turns=2)
        self.assertEqual(self.tools.execute_tool.call_count, 1)

    def test_all_tools_remain_available_after_lookup(self):
        definitions = [{"type":"function", "function":{"name": name}} for name in
                       ("lookup_anime_ids", "find_similar_tool", "get_metadata", "filter_anime",
                        "get_recs_with_username_tool")]
        similar = tool_turn("find_similar_tool")
        similar.choices[0].message.tool_calls[0].function.arguments = '{"positive_anime_ids":[20],"negative_anime_ids":[99]}'
        valid = tool_turn("find_similar_tool")
        valid.choices[0].message.tool_calls[0].function.arguments = '{"positive_anime_ids":[20],"negative_anime_ids":[99]}'
        self.client.chat.completions.create.side_effect = [tool_turn("lookup_anime_ids"), similar,
            tool_turn("lookup_anime_ids"), valid, answer("Ready"), answer()]
        self.tools.execute_tool.side_effect = ['[{"anime_id":20}]', '[{"anime_id":99}]', '{}']
        run_agent(self.client, self.tools, "similar to a title but exclude another", tools=definitions)
        requests = self.client.chat.completions.create.call_args_list
        names = lambda i: {t["function"]["name"] for t in requests[i].kwargs["tools"]}
        self.assertEqual(names(1), names(0))
        self.assertIn("lookup_anime_ids", names(2))
        self.assertEqual(names(3), names(0))
        self.assertIn("filter_anime", names(4))

    def test_final_answer_uses_json_mode_and_schema_prompt(self):
        self.client.chat.completions.create.return_value = answer()
        _format_answer(self.client, [{"role":"user","content":"Find anime"}], "test", ranking={}, agent_tools=self.tools)
        kwargs = self.client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["response_format"], {"type": "json_object"})
        self.assertIn(json.dumps(AgentAnswer.model_json_schema()), kwargs["messages"][-1]["content"])
        self.assertNotIn("tools", kwargs)

    def test_sequential_tools_keep_prompt_history_signatures_and_model(self):
        first, second, final = tool_turn("lookup_anime_ids"), tool_turn("find_similar_tool"), answer()
        self.client.chat.completions.create.side_effect = [first, second, final, answer(json.dumps({
            "recommendations": [{"anime_id": 30, "explanation": "Similar themes"}],
        }))]
        self.tools.execute_tool.side_effect = ['[{"anime_id": 20}]', '{"30": 0.9}']
        self.assertEqual([r.anime_id for r in self.run_agent().parsed.recommendations], [30])
        calls = self.client.chat.completions.create.call_args_list
        self.assertEqual([len(c.kwargs["messages"]) for c in calls], [2, 4, 6, 2])
        self.assertEqual({c.kwargs["model"] for c in calls}, {"llama-3.1-8b-instant"})
        history = calls[2].kwargs["messages"]
        self.assertEqual(history[1]["content"], "Find anime")
        self.assertEqual(history[2]["tool_calls"][0]["function"]["name"], "lookup_anime_ids")
        self.assertEqual(json.loads(history[3]["content"])["result"], [{"anime_id": 20}])
        self.assertIn("Next, call find_similar_tool", json.loads(history[3]["content"])["message"])
        self.assertIn("get_metadata", json.loads(history[3]["content"])["message"])
        self.assertEqual(history[3]["tool_call_id"], "0")
        self.assertEqual([turn["role"] for turn in history], ["system", "user", "assistant", "tool", "assistant", "tool"])
        self.assertEqual(json.loads(history[5]["content"])["selected_anime_ids"], [30])
        summary = json.loads(history[5]["content"])
        self.assertEqual(summary["selected_candidates"], [{"anime_id": 30, "title": "Test Anime Thirty"}])
        self.assertNotIn("score", summary["selected_candidates"][0])

    def test_backend_controls_filter_input_metadata_selection_and_final_order(self):
        self.tools.get_metadata.return_value = [{"anime_id": 20, "title": "Test title", "synopsis": "Test synopsis"}]
        source = {"10": 1.0, "20": 0.9, "30": 0.8, "40": 0.7, "50": 0.6, "60": 0.5}
        filtered = {"20": 0.9, "30": 0.8, "40": 0.7, "50": 0.6, "60": 0.5}
        filter_turn = tool_turn("filter_anime")
        filter_turn.choices[0].message.tool_calls[0].function.arguments = json.dumps({
            "max_episodes": 219, "sim_ids_scores": {"60": 99},
        })
        self.client.chat.completions.create.side_effect = [
            tool_turn("find_similar_tool"), filter_turn, tool_turn("get_metadata"), answer("Ready"),
            answer(json.dumps({"recommendations": [
                {"anime_id": i, "explanation": str(i)} for i in [60, 50, 40, 30, 20]
            ]})),
        ]
        self.tools.execute_tool.side_effect = [json.dumps(source), json.dumps(filtered), "[]"]
        response = self.run_agent()
        calls = self.tools.execute_tool.call_args_list
        self.assertEqual(list(calls[1].args[1]["sim_ids_scores"]), [10, 20, 30, 40, 50, 60])
        self.assertEqual(calls[1].args[1]["max_episodes"], 219)
        self.assertEqual(len(calls), 3)
        self.assertEqual(list(self.tools.get_metadata.call_args.args[0]), [20, 30, 40, 50, 60])
        self.assertEqual(self.client.chat.completions.create.call_count, 5)
        final_request = self.client.chat.completions.create.call_args.kwargs
        self.assertIn("Test synopsis", final_request["messages"][-1]["content"])
        self.assertNotIn("tools", final_request)
        self.assertEqual([r.anime_id for r in response.parsed.recommendations], [20, 30, 40, 50, 60])

    def test_metadata_after_ranking_returns_facts_and_allows_later_filtering(self):
        metadata = tool_turn("get_metadata")
        metadata.choices[0].message.tool_calls[0].function.arguments = '{"anime_ids":[99],"columns_to_get":["num_episodes"]}'
        self.client.chat.completions.create.side_effect = [tool_turn("find_similar_tool"), metadata,
            tool_turn("filter_anime"), answer("Ready"), answer()]
        self.tools.execute_tool.side_effect = ['{"20":1}', '[{"num_episodes":12}]', '{}']
        self.run_agent()
        calls = self.tools.execute_tool.call_args_list
        self.assertEqual([c.args[0] for c in calls], ["find_similar_tool", "get_metadata", "filter_anime"])
        self.assertEqual(calls[1].args[1]["anime_ids"], [99])
        self.assertEqual(calls[2].args[1]["sim_ids_scores"], {20:1})
        history = self.client.chat.completions.create.call_args_list[2].kwargs["messages"]
        self.assertEqual(json.loads(history[-1]["content"])["result"], [{"num_episodes":12}])

    def test_final_prompt_omits_history_and_truncates_synopses(self):
        self.tools.get_metadata.return_value = [{"anime_id":20,"title":"Title","synopsis":"x" * 5000}]
        self.client.chat.completions.create.side_effect = [tool_turn("find_similar_tool"), answer("Verbose intermediate text"),
            answer(json.dumps({"recommendations":[{"anime_id":20,"explanation":"Matches"}]}))]
        self.tools.execute_tool.return_value = '{"20":0.9}'
        self.run_agent()
        messages = self.client.chat.completions.create.call_args.kwargs["messages"]
        self.assertEqual(len(messages), 2)
        self.assertNotIn("Verbose intermediate text", json.dumps(messages))
        self.assertNotIn("x" * 801, messages[-1]["content"])
        self.assertIn("x" * 800, messages[-1]["content"])

    @patch("anirec.llm_client.time.sleep")
    def test_retry_respects_header_and_recovers(self, sleep):
        exc = groq.RateLimitError("limited", response=httpx.Response(429, headers={"retry-after":"2"},
            request=httpx.Request("GET", "https://api.groq.com")), body={})
        self.client.chat.completions.create.side_effect = [exc, answer()]
        _create_completion(self.client, model="test", messages=[])
        sleep.assert_called_once_with(2.1)

    @patch("anirec.llm_client.time.sleep")
    def test_retry_is_bounded_and_uses_message_delay(self, sleep):
        exc = groq.RateLimitError("Please try again in 14.415s.", response=httpx.Response(429,
            request=httpx.Request("GET", "https://api.groq.com")), body={})
        self.client.chat.completions.create.side_effect = exc
        with self.assertRaises(groq.RateLimitError):
            _create_completion(self.client, model="test", messages=[])
        self.assertEqual(self.client.chat.completions.create.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    @patch("anirec.llm_client.time.sleep")
    def test_long_retry_delay_is_not_shortened(self, sleep):
        exc = groq.RateLimitError("limited", response=httpx.Response(429, headers={"retry-after":"3600"},
            request=httpx.Request("GET", "https://api.groq.com")), body={})
        self.client.chat.completions.create.side_effect = exc
        with self.assertRaises(groq.RateLimitError):
            _create_completion(self.client, model="test", messages=[])
        sleep.assert_not_called()

    def test_reference_metadata_still_executes_before_ranking(self):
        self.client.chat.completions.create.side_effect = [
            tool_turn("get_metadata"), answer("Ready"), answer(),
        ]
        self.tools.execute_tool.return_value = '[{"anime_id":20,"num_episodes":220}]'
        with self.assertRaises(AgentLoopError):
            self.run_agent(max_turns=2)
        self.tools.execute_tool.assert_called_once_with("get_metadata", {})
        self.tools.get_metadata.assert_not_called()

    def test_final_selection_rejects_missing_extra_and_duplicate_ids(self):
        for ids in ([20], [20, 30, 99], [20, 30, 30]):
            with self.subTest(ids=ids):
                self.client.chat.completions.create.side_effect = [
                    tool_turn("find_similar_tool"), answer("Ready"),
                    answer(json.dumps({"recommendations": [
                        {"anime_id": i, "explanation": "Similar"} for i in ids
                    ]})),
                ]
                self.tools.execute_tool.return_value = '{"20": 1, "30": 0.5}'
                with self.assertRaisesRegex(AgentLoopError, "backend-selected"):
                    self.run_agent()

    def test_premature_answer_recovers_with_backend_ranking(self):
        self.client.chat.completions.create.side_effect = [answer("Here are guessed recommendations"),
            tool_turn("get_recs_with_username_tool"), answer("Ready"), answer()]
        self.tools.execute_tool.return_value = '{}'
        trace = []
        response = self.run_agent(trace=trace)
        self.assertEqual(response.parsed.recommendations, [])
        self.assertEqual(trace[0]["type"], "premature_answer")
        self.assertTrue(trace[1]["executed"])
        self.assertIn("No backend ranking exists", self.client.chat.completions.create.call_args_list[1].kwargs["messages"][-1]["content"])

    def test_text_answer_without_ranking_is_rejected_and_traced(self):
        self.client.chat.completions.create.return_value = answer('{"recommendations":[{"anime_id":785,"explanation":"Guessed"}]}')
        trace = []
        with self.assertRaisesRegex(AgentLoopError, "within 2"):
            self.run_agent(max_turns=2, trace=trace)
        self.assertEqual([e["type"] for e in trace], ["premature_answer"] * 2)
        self.assertTrue(all("Guessed" in e["response_text"] for e in trace))
        self.assertNotIn("response_format", self.client.chat.completions.create.call_args.kwargs)
        self.tools.execute_tool.assert_not_called()

    def test_provider_malformed_call_recovers_without_executing_a_phantom_tool(self):
        malformed = answer("")
        malformed.choices[0].finish_reason = "function_call_filter: MALFORMED_FUNCTION_CALL"
        self.client.chat.completions.create.side_effect = [tool_turn("lookup_anime_ids"),
            malformed, tool_turn("get_recs_with_username_tool"), answer("Ready"), answer()]
        self.tools.execute_tool.side_effect = ['[{"anime_id":20}]', '{}']
        trace = []
        response = self.run_agent(trace=trace)
        self.assertEqual(response.parsed.recommendations, [])
        self.assertEqual(self.tools.execute_tool.call_count, 2)
        event = next(e for e in trace if e["type"] == "malformed_tool_call")
        self.assertTrue(event["retry"])
        history = self.client.chat.completions.create.call_args_list[2].kwargs["messages"]
        self.assertEqual(history[-1]["role"], "user")
        self.assertIn("not executed", history[-1]["content"])
        self.assertEqual(json.loads(history[3]["content"])["result"], [{"anime_id":20}])

    def test_provider_malformed_call_retries_are_bounded(self):
        malformed = answer("")
        malformed.choices[0].finish_reason = "function_call_filter: MALFORMED_FUNCTION_CALL"
        self.client.chat.completions.create.return_value = malformed
        trace = []
        with self.assertRaisesRegex(AgentLoopError, "repeatedly returned malformed"):
            self.run_agent(trace=trace)
        self.assertEqual(self.client.chat.completions.create.call_count, 3)
        self.assertEqual([e["retry"] for e in trace], [True, True, False])
        self.tools.execute_tool.assert_not_called()
        with self.assertRaisesRegex(AgentLoopError, "within 1"):
            self.run_agent(max_turns=1)

    def test_malformed_tool_arguments_return_recoverable_errors(self):
        for arguments in ('{broken', '[]', 'null'):
            with self.subTest(arguments=arguments):
                turn = tool_turn("lookup_anime_ids")
                turn.choices[0].message.tool_calls[0].function.arguments = arguments
                self.client.chat.completions.create.side_effect = [turn, answer("Retry later"), answer()]
                with self.assertRaises(AgentLoopError):
                    self.run_agent(max_turns=3)
                history = self.client.chat.completions.create.call_args_list[-2].kwargs["messages"]
                self.assertEqual(json.loads(history[3]["content"])["error"]["code"], "invalid_arguments")
                self.tools.execute_tool.assert_not_called()

    def test_tool_requests_disable_parallel_calls(self):
        self.client.chat.completions.create.return_value = answer()
        definitions = [{"type": "function", "function": {"name": "lookup_anime_ids"}}]
        with self.assertRaises(AgentLoopError):
            run_agent(self.client, self.tools, "Find anime", tools=definitions, max_turns=1)
        request = self.client.chat.completions.create.call_args_list[0].kwargs
        self.assertFalse(request["parallel_tool_calls"])
        self.assertEqual(request["tools"], definitions)

    @patch.dict("os.environ", {"GROQ_REASONING_EFFORT": "medium"})
    def test_gpt_oss_uses_configured_reasoning_and_larger_output_budget(self):
        self.client.chat.completions.create.return_value = answer()
        with self.assertRaises(AgentLoopError):
            run_agent(self.client, self.tools, "Find anime", tools=[], model="openai/gpt-oss-20b", max_turns=1)
        for call in self.client.chat.completions.create.call_args_list:
            self.assertEqual(call.kwargs["reasoning_effort"], "medium")
            self.assertEqual(call.kwargs["max_completion_tokens"], 2048)

    def test_truncated_empty_response_reports_token_limit(self):
        response = answer("")
        response.choices[0].finish_reason = "length"
        self.client.chat.completions.create.return_value = response
        with self.assertLogs("anirec.agent_service", level="WARNING") as logs:
            with self.assertRaisesRegex(AgentLoopError, "output token limit"):
                self.run_agent()
        self.assertIn("finish_reason=length", logs.output[0])

    def test_multiple_calls_are_rejected_before_execution(self):
        self.client.chat.completions.create.side_effect = [
            tool_turn("lookup_anime_ids", "find_similar_tool"),
            tool_turn("lookup_anime_ids"), answer(), answer(),
        ]
        self.tools.execute_tool.return_value = '[]'
        with self.assertRaises(AgentLoopError):
            self.run_agent(max_turns=3)
        self.tools.execute_tool.assert_called_once_with("lookup_anime_ids", {})
        parts = self.client.chat.completions.create.call_args_list[1].kwargs["messages"][3:]
        self.assertEqual(len(parts), 2)
        self.assertTrue(all(json.loads(p["content"])["error"]["code"] == "one_tool_per_turn" for p in parts))

    def test_tool_failures_are_sanitized_and_returned_to_model(self):
        self.client.chat.completions.create.side_effect = [tool_turn("lookup_anime_ids"), answer(), answer()]
        self.tools.execute_tool.side_effect = RuntimeError("private credentials")
        with self.assertLogs("anirec.agent_service", level="ERROR"):
            with self.assertRaises(AgentLoopError):
                self.run_agent(max_turns=2)
        content = self.client.chat.completions.create.call_args_list[1].kwargs["messages"][3]
        self.assertEqual(json.loads(content["content"])["error"]["code"], "tool_failed")
        self.assertNotIn("private credentials", str(content))

    def test_user_errors_reach_model_and_are_preserved_in_final_answer(self):
        for exc, code in (
            (MALUserNotFoundError("MyAnimeList user not found. Check the username and try again."), "user_not_found"),
            (MALProfileRestrictedError("This MyAnimeList list is private."), "profile_restricted"),
            (NoRatedAnimeError("No completed, scored TV anime found for this user"), "no_rated_anime"),
        ):
            with self.subTest(code=code):
                self.client.chat.completions.create.side_effect = [
                    tool_turn("get_recs_with_username_tool"), answer("Cannot recommend"), answer(),
                ]
                self.tools.execute_tool.side_effect = exc
                response = self.run_agent()
                self.assertEqual(response.parsed.error, str(exc))
                self.assertEqual(response.parsed.recommendations, [])
                calls = self.client.chat.completions.create.call_args_list
                tool_error = calls[-2].kwargs["messages"][3]["content"]
                tool_error = json.loads(tool_error)["error"]
                self.assertEqual(tool_error, {"code": code, "message": str(exc)})

    def test_turn_limit_and_empty_responses(self):
        self.client.chat.completions.create.return_value = tool_turn("lookup_anime_ids")
        self.tools.execute_tool.return_value = '[]'
        with self.assertRaisesRegex(AgentLoopError, "within 2"):
            self.run_agent(max_turns=2)
        self.assertEqual(self.client.chat.completions.create.call_count, 2)
        for response in (ChatCompletion(id="empty", object="chat.completion", created=0, model="test", choices=[]), answer("")):
            self.client.chat.completions.create.return_value = response
            with self.assertRaises(AgentLoopError):
                self.run_agent()

    def test_conversations_do_not_share_history(self):
        self.client.chat.completions.create.return_value = answer()
        for _ in range(2):
            with self.assertRaises(AgentLoopError):
                self.run_agent(max_turns=1)
        for call in self.client.chat.completions.create.call_args_list:
            self.assertEqual(len(call.kwargs["messages"]), 2)

    def test_final_schema_is_tool_free_and_returns_ranked_recommendations(self):
        payload = {"recommendations": [
            {"anime_id": 30, "explanation": "Similar themes"},
            {"anime_id": 20, "explanation": "Similar setting"},
        ]}
        self.client.chat.completions.create.side_effect = [tool_turn("get_recs_with_username_tool"), answer("Try these"), answer(json.dumps(payload))]
        self.tools.execute_tool.return_value = '{"30":1,"20":0.9}'
        self.tools.get_metadata.return_value = []
        self.assertEqual(self.run_agent().parsed.model_dump(exclude_defaults=True), payload)
        first, _, final = self.client.chat.completions.create.call_args_list
        self.assertNotIn("response_format", first.kwargs)
        self.assertEqual(final.kwargs["response_format"], {"type": "json_object"})
        self.assertNotIn("tools", final.kwargs)
        self.assertEqual(len(final.kwargs["messages"]), 2)
        self.assertIn("Original request: Find anime", final.kwargs["messages"][-1]["content"])

    def test_service_returns_sdk_response_when_parsing_fails(self):
        for text in ("not JSON", "", '{}',
                     '{"recommendations":[{"anime_id":0,"explanation":"x"}]}',
                     '{"recommendations":[{"anime_id":"20","explanation":"x"}]}'):
            with self.subTest(text=text):
                self.client.chat.completions.create.return_value = answer(text)
                self.assertIsNone(_format_answer(self.client, [{"role":"user","content":"Find anime"}], "test", ranking={}, agent_tools=self.tools).parsed)


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
        self.app.config.update(GROQ_CLIENT=self.model_client, AGENT_TOOLS=self.agent_tools)
        init_agent_tools(self.app)
        self.app.register_blueprint(agent_py, url_prefix="/agent")
        self.http = self.app.test_client()

    def test_route_returns_final_response_and_accepts_prompt(self):
        self.model_client.chat.completions.create.side_effect = [tool_turn("get_recs_with_username_tool"), answer(), answer()]
        self.agent_tools.execute_tool.return_value = "{}"
        self.agent_tools.get_metadata.return_value = []
        result = self.http.get("/agent/?prompt=Suggest+anime")
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json, FINAL_ANSWER)
        contents = self.model_client.chat.completions.create.call_args.kwargs["messages"]
        self.assertIn("Original request: Suggest anime\n", contents[1]["content"])

    def test_route_enriches_ids_and_includes_details(self):
        payload = {"recommendations": [
            {"anime_id": 30, "explanation": "Similar themes"},
            {"anime_id": 20, "explanation": "Similar setting"},
        ]}
        self.model_client.chat.completions.create.side_effect = [tool_turn("get_recs_with_username_tool"), answer("Ready"), answer(json.dumps(payload))]
        self.agent_tools.execute_tool.return_value = json.dumps({str(r["anime_id"]): 1 for r in payload["recommendations"]})
        self.agent_tools.get_metadata.return_value = []
        result = self.http.get("/agent/")
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json, {"recommendations": [
            {"anime_id": 30, "title": "Anime Thirty", "picture_link": "https://example.com/30.jpg",
             "explanation": "Similar themes", "synopsis": None, "genres": [],
             "mean_rating": None, "num_list_users": None, "num_episodes": None},
            {"anime_id": 20, "title": "Anime Twenty", "picture_link": None,
             "explanation": "Similar setting", "synopsis": None, "genres": [],
             "mean_rating": None, "num_list_users": None, "num_episodes": None},
        ]})

    def test_missing_metadata_returns_contract_error(self):
        payload = {"recommendations": [{"anime_id": 999, "explanation": "Similar"}]}
        self.model_client.chat.completions.create.side_effect = [tool_turn("get_recs_with_username_tool"), answer("Ready"), answer(json.dumps(payload))]
        self.agent_tools.execute_tool.return_value = json.dumps({str(r["anime_id"]): 1 for r in payload["recommendations"]})
        self.agent_tools.get_metadata.return_value = []
        result = self.http.get("/agent/")
        self.assertEqual(result.status_code, 502)
        self.assertEqual(result.json["recommendations"], [])
        self.assertIn("error", result.json)

    def test_missing_username_returns_user_facing_error(self):
        self.agent_tools.execute_tool.side_effect = MALUserNotFoundError("MyAnimeList user not found. Check the username and try again.")
        self.model_client.chat.completions.create.side_effect = [
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
        self.model_client.chat.completions.create.assert_not_called()

    def test_invalid_final_answer_returns_json_error(self):
        self.model_client.chat.completions.create.side_effect = [tool_turn("get_recs_with_username_tool"), answer(), answer("not JSON")]
        self.agent_tools.execute_tool.return_value = "{}"
        self.agent_tools.get_metadata.return_value = []
        result = self.http.get("/agent/")
        self.assertEqual(result.status_code, 502)
        self.assertIn("error", result.json)

    def test_provider_errors_return_json(self):
        for exc, status in (
            (groq.NotFoundError("model unavailable", response=httpx.Response(404, request=httpx.Request("GET", "https://api.groq.com")), body={}), 502),
            (groq.RateLimitError("rate limited", response=httpx.Response(429, request=httpx.Request("GET", "https://api.groq.com")), body={}), 503),
            (groq.InternalServerError("high demand", response=httpx.Response(503, request=httpx.Request("GET", "https://api.groq.com")), body={}), 503),
            (groq.BadRequestError("bad request", response=httpx.Response(400, request=httpx.Request("GET", "https://api.groq.com")), body={}), 502),
            (httpx.ReadTimeout("timeout"), 504),
            (httpx.ConnectError("connection failed"), 502),
        ):
            with self.subTest(status=status):
                self.model_client.chat.completions.create.side_effect = exc
                result = self.http.get("/agent/")
                self.assertEqual(result.status_code, status)
                self.assertIn("error", result.json)


if __name__ == "__main__":
    unittest.main()
