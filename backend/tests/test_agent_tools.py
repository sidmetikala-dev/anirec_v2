"""Agent-tool contracts tested without network calls or model training."""

import copy
import json
import unittest
from unittest.mock import Mock, patch

from anirec.agent_tools import AgentTools
from anirec.anime_recommender import SimilarityRecommender


class AgentToolsTests(unittest.TestCase):
    def setUp(self):
        self.recommender = Mock(spec=SimilarityRecommender)
        self.provider = Mock()
        self.metadata = {
            "30": {
                "title": "Short finished anime", "num_episodes": 12,
                "main_picture": {"medium": "cover.jpg"},
            },
            "10": {
                "title": "Long finished anime", "num_episodes": 100,
            },
            "20": {
                "title": "Boundary anime", "num_episodes": 50,
                "main_picture": None,
            },
            "40": {"num_episodes": 24},
            "50": {},
        }
        self.tools = AgentTools(self.recommender, self.metadata, self.provider)
        self.ranked = {30: 9.5, 10: 9.0, 20: 8.5, 40: 8.0, 50: 7.5}

    def test_personalized_results_preserve_service_order_and_scores(self):
        self.provider.return_value = {"recommendations": [
            {"anime_id": 30, "predicted_score": 9.5},
            {"anime_id": 10, "predicted_score": 9.0},
        ]}
        result = self.tools.get_recs_with_username_tool("test_user", limit=2)
        self.provider.assert_called_once_with("test_user", top_k=2)
        self.assertEqual(list(result.items()), [(30, 9.5), (10, 9.0)])

    def test_personalized_empty_results_and_errors(self):
        self.provider.return_value = {"recommendations": []}
        self.assertEqual(self.tools.get_recs_with_username_tool("test_user"), {})
        self.provider.side_effect = ValueError("Username is required")
        with self.assertRaisesRegex(ValueError, "Username is required"):
            self.tools.get_recs_with_username_tool("")

    @patch("anirec.agent_tools.search_anime")
    def test_search_accepts_single_and_multiple_names(self, search):
        search.return_value = [{"anime_id": 20, "anime_name": "Naruto"}]
        for names in ("Naruto", ["Naruto", "Death Note"]):
            with self.subTest(names=names):
                self.assertEqual(self.tools.search_anime_tool(names, limit=3), search.return_value)
                search.assert_called_with(names, limit=3)

    def test_similarity_forwards_positive_and_negative_preferences(self):
        self.recommender.find_similar.return_value = {30: 0.9, 20: 0.7}
        result = self.tools.find_similar_tool([1], [2], negative_weight=0.25, limit=2)
        self.recommender.find_similar.assert_called_once_with([1], [2], 0.25, 2)
        self.assertEqual(list(result.items()), [(30, 0.9), (20, 0.7)])

    def test_similarity_allows_positive_only_requests(self):
        self.recommender.find_similar.return_value = {}
        self.assertEqual(self.tools.find_similar_tool([1]), {})
        self.recommender.find_similar.assert_called_once_with([1], None, 0.5, 100)

    def test_episode_filter_preserves_rank_and_does_not_mutate_inputs(self):
        original_metadata = copy.deepcopy(self.metadata)
        original_ranking = list(self.ranked.items())
        result = self.tools.filter_anime(self.ranked, max_episodes=50)
        self.assertEqual(list(result.items()), [(30, 9.5), (20, 8.5), (40, 8.0)])
        self.assertEqual(list(self.ranked.items()), original_ranking)
        self.assertEqual(self.metadata, original_metadata)

    def test_no_filters_preserve_all_candidates_and_order(self):
        self.assertEqual(list(self.tools.filter_anime(self.ranked).items()), list(self.ranked.items()))

    def test_shuffled_filter_input_returns_descending_scores(self):
        result = self.tools.filter_anime({40: 8.0, 10: 9.0, 20: 8.5, 30: 9.5}, max_episodes=50)
        self.assertEqual(list(result.items()), [(30, 9.5), (20, 8.5), (40, 8.0)])

    def test_unknown_episode_counts_and_missing_metadata_fail_episode_filter(self):
        self.assertEqual(self.tools.filter_anime({50: 1.0, 999: 0.5}, max_episodes=50), {})
        self.assertEqual(self.tools.filter_anime(self.ranked, max_episodes=1), {})
        self.assertEqual(self.tools.filter_anime({}, max_episodes=50), {})

    def test_metadata_preserves_rank_and_handles_missing_values(self):
        result = self.tools.get_metadata({30: 9.5, 20: 8.5, 999: 0.5})
        self.assertEqual(result, [
            {"anime_id": 30, "title": "Short finished anime", "picture_link": "cover.jpg", "score": 9.5},
            {"anime_id": 20, "title": "Boundary anime", "picture_link": None, "score": 8.5},
            {"anime_id": 999, "title": None, "picture_link": None, "score": 0.5},
        ])
        self.assertEqual(json.loads(json.dumps(result)), result)

    def test_metadata_custom_columns_and_empty_results(self):
        self.assertEqual(
            self.tools.get_metadata({30: 9.5}, ["anime_id", "num_episodes"]),
            [{"anime_id": 30, "num_episodes": 12}],
        )
        self.assertEqual(self.tools.get_metadata({}), [])

    def test_sdk_schemas_export_all_tools_and_formats(self):
        expected = {"get_recs_with_username_tool", "search_anime_tool", "find_similar_tool",
                    "filter_anime", "get_metadata"}
        for format in ("chat_completions", "responses", "anthropic"):
            schemas = json.loads(json.dumps(self.tools.tool_schemas(format)))
            definitions = [tool["function"] for tool in schemas] if format == "chat_completions" else schemas
            self.assertEqual({tool["name"] for tool in definitions}, expected)
            for tool in definitions:
                schema = tool["input_schema" if format == "anthropic" else "parameters"]
                self.assertFalse(schema["additionalProperties"])
                self.assertTrue(tool["description"])
                if tool["name"] == "filter_anime":
                    self.assertEqual(set(schema["properties"]), {"sim_ids_scores", "max_episodes"})
        with self.assertRaises(ValueError):
            self.tools.tool_schemas("invalid")

    def test_sdk_dispatch_uses_defaults_and_serializes_results(self):
        self.recommender.find_similar.return_value = {30: 0.9, 20: 0.7}
        result = self.tools.execute_tool("find_similar_tool", '{"positive_anime_ids": [1]}')
        self.assertEqual(json.loads(result), {"30": 0.9, "20": 0.7})
        self.recommender.find_similar.assert_called_once_with([1], None, 0.5, 100)

    def test_sdk_rankings_round_trip_through_filter_and_metadata(self):
        filtered = self.tools.execute_tool("filter_anime", {
            "sim_ids_scores": {"30": 9.5, "10": 9.0, "20": 8.5},
            "max_episodes": 50,
        })
        self.assertEqual(list(json.loads(filtered)), ["30", "20"])
        output = self.tools.execute_tool("get_metadata", {
            "recommendations": json.loads(filtered),
        })
        self.assertEqual([row["anime_id"] for row in json.loads(output)], [30, 20])

    def test_sdk_rejects_invalid_arguments_before_backend_execution(self):
        for name, arguments in (
            ("get_recs_with_username_tool", {"username": "   "}),
            ("get_recs_with_username_tool", {"username": "user", "limit": 501}),
            ("get_recs_with_username_tool", {"username": "user", "limit": True}),
            ("get_recs_with_username_tool", {"username": "user", "unexpected": 1}),
            ("find_similar_tool", {"positive_anime_ids": []}),
            ("find_similar_tool", {"positive_anime_ids": [-1]}),
            ("find_similar_tool", {"positive_anime_ids": [1], "negative_weight": 2}),
            ("search_anime_tool", {"anime_names": []}),
            ("filter_anime", {"sim_ids_scores": {"bad": 1}}),
            ("filter_anime", {"sim_ids_scores": {"30": float("nan")}}),
            ("get_metadata", {"recommendations": {}, "columns_to_get": []}),
            ("get_recs_with_username_tool", "{broken json"),
            ("get_recs_with_username_tool", "[]"),
        ):
            with self.subTest(name=name, arguments=arguments):
                output = json.loads(self.tools.execute_tool(name, arguments))
                self.assertEqual(output["error"]["code"], "invalid_arguments")
        self.provider.assert_not_called()
        self.recommender.find_similar.assert_not_called()

    def test_sdk_dispatch_allowlist_and_backend_errors(self):
        output = json.loads(self.tools.execute_tool("__init__", {}))
        self.assertEqual(output["error"]["code"], "unknown_tool")
        self.provider.side_effect = RuntimeError("backend unavailable")
        with self.assertRaisesRegex(RuntimeError, "backend unavailable"):
            self.tools.execute_tool("get_recs_with_username_tool", {"username": "user"})

    def test_metadata_cannot_overwrite_ranked_score_or_id(self):
        self.metadata["30"].update({"score": 1, "anime_id": 999})
        result = self.tools.execute_tool("get_metadata", {"recommendations": {"30": 9.5}})
        self.assertEqual(json.loads(result)[0]["score"], 9.5)
        self.assertEqual(json.loads(result)[0]["anime_id"], 30)


if __name__ == "__main__":
    unittest.main()
