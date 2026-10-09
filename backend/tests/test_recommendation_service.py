import hashlib
import json
import unittest
from functools import partial
from unittest.mock import MagicMock, patch

import pandas as pd

from anirec.mal_client import MALClient, MALUserNotFoundError
from anirec.agent_tools import AgentTools
from anirec.recommendation_service import (
    get_recs_with_username, NoRatedAnimeError, build_recommendation_hash,
)


class RecommendationServiceTests(unittest.TestCase):
    def setUp(self):
        self.pool = MagicMock()
        self.cursor = self.pool.connection.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value
        self.cursor.fetchall.return_value = []
        self.metadata = {"20": {"title": "Naruto", "num_episodes": 220}}
        self.dependencies = dict(
            pool=self.pool, client_id="test", anime_data_client=MagicMock(),
            anime_data=self.metadata, builder=MagicMock(),
            recommender=MagicMock(), anime_df=MagicMock(),
        )
        self.mal_patch = patch("anirec.recommendation_service.MALClient")
        self.mal = self.mal_patch.start().return_value
        self.addCleanup(self.mal_patch.stop)
        self.mal.get_scores.return_value = {1: 9}
        self.model_patch = patch("anirec.recommendation_service.KNNRegressor")
        self.model = self.model_patch.start()
        self.addCleanup(self.model_patch.stop)
        self.model.return_value.get_recs.return_value = pd.DataFrame([
            {"anime_id": 20, "title": "Naruto", "picture_link": None, "predicted_score": 8.5},
        ])

    def test_cache_hit_skips_fit_and_preserves_scores(self):
        self.cursor.fetchall.return_value = [(1, "Naruto", None, 20, 1, 8.5)]
        result = get_recs_with_username(" user ", **self.dependencies)
        self.assertTrue(result["cached"])
        self.assertEqual(result["username"], "user")
        self.assertEqual(result["recommendations"][0]["predicted_score"], 8.5)
        self.model.assert_not_called()
        self.assertEqual(self.cursor.execute.call_count, 1)

    def test_cache_miss_fits_and_saves_scores(self):
        result = get_recs_with_username("user", **self.dependencies)
        self.assertFalse(result["cached"])
        self.model.return_value.fit.assert_called_once()
        self.assertEqual(self.cursor.execute.call_count, 2)
        self.assertEqual(self.cursor.execute.call_args.args[1][-1], [8.5])
        self.assertEqual(result["recommendations"][0]["predicted_score"], 8.5)

    def test_validation_and_empty_profile(self):
        for username, top_k in [("", 5), ("user", 51), ("user", 0)]:
            with self.assertRaises(ValueError):
                get_recs_with_username(username, top_k, **self.dependencies)
        self.mal.get_user_data.assert_not_called()
        self.mal.get_scores.return_value = {}
        with self.assertRaises(NoRatedAnimeError):
            get_recs_with_username("user", **self.dependencies)
        self.pool.connection.assert_not_called()

    def test_missing_user_propagates_through_service_and_tool(self):
        self.mal.get_user_data.side_effect = MALUserNotFoundError("MyAnimeList user not found.")
        provider = partial(get_recs_with_username, **self.dependencies)
        tools = AgentTools(self.dependencies["recommender"], self.metadata, provider)
        with self.assertRaises(MALUserNotFoundError):
            tools.execute_tool("get_recs_with_username_tool", {"username": "missing"})
        self.mal.get_scores.assert_not_called()
        self.pool.connection.assert_not_called()
        self.model.assert_not_called()

    @patch("anirec.mal_client.requests.get")
    def test_mal_404_raises_specific_user_error(self, get):
        get.return_value.status_code = 404
        with self.assertRaisesRegex(MALUserNotFoundError, "user not found"):
            MALClient("test").get_user_data("missing")

    def test_tool_uses_service_and_shared_metadata(self):
        provider = partial(get_recs_with_username, **self.dependencies)
        tools = AgentTools(self.dependencies["recommender"], self.metadata, provider)
        ranked = tools.get_recs_with_username_tool("user", limit=5)
        self.assertEqual(ranked, {20: 8.5})
        self.assertEqual(tools.filter_anime(ranked, max_episodes=50), {})
        self.assertEqual(tools.get_metadata(ranked)[0]["title"], "Naruto")

    def test_hash_preserves_existing_cache_keys(self):
        scores = {20: 9, 1: 8}
        payload = {
            "model_type": "knn", "model_name": "knn_all_available",
            "max_n_neighbors": None, "user_scores": scores,
        }
        expected = hashlib.md5(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        self.assertEqual(
            build_recommendation_hash(scores, "knn", "knn_all_available", None),
            expected,
        )
        self.assertEqual(
            build_recommendation_hash({1: 8, 20: 9}, "knn", "knn_all_available", None),
            expected,
        )

    def test_elapsed_time_reported_for_cache_hits_and_misses(self):
        for rows in ([], [(1, "Naruto", None, 20, 1, 8.5)]):
            with self.subTest(cached=bool(rows)):
                self.cursor.fetchall.return_value = rows
                with patch("anirec.recommendation_service.time.perf_counter", side_effect=[10, 10.125]):
                    result = get_recs_with_username("user", **self.dependencies)
                self.assertEqual(result["cache_pipeline_ms"], 125.0)
                self.assertEqual(result["cached"], bool(rows))


if __name__ == "__main__":
    unittest.main()
