# User query
#    ↓
# LLM / agent
#    ↓
# Intent + constraints
#    ↓
# Tools
#    ├── search_anime("Naruto")
#    ├── find_similar(anime_id)
#    ├── filter_anime(max_episodes=50, status="finished")
#    ├── personalized_rank(username, candidates)
#    └── get_metadata(anime_ids)
#    ↓
# Ranked recommendations
#    ↓
# LLM explanation
from collections.abc import Callable

from .anime_recommender import SimilarityRecommender
from .anime_repository import search_anime

class AgentTools:
    def __init__(self, 
                 recommender: SimilarityRecommender, 
                 anime_data: dict,
                 recommendation_provider: Callable):
        self.recommender = recommender
        self.anime_data = anime_data
        self.recommendation_provider = recommendation_provider

    def get_recs_tool(self, username: str, limit: int = 50) -> dict:
        """Return personalized anime IDs and scores in ranked order."""
        result = self.recommendation_provider(username, top_k=limit)
        return {
            row["anime_id"]: row["predicted_score"]
            for row in result["recommendations"]
        }


    def search_anime_tool(
        self,
        anime_names: str | list[str],
        limit: int = 10,
    ) -> list[dict]:
        """Find anime IDs whose titles match one or more names."""
        return search_anime(anime_names, limit=limit)

    def find_similar_tool(
            self,
            positive_anime_ids: list[int],
            negative_anime_ids: list[int] | None = None,
        negative_weight: float = 0.5,
        limit: int = 100,
    ) -> dict:
        """Rank anime using positive examples and optional negative examples."""
        sim_ids_scores = self.recommender.find_similar(positive_anime_ids, 
                                 negative_anime_ids, 
                                 negative_weight, 
                                 limit)

        return sim_ids_scores
        

    def filter_anime(
        self,
        sim_ids_scores: dict[int, float],
        max_episodes: int | None = None,
        status: str | None = None,
    ) -> dict[int, float]:
        """Filter ranked anime without changing their existing order."""
        anime_data = self.anime_data
        status_aliases = {
            "finished": "finished_airing",
            "airing": "currently_airing",
            "upcoming": "not_yet_aired",
        }
        if status is not None:
            status = status_aliases.get(status.lower(), status.lower())

        def passes_filters(anime_id):
            anime = anime_data.get(str(anime_id), {})
            num_episodes = anime.get("num_episodes")
            if max_episodes is not None and (
                num_episodes is None or num_episodes > max_episodes
            ):
                return False
            if status is not None and anime.get("status") != status:
                return False
            return True

        return {
            anime_id: score
            for anime_id, score in sim_ids_scores.items()
            if passes_filters(anime_id)
        }

    def get_metadata(
        self,
        recommendations: dict[int, float],
        columns_to_get: list[str] | None = None,
    ) -> list[dict]:
        """Attach selected metadata to ranked anime IDs and scores."""
        columns_to_get = columns_to_get or [
            "anime_id",
            "title",
            "picture_link",
            "score",
        ]
        anime_data = self.anime_data
        results = []

        for anime_id, score in recommendations.items():
            anime = anime_data.get(str(anime_id), {})
            row = {"anime_id": anime_id, "score": score, **anime}
            row["picture_link"] = (anime.get("main_picture") or {}).get("medium")
            results.append({column: row.get(column) for column in columns_to_get})

        return results
