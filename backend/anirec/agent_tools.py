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
import json
from collections.abc import Callable
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from .anime_recommender import SimilarityRecommender
from .anime_repository import search_anime

AnimeId = Annotated[int, Field(gt=0, strict=True)]
Limit = Annotated[int, Field(ge=1, le=500, strict=True)]
Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
Score = Annotated[float, Field(allow_inf_nan=False)]
Ranking = dict[AnimeId, Score]


class ToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GetRecsArguments(ToolArguments):
    username: Name = Field(description="MyAnimeList username to personalize recommendations for.")
    limit: Limit = 50


class SearchAnimeArguments(ToolArguments):
    anime_names: Name | Annotated[list[Name], Field(min_length=1, max_length=50)]
    limit: Limit = 10


class FindSimilarArguments(ToolArguments):
    positive_anime_ids: Annotated[list[AnimeId], Field(min_length=1, max_length=100)]
    negative_anime_ids: Annotated[list[AnimeId], Field(max_length=100)] | None = None
    negative_weight: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] = 0.5
    limit: Limit = 100


class FilterAnimeArguments(ToolArguments):
    sim_ids_scores: Ranking = Field(description="Ranked anime ID-to-score object from a recommendation tool; preserve order.")
    max_episodes: Annotated[int, Field(ge=1, strict=True)] | None = None
    status: Literal["finished", "airing", "upcoming", "finished_airing", "currently_airing", "not_yet_aired"] | None = None


class GetMetadataArguments(ToolArguments):
    recommendations: Ranking = Field(description="Ranked anime ID-to-score object from a recommendation or filtering tool.")
    columns_to_get: Annotated[list[Name], Field(min_length=1, max_length=50)] | None = None


# Explicit allowlist: model-generated names cannot access arbitrary attributes.
TOOL_ARGUMENTS: dict[str, type[ToolArguments]] = {
    "get_recs_tool": GetRecsArguments,
    "search_anime_tool": SearchAnimeArguments,
    "find_similar_tool": FindSimilarArguments,
    "filter_anime": FilterAnimeArguments,
    "get_metadata": GetMetadataArguments,
}


class AgentTools:
    """Backend tools with Pydantic contracts for LLM function calling.

    Use ``tool_schemas()`` in the SDK request, then pass the returned function
    name and JSON arguments to ``execute_tool()``. Attach its JSON string to
    the SDK's tool-result message using the original tool-call ID.
    Direct Python methods remain available without the SDK boundary.
    """

    def __init__(self, recommender: SimilarityRecommender, anime_data: dict,
                 recommendation_provider: Callable):
        self.recommender = recommender
        self.anime_data = anime_data
        self.recommendation_provider = recommendation_provider

    @classmethod
    def tool_schemas(cls, format: Literal["chat_completions", "responses", "anthropic"] = "chat_completions") -> list[dict]:
        """Export SDK tool definitions, generated from the argument models.

        Strict structured generation is disabled because rankings contain
        dynamic ID keys. Pydantic still validates every invocation locally.
        """
        if format not in ("chat_completions", "responses", "anthropic"):
            raise ValueError(f"Unsupported tool schema format: {format}")
        tools = []
        for name, model in TOOL_ARGUMENTS.items():
            description = getattr(cls, name).__doc__.strip()
            schema = model.model_json_schema()
            if format == "anthropic":
                tools.append({"name": name, "description": description, "input_schema": schema})
            else:
                definition = {"name": name, "description": description,
                              "parameters": schema, "strict": False}
                tools.append({"type": "function", "function": definition}
                             if format == "chat_completions"
                             else {"type": "function", **definition})
        return tools

    def execute_tool(self, name: str, arguments: str | dict) -> str:
        """Validate and execute an allowed tool; return JSON for the SDK.

        Unknown tools and malformed arguments return recoverable errors.
        Backend failures propagate to the orchestration layer for logging,
        retry policy, and sanitized user-facing error handling.
        """
        model = TOOL_ARGUMENTS.get(name)
        if model is None:
            return json.dumps({"error": {"code": "unknown_tool", "message": "Unknown tool name"}})
        try:
            # JSON validation handles string object keys as integer anime IDs.
            payload = arguments if isinstance(arguments, str) else json.dumps(arguments, allow_nan=False)
            validated = model.model_validate_json(payload)
        except (ValidationError, TypeError, ValueError) as exc:
            details = (exc.errors(include_url=False, include_context=False, include_input=False)
                       if isinstance(exc, ValidationError) else [])
            return json.dumps({"error": {"code": "invalid_arguments",
                                         "message": "Tool arguments failed validation", "details": details}})
        result = getattr(self, name)(**validated.model_dump())
        return json.dumps(result, allow_nan=False)

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
            row = {**anime, "anime_id": anime_id, "score": score}
            row["picture_link"] = (anime.get("main_picture") or {}).get("medium")
            results.append({column: row.get(column) for column in columns_to_get})

        return results
