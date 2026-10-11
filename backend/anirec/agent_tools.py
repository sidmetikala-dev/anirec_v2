# User query
#    ↓
# LLM / agent
#    ↓
# Intent + constraints
#    ↓
# Tools
#    ├── search_anime("Naruto")
#    ├── find_similar(anime_id)
#    ├── filter_anime(max_episodes=50)
#    ├── personalized_rank(username, candidates)
#    └── get_metadata(anime_ids)
#    ↓
# Ranked recommendations
#    ↓
# LLM explanation
import json
from collections.abc import Callable
from typing import Annotated, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from .anime_recommender import SimilarityRecommender
from .anime_repository import search_anime

AnimeId = Annotated[int, Field(gt=0, strict=True)]
Limit = Annotated[int, Field(ge=1, le=500, strict=True)]
Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
Score = Annotated[float, Field(allow_inf_nan=False)]
Ranking = dict[AnimeId, Score]
MetadataColumn = Literal[
    "anime_id", "title", "picture_link", "score", "num_episodes", "synopsis",
    "genres", "mean", "rank", "popularity", "num_list_users", "num_scoring_users",
    "media_type", "rating", "studios", "statistics", "main_picture", "recommendations", "related_anime",
]


class ToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GetRecsArguments(ToolArguments):
    username: Name = Field(description="MyAnimeList username to personalize recommendations for.")
    limit: Annotated[int, Field(ge=1, le=50, strict=True)] = 50


class SearchAnimeArguments(ToolArguments):
    anime_names: Name | Annotated[list[Name], Field(min_length=1, max_length=50)] = Field(
        description="Anime title or list of titles.",
    )
    limit: Limit = 10


class FindSimilarArguments(ToolArguments):
    positive_anime_ids: Annotated[list[AnimeId], Field(min_length=1, max_length=100)]
    negative_anime_ids: Annotated[list[AnimeId], Field(max_length=100)] | None = None
    negative_weight: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] = 0.5
    limit: Limit = 100


class FilterAnimeArguments(ToolArguments):
    sim_ids_scores: Ranking = Field(description="Ranked anime ID-to-score object from a recommendation tool; preserve order.")
    max_episodes: Annotated[int, Field(ge=1, strict=True)] | None = None


class GetMetadataArguments(ToolArguments):
    anime_ids: Annotated[list[AnimeId], Field(min_length=1, max_length=100)] = Field(
        description="Anime IDs to look up, returned by lookup_anime_ids or recommendation tools.")
    columns_to_get: Annotated[list[MetadataColumn], Field(min_length=1, max_length=50)] | None = Field(
        default=None,
        description=(
            "Columns to return; defaults to anime_id, title, picture_link, score. "
            "Allowed names: " + ", ".join(get_args(MetadataColumn)) + ". "
            "Use num_episodes for episode counts, media_type for format, and "
            "genres/synopsis for content. Missing fields return null."
        ),
    )


# Explicit allowlist: model-generated names cannot access arbitrary attributes.
TOOL_ARGUMENTS: dict[str, type[ToolArguments]] = {
    "get_recs_with_username_tool": GetRecsArguments,
    "lookup_anime_ids": SearchAnimeArguments,
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
    def tool_schemas(cls, format: Literal["chat_completions", "responses", "anthropic", "gemini", "groq", "ollama"] = "chat_completions") -> list[dict]:
        """Export SDK tool definitions, generated from the argument models.

        Strict structured generation is disabled because rankings contain
        dynamic ID keys. Pydantic still validates every invocation locally.
        """
        if format not in ("chat_completions", "responses", "anthropic", "gemini", "groq", "ollama"):
            raise ValueError(f"Unsupported tool schema format: {format}")
        tools = []
        for name, model in TOOL_ARGUMENTS.items():
            description = getattr(cls, name).__doc__.strip()
            schema = model.model_json_schema()
            if format in ("groq", "ollama") and name == "filter_anime":
                # The orchestrator supplies the full saved ranking locally.
                schema["properties"].pop("sim_ids_scores")
                schema["required"] = []
            if format == "anthropic":
                tools.append({"name": name, "description": description, "input_schema": schema})
            elif format == "gemini":
                tools.append({"name": name, "description": description, "parameters": schema})
            else:
                definition = {"name": name, "description": description,
                              "parameters": schema, "strict": False}
                if format in ("groq", "ollama"):
                    definition.pop("strict")
                tools.append({"type": "function", "function": definition}
                             if format in ("chat_completions", "groq", "ollama")
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
        kwargs = validated.model_dump()
        if name == "get_metadata":
            # Metadata lookup needs IDs, not model-generated scores. Preserve the direct Python API.
            kwargs["recommendations"] = {anime_id: 0.0 for anime_id in kwargs.pop("anime_ids")}
        result = getattr(self, name)(**kwargs)
        return json.dumps(result, allow_nan=False)

    def get_recs_with_username_tool(self, username: str, limit: int = 50) -> dict:
        """Get personalized recommendations from a MyAnimeList username.
        Use for an explicit username or a bare handle such as 'chekkit', not an anime title.
        Keep the default candidate limit so requested constraints can be applied afterward.
        Apply those constraints with the appropriate tools before finishing; skip filtering
        when none were requested. On lookup or ratings errors, report the error without
        substituting generic recommendations."""
        result = self.recommendation_provider(username, top_k=limit)
        return {
            row["anime_id"]: row["predicted_score"]
            for row in result["recommendations"]
        }


    def lookup_anime_ids(
        self,
        anime_names: str | list[str],
        limit: int = 10,
    ) -> list[dict]:
        """Look up anime IDs by matching titles; this does not get similar anime or recommendations.
        Each title may return several possible matches, including sequels and spin-offs.
        Select the match for each user-requested liked or excluded title, not every result.
        Pass names, not IDs, and use the returned integer IDs with find_similar_tool.
        Use get_metadata when facts are needed to interpret a requested constraint."""
        return search_anime(anime_names, limit=limit)

    def find_similar_tool(
            self,
            positive_anime_ids: list[int],
            negative_anime_ids: list[int] | None = None,
        negative_weight: float = 0.5,
        limit: int = 100,
    ) -> dict:
        """Get recommendations similar to liked anime, with optional excluded anime.
        Use integer IDs from lookup_anime_ids in this request; never guess IDs.
        Keep the default candidate limit so filtering has enough candidates.
        Before finishing, obtain any facts needed for the user's constraints and apply
        all requested constraints using the appropriate tools. Metadata lookup alone
        does not apply constraints. Skip filtering for unconstrained requests.
        Filter parameters belong in filter_anime, not in this tool's arguments."""
        sim_ids_scores = self.recommender.find_similar(positive_anime_ids, 
                                 negative_anime_ids, 
                                 negative_weight, 
                                 limit)

        return sim_ids_scores
        

    def filter_anime(
        self,
        sim_ids_scores: dict[int, float],
        max_episodes: int | None = None,
    ) -> dict[int, float]:
        """Apply supported user-requested constraints to the saved recommendation ranking.
        Retrieve recommendations first. Supply constraint parameters only; Python injects
        the saved ranking. Derive parameters from the request and fetched reference facts;
        do not guess values, add unstated constraints, or relax constraints to fill the list.
        Currently supports max_episodes, an inclusive maximum: under 20 means 19;
        at most 20 means 20. For a relative episode limit, fetch the reference episode
        count with get_metadata and subtract one for strictly fewer episodes.
        Finish only after requested constraints have been applied."""
        anime_data = self.anime_data

        def passes_filters(anime_id):
            anime = anime_data.get(str(anime_id), {})
            num_episodes = anime.get("num_episodes")
            if max_episodes is not None and (
                num_episodes is None or num_episodes <= 0 or num_episodes > max_episodes
            ):
                return False
            return True

        return {
            anime_id: score
            for anime_id, score in sorted(sim_ids_scores.items(), key=lambda item: item[1], reverse=True)
            if passes_filters(anime_id)
        }

    def get_metadata(
        self,
        recommendations: dict[int, float],
        columns_to_get: list[str] | None = None,
    ) -> list[dict]:
        """Get facts about anime IDs to interpret or verify the user's requested constraints.
        Explicitly request the needed fields with columns_to_get. Supply anime_ids as
        a list of integer IDs returned by lookup or recommendation tools, e.g. [20].
        Fetch reference facts before deciding constraint parameters. Missing fields are
        null; never guess them. This tool returns facts, not filtered recommendations.
        Continue with the appropriate constraint tools afterward. Python fetches final
        explanation metadata automatically, so this is not needed just to write explanations."""
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
