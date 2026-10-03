"""Shared username recommendation workflow for routes and agent tools."""

import hashlib
import json
import logging
import time

from .anime_recommender import KNNRegressor
from .mal_client import MALClient

logger = logging.getLogger(__name__)

MODEL_TYPE = "knn"
MODEL_NAME = "knn_all_available"
MAX_N_NEIGHBORS = None


class NoRatedAnimeError(ValueError):
    """The user has no eligible ratings."""


def validate_recommendation_input(username, top_k):
    """Validate request values and return the trimmed username."""
    if not isinstance(username, str) or not username.strip():
        raise ValueError("Username is required")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 50:
        raise ValueError("top_k must be between 1 and 50")
    return username.strip()


def fetch_user_scores(username, client_id):
    """Fetch eligible MAL ratings, rejecting profiles without any."""
    user_client = MALClient(client_id)
    user_data = user_client.get_user_data(username)
    user_scores = user_client.get_scores(user_data)
    if not user_scores:
        raise NoRatedAnimeError("No completed, scored TV anime found for this user")
    return user_scores


def build_recommendation_hash(user_scores, model_type, model_name, max_n_neighbors):
    """Keep the existing cache key format for ratings and model settings."""
    input_payload = {
        "model_type": model_type,
        "model_name": model_name,
        "max_n_neighbors": max_n_neighbors,
        "user_scores": user_scores,
    }
    input_string = json.dumps(input_payload, sort_keys=True)
    return hashlib.md5(input_string.encode("utf-8")).hexdigest()


def load_cached_recommendations(
    pool, username, input_hash, model_type, model_name, top_k, anime_data,
):
    """Read ranked records from the cache, falling back to metadata for pictures."""
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                WITH latest_run AS (
                    SELECT rr.run_id
                    FROM users u
                    JOIN recommendation_runs rr
                        ON u.user_id = rr.user_id
                    WHERE u.username = %s
                    AND rr.input_hash = %s
                    AND rr.model_type = %s
                    AND rr.model_name = %s
                    AND rr.top_k = %s
                    ORDER BY rr.created_at DESC
                    LIMIT 1
                )
                SELECT
                    latest_run.run_id,
                    ri.title,
                    ri.picture_link,
                    ri.anime_id,
                    ri.rank_position,
                    ri.predicted_score
                FROM latest_run
                JOIN recommendation_items ri
                    ON latest_run.run_id = ri.run_id
                ORDER BY ri.rank_position ASC
            """, (
                username,
                input_hash,
                model_type,
                model_name,
                top_k,
            ))
            existing_rows = cur.fetchall()

    return [
        {
            "anime_id": row[3],
            "predicted_score": row[5],
            "title": row[1],
            "picture_link": row[2] or (
                anime_data.get(str(row[3]), {}).get("main_picture")
                or {}
            ).get("medium"),
        }
        for row in existing_rows
    ]


def generate_recommendations(
    user_scores, top_k, *, anime_data_client, anime_data,
    builder, recommender, anime_df, max_n_neighbors,
):
    """Fit a request-specific KNN model and convert its ranking to records."""
    knn_recs = KNNRegressor(
        anime_data_client,
        user_scores=user_scores,
        anime_data=anime_data,
        builder=builder,
        recommender=recommender,
        anime_df=anime_df,
        anime_df_scaled=recommender.anime_df_scaled,
        anime_vectors=recommender.anime_vectors,
        retrieve_missing_anime=False,
        max_n_neighbors=max_n_neighbors,
    )
    knn_recs.fit()
    recommendations = knn_recs.get_recs(top_k=top_k)
    return recommendations[
        ["anime_id", "title", "picture_link", "predicted_score"]
    ].to_dict(
        orient="records"
    )



def save_recommendations(
    pool, username, input_hash, model_type, model_name, top_k, recommendations,
):
    """Persist ranked records and their run using the shared connection pool."""
    rank_positions = list(range(1, len(recommendations) + 1))
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                WITH add_user AS (
                    INSERT INTO users (username)
                    VALUES (%s)
                    ON CONFLICT (username)
                    DO UPDATE SET updated_at = NOW()
                    RETURNING user_id
                ),
                add_run AS (
                    INSERT INTO recommendation_runs (
                        user_id,
                        input_hash,
                        model_type,
                        model_name,
                        top_k
                    )
                    SELECT user_id, %s, %s, %s, %s
                    FROM add_user
                    ON CONFLICT (
                        user_id,
                        input_hash,
                        model_type,
                        model_name,
                        top_k
                    )
                    DO UPDATE SET created_at = recommendation_runs.created_at
                    RETURNING run_id
                )
                INSERT INTO recommendation_items (
                    run_id,
                    anime_id,
                    title,
                    picture_link,
                    rank_position,
                    predicted_score
                )
                SELECT
                    add_run.run_id,
                    x.anime_id,
                    x.title,
                    x.picture_link,
                    x.rank_position,
                    x.predicted_score
                FROM add_run,
                UNNEST(
                    %s::bigint[],
                    %s::text[],
                    %s::text[],
                    %s::smallint[],
                    %s::real[]
                ) AS x(
                    anime_id,
                    title,
                    picture_link,
                    rank_position,
                    predicted_score
                )
                ON CONFLICT DO NOTHING
            """, (
                username,
                input_hash,
                model_type,
                model_name,
                top_k,
                [row["anime_id"] for row in recommendations],
                [row["title"] for row in recommendations],
                [row["picture_link"] for row in recommendations],
                rank_positions,
                [row["predicted_score"] for row in recommendations],
            ))


def start_cache_pipeline_timer(username):
    """Start telemetry after MAL retrieval and cache-key construction."""
    logger.info("Cache pipeline started for username=%s", username)
    return time.perf_counter()


def finish_cache_pipeline_timer(username, started_at, cached):
    """Log cache outcome and return the elapsed milliseconds."""
    elapsed_ms = (time.perf_counter() - started_at) * 1_000
    logger.info(
        "Cache %s for username=%s completed in %.3f ms",
        "hit" if cached else "miss", username, elapsed_ms,
    )
    return round(elapsed_ms, 3)


def get_recs_with_username(
    username, top_k=5, *, pool, client_id, anime_data_client,
    anime_data, builder, recommender, anime_df,
):
    """Coordinate validation, retrieval, caching, generation, and telemetry."""
    username = validate_recommendation_input(username, top_k)
    user_scores = fetch_user_scores(username, client_id)
    input_hash = build_recommendation_hash(
        user_scores, MODEL_TYPE, MODEL_NAME, MAX_N_NEIGHBORS,
    )
    started_at = start_cache_pipeline_timer(username)
    recommendations = load_cached_recommendations(
        pool, username, input_hash, MODEL_TYPE, MODEL_NAME, top_k, anime_data,
    )
    cached = bool(recommendations)
    if not cached:
        recommendations = generate_recommendations(
            user_scores, top_k, anime_data_client=anime_data_client,
            anime_data=anime_data, builder=builder, recommender=recommender,
            anime_df=anime_df, max_n_neighbors=MAX_N_NEIGHBORS,
        )
        save_recommendations(
            pool, username, input_hash, MODEL_TYPE, MODEL_NAME, top_k, recommendations,
        )
    elapsed_ms = finish_cache_pipeline_timer(username, started_at, cached)
    return {
        "username": username,
        "top_k": top_k,
        "model_type": MODEL_TYPE,
        "model": MODEL_NAME,
        "recommendations": recommendations,
        "cached": cached,
        "cache_pipeline_ms": elapsed_ms,
    }
