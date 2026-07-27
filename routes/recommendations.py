from flask import Blueprint, current_app, jsonify, request

from anime_recommender import KNNRegressor
from anime_recommender import SimilarityRecommender
from anime_features import AnimeFeatureBuilder
from anime_data import AnimeDataClient
from mal_client import MALClient
from dotenv import load_dotenv
from requests.exceptions import RequestException

import hashlib
import json
import psycopg
import os
import time

load_dotenv()

recommendations_py = Blueprint("recommendations", __name__)

client_id = os.getenv("CLIENT_ID")

#Initialize all important objects
if not client_id:
    raise RuntimeError("CLIENT_ID is not set. Add it to your .env file.")

anime_data_client = AnimeDataClient(client_id)
anime_data = anime_data_client.get_cache()
builder = AnimeFeatureBuilder(
    anime_data,
    max_tfidf_features=4000,
    n_svd_components=400
)
anime_df = builder.build_features(fit_svd=False, fit_tfidf=False)
recommender = SimilarityRecommender()
anime_vectors = recommender.create_anime_vectors(anime_df)
anime_df_scaled = recommender.anime_df_scaled

@recommendations_py.route('/health')
def health():
    return jsonify({
        "status": "ok",
        "cached_anime": len(anime_data),
        "feature_rows": int(anime_df.shape[0]),
        "feature_columns": int(anime_df.shape[1]),
    })

@recommendations_py.route("", methods=["GET", "POST"])
@recommendations_py.route("/", methods=["GET", "POST"])
def recommend():
    data = request.get_json(silent=True) or {}
    
    #Get username
    username = (
        data.get("username")
        or request.args.get("username")
        or ""
    ).strip()
    if not username:
        return jsonify({"error": "Username is required"}), 400

    #Get top_k
    raw_top_k = data.get("top_k")
    if raw_top_k is None:
        raw_top_k = request.args.get("top_k", 5)

    try:
        top_k = int(raw_top_k)
    except ValueError:
        return jsonify({"error": "top_k must be an integer"}), 400

    if top_k < 1 or top_k > 50:
        return jsonify({"error": "top_k must be between 1 and 50"}), 400

    #Get recs
    pool = current_app.config["DB_POOL"]
    model_type = "knn"
    model_name = "knn_all_available"
    max_n_neighbors = None
    user_client = MALClient(client_id)

    try:
        user_data = user_client.get_user_data(username)
        user_scores = user_client.get_scores(user_data)

        if not user_scores:
            return jsonify({
                "error": "No completed, scored TV anime found for this user"
            }), 404

        #Generate input_hash
        input_payload = {
            "model_type": model_type,
            "model_name": model_name,
            "max_n_neighbors": max_n_neighbors,
            "user_scores": user_scores,
        }
        input_string = json.dumps(input_payload, sort_keys=True)
        input_hash = hashlib.md5(input_string.encode('utf-8')).hexdigest()

        # Check if we already have an identical saved run
        current_app.logger.info(
            "Cache pipeline started for username=%s", username
        )
        cache_pipeline_start = time.perf_counter()
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
                        ri.rank_position
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

        if existing_rows:
            recs = [row[1] for row in existing_rows]
            cache_pipeline_ms = (
                time.perf_counter() - cache_pipeline_start
            ) * 1_000
            current_app.logger.info(
                "Cache hit for username=%s completed in %.3f ms",
                username,
                cache_pipeline_ms,
            )
            return jsonify({
                "username": username,
                "top_k": top_k,
                "model_type": model_type,
                "model": model_name,
                "recommendations": recs,
                "cached": True,
                "cache_pipeline_ms": round(cache_pipeline_ms, 3),
            })

        #Generate fresh recs
        knn_recs = KNNRegressor(
            anime_data_client,
            user_scores=user_scores,
            anime_data=anime_data,
            builder=builder,
            recommender=recommender,
            anime_df=anime_df,
            anime_df_scaled=anime_df_scaled,
            anime_vectors=anime_vectors,
            retrieve_missing_anime=False,
            max_n_neighbors=max_n_neighbors,
        )
        knn_recs.fit()
        recommendations = knn_recs.get_recs(top_k=top_k)
        recs = recommendations["title"].tolist()
    except RuntimeError as error:
        return jsonify({"error": str(error)}), 502
    except RequestException as error:
        return jsonify({
            "error": "Could not connect to the MyAnimeList API",
            "details": str(error),
        }), 502
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    #Store recommendation request
    try:
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
                        rank_position,
                        predicted_score
                    )
                    SELECT
                        add_run.run_id,
                        x.anime_id,
                        x.title,
                        x.rank_position,
                        x.predicted_score
                    FROM add_run,
                    UNNEST(
                        %s::bigint[],
                        %s::text[],
                        %s::smallint[],
                        %s::real[]
                    ) AS x(anime_id, title, rank_position, predicted_score)
                    ON CONFLICT DO NOTHING
                """, (
                    username,
                    input_hash,
                    model_type,
                    model_name,
                    top_k,
                    recommendations["anime_id"].tolist(),
                    recommendations["title"].tolist(),
                    rank_positions,
                    recommendations["predicted_score"].tolist(),
                ))
            
    except psycopg.Error as error:
        return jsonify({
            "error": "Recommendations were generated but could not be saved",
            "details": str(error),
        }), 500

    cache_pipeline_ms = (
        time.perf_counter() - cache_pipeline_start
    ) * 1_000
    current_app.logger.info(
        "Cache miss for username=%s completed in %.3f ms",
        username,
        cache_pipeline_ms,
    )
    return jsonify({
        "username": username,
        "top_k": top_k,
        "model_type": model_type,
        "model": model_name,
        "recommendations": recs,
        "cached": False,
        "cache_pipeline_ms": round(cache_pipeline_ms, 3),
    })
