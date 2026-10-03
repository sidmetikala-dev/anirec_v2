from flask import Blueprint, current_app, jsonify, request

from anirec.anime_recommender import SimilarityRecommender
from anirec.anime_features import AnimeFeatureBuilder
from anirec.anime_data import AnimeDataClient
from anirec.recommendation_service import get_recs_with_username, NoRatedAnimeError
from dotenv import load_dotenv
from requests.exceptions import RequestException

import psycopg
import os

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

def recommendation_dependencies(pool):
    """Provide shared runtime objects to the service and agent tools."""
    return {
        "pool": pool,
        "client_id": client_id,
        "anime_data_client": anime_data_client,
        "anime_data": anime_data,
        "builder": builder,
        "recommender": recommender,
        "anime_df": anime_df,
    }


@recommendations_py.route("", methods=["GET", "POST"])
@recommendations_py.route("/", methods=["GET", "POST"])
def recommend():
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be an object"}), 400
    username = data.get("username") or request.args.get("username") or ""
    raw_top_k = data.get("top_k")
    if raw_top_k is None:
        raw_top_k = request.args.get("top_k", 5)
    try:
        if isinstance(raw_top_k, bool) or isinstance(raw_top_k, float):
            raise ValueError
        top_k = int(raw_top_k)
    except (ValueError, TypeError):
        return jsonify({"error": "top_k must be an integer"}), 400

    try:
        result = get_recs_with_username(
            username, top_k,
            **recommendation_dependencies(current_app.config["DB_POOL"]),
        )
    except NoRatedAnimeError as error:
        return jsonify({"error": str(error)}), 404
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except RequestException as error:
        return jsonify({
            "error": "Could not connect to the MyAnimeList API",
            "details": str(error),
        }), 502
    except RuntimeError as error:
        return jsonify({"error": str(error)}), 502
    except psycopg.Error as error:
        return jsonify({
            "error": "Could not read or save recommendations",
            "details": str(error),
        }), 500

    # Keep the existing HTTP response fields; scores are available to tools.
    result["recommendations"] = [
        {key: row[key] for key in ("anime_id", "title", "picture_link")}
        for row in result["recommendations"]
    ]
    return jsonify(result)
