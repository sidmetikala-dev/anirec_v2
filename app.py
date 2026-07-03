from flask import Flask, jsonify, request
from anime_recommender import BayesianRidgeRecommender
from anime_recommender import SimilarityRecommender
from anime_features import AnimeFeatureBuilder
from anime_data import AnimeDataClient
from mal_client import MALClient
from dotenv import load_dotenv
from requests.exceptions import RequestException

import os

load_dotenv()
client_id = os.getenv("CLIENT_ID")

app = Flask(__name__)

if not client_id:
    raise RuntimeError("CLIENT_ID is not set. Add it to your .env file.")

anime_data_client = AnimeDataClient(client_id)
anime_data = anime_data_client.get_cache()
builder = AnimeFeatureBuilder(
    anime_data,
    max_tfidf_features=3000,
    n_svd_components=300
)
anime_df = builder.build_features()
recommender = SimilarityRecommender()
anime_vectors = recommender.create_anime_vectors(anime_df)
anime_df_scaled = recommender.anime_df_scaled

@app.route('/')
def home():
    return jsonify({
        "status": "ok",
        "message": "AniRec API is running",
        "endpoints": {
            "health": "/health",
            "recommendations": "/rec?username=<mal_username>&top_k=5",
        },
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "ok",
        "cached_anime": len(anime_data),
        "feature_rows": int(anime_df.shape[0]),
        "feature_columns": int(anime_df.shape[1]),
    })

@app.route('/rec', methods=['GET', 'POST'])
def recommend():
    data = request.get_json(silent=True) or {}

    username = (
        data.get("username")
        or request.args.get("username")
        or ""
    ).strip()
    if not username:
        return jsonify({"error": "Username is required"}), 400

    try:
        top_k = int(data.get("top_k") or request.args.get("top_k") or 5)
    except ValueError:
        return jsonify({"error": "top_k must be an integer"}), 400

    if top_k < 1 or top_k > 50:
        return jsonify({"error": "top_k must be between 1 and 50"}), 400

    user_client = MALClient(client_id)

    try:
        user_data = user_client.get_user_data(username)
        user_scores = user_client.get_scores(user_data)

        if not user_scores:
            return jsonify({
                "error": "No completed, scored TV anime found for this user"
            }), 404

        bayesian_rec = BayesianRidgeRecommender(
            anime_data_client,
            user_scores=user_scores,
            anime_data=anime_data,
            builder=builder,
            recommender=recommender,
            anime_df=anime_df,
            anime_df_scaled=anime_df_scaled,
            anime_vectors=anime_vectors,
        )
        bayesian_rec.fit()
        recs = list(bayesian_rec.get_recs().head(top_k))
    except RuntimeError as error:
        return jsonify({"error": str(error)}), 502
    except RequestException as error:
        return jsonify({
            "error": "Could not connect to the MyAnimeList API",
            "details": str(error),
        }), 502
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    return jsonify({
        "username": username,
        "top_k": top_k,
        "recommendations": recs,
    })


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
