from flask import Flask, jsonify, request

from anime_recommender import BayesianRidgeRecommender
from anime_recommender import SimilarityRecommender
from anime_features import AnimeFeatureBuilder
from anime_data import AnimeDataClient
from mal_client import MALClient
from dotenv import load_dotenv
from requests.exceptions import RequestException

import psycopg
import atexit
import os

load_dotenv()
client_id = os.getenv("CLIENT_ID")
database_url = os.getenv("DATABASE_URL_DEV")

app = Flask(__name__)

#Connect to Postgres
if not database_url:
    raise RuntimeError("DATABASE_URL_DEV is not set. Add it to your .env file.")

conn = psycopg.connect(
    database_url,
    prepare_threshold=None,
)


def close_db_connection():
    if not conn.closed:
        conn.close()


atexit.register(close_db_connection)

with conn.cursor() as cur:
    cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                username VARCHAR(255) NOT NULL UNIQUE,
                recommendations TEXT[],
                top_k INTEGER,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            );
""")
    cur.execute("""
            ALTER TABLE users
            ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW();
""")
    cur.execute("""
            ALTER TABLE users
            ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();
""")
    cur.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS users_username_idx
            ON users (username);
""")
    cur.execute("""
            ALTER TABLE users ENABLE ROW LEVEL SECURITY
""")

conn.commit()

#Initialize all important variables
if not client_id:
    raise RuntimeError("CLIENT_ID is not set. Add it to your .env file.")

anime_data_client = AnimeDataClient(client_id)
anime_data = anime_data_client.get_cache()
builder = AnimeFeatureBuilder(
    anime_data,
    max_tfidf_features=4000,
    n_svd_components=400
)
anime_df = builder.build_features()
recommender = SimilarityRecommender()
anime_vectors = recommender.create_anime_vectors(anime_df)
anime_df_scaled = recommender.anime_df_scaled

#Routes
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
            uncertainty_weight=20,
            user_scores=user_scores,
            anime_data=anime_data,
            builder=builder,
            recommender=recommender,
            anime_df=anime_df,
            anime_df_scaled=anime_df_scaled,
            anime_vectors=anime_vectors,
        )
        bayesian_rec.fit()
        recommendations = bayesian_rec.get_recs(top_k=top_k)
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
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO users
                (username, recommendations, top_k)
                VALUES
                (%s, %s, %s)
                ON CONFLICT (username)
                DO UPDATE SET
                    recommendations = EXCLUDED.recommendations,
                    top_k = EXCLUDED.top_k,
                    updated_at = NOW()
            """, (username, recs, top_k))
        conn.commit()
    except psycopg.Error as error:
        conn.rollback()
        return jsonify({
            "error": "Recommendations were generated but could not be saved",
            "details": str(error),
        }), 500

    return jsonify({
        "username": username,
        "top_k": top_k,
        "recommendations": recs,
    })


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
