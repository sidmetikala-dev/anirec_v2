from flask import Flask, jsonify, render_template
from dotenv import load_dotenv
from routes.recommendations import recommendations_py

import psycopg
import atexit
import os

load_dotenv()
database_url = os.getenv("DATABASE_URL_PROD")

app = Flask(__name__)
app.register_blueprint(recommendations_py, url_prefix="/recs")

#Connect to Postgres
if not database_url:
    raise RuntimeError("DATABASE_URL is not set. Add it to your .env file.")

conn = psycopg.connect(
    database_url,
    prepare_threshold=None,
)

app.config["DB_CONN"] = conn

def close_db_connection():
    if not conn.closed:
        conn.close()


atexit.register(close_db_connection)

# with conn.cursor() as cur:
#     cur.execute("""
#         CREATE TABLE IF NOT EXISTS users (
#             user_id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
#             username VARCHAR(255) NOT NULL UNIQUE,
#             created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
#             updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
#         );
#     """)

#     cur.execute("""
#         CREATE TABLE IF NOT EXISTS recommendation_runs (
#             run_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
#             user_id INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
#             input_hash TEXT NOT NULL,
#             top_k SMALLINT NOT NULL CONSTRAINT top_k_in_range CHECK (top_k BETWEEN 1 AND 50),
#             uncertainty_weight NUMERIC(5,1) NOT NULL,
#             created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
#             UNIQUE (user_id, input_hash, top_k, uncertainty_weight)
#         );
#     """)

#     cur.execute("""
#         CREATE UNIQUE INDEX IF NOT EXISTS recommendation_runs_cache_idx
#         ON recommendation_runs (
#             user_id,
#             input_hash,
#             top_k,
#             uncertainty_weight,
#             created_at DESC
#         );
#     """)

#     cur.execute("""
#         CREATE TABLE IF NOT EXISTS recommendation_items (
#             run_id BIGINT NOT NULL REFERENCES recommendation_runs(run_id) ON DELETE CASCADE,
#             anime_id BIGINT NOT NULL,
#             title TEXT NOT NULL,
#             rank_position SMALLINT NOT NULL CONSTRAINT rank_pos_in_range CHECK (rank_position BETWEEN 1 AND 50),
#             raw_score REAL NOT NULL,
#             uncertainty REAL NOT NULL,
#             final_score REAL NOT NULL,
#             PRIMARY KEY (run_id, anime_id),
#             UNIQUE (run_id, rank_position)
#         );
#     """)

#     cur.execute("""
#         ALTER TABLE users ENABLE ROW LEVEL SECURITY;
#         ALTER TABLE recommendation_runs ENABLE ROW LEVEL SECURITY;
#         ALTER TABLE recommendation_items ENABLE ROW LEVEL SECURITY;
#     """)

# conn.commit()

#Routes
@app.route('/')
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)
