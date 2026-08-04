from flask import Flask, jsonify, render_template
from dotenv import load_dotenv
from routes.recommendations import recommendations_py

from psycopg_pool import ConnectionPool
import os

load_dotenv()
database_url = os.getenv("DATABASE_URL_PROD")

app = Flask(__name__)
app.register_blueprint(recommendations_py, url_prefix="/recs")

#Connect to Postgres
if not database_url:
    raise RuntimeError("DATABASE_URL is not set. Add it to your .env file.")

pool = ConnectionPool(
    conninfo=database_url,
    min_size=1,       # Keep 1 connection open for instant response on warm starts
    max_size=4,       # Allow scaling up to 4 concurrent paths per container instance
    max_idle=5.0,     # If a connection sits idle for 5s, close it (prevents frozen socket errors)
    timeout=5.0,      # If all connections are busy, wait a max of 5s before throwing an error
    kwargs={
        "autocommit": True,         # Required for transaction pooling
        "prepare_threshold": None,  # Required: Disables prepared statements
        "connect_timeout": 5,       # Network timeout for establishing new sockets
        "sslmode" : "require"     
    }   
)

app.config["DB_POOL"] = pool

with pool.connection() as conn:
    with conn.cursor() as cur:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            username VARCHAR(255) NOT NULL UNIQUE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """)
        
        cur.execute("""
        CREATE TABLE IF NOT EXISTS recommendation_runs (
            run_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            input_hash TEXT NOT NULL,
            model_type TEXT NOT NULL,
            model_name TEXT NOT NULL,
            top_k SMALLINT NOT NULL CONSTRAINT top_k_in_range CHECK (top_k BETWEEN 1 AND 50),
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (user_id, input_hash, model_type, model_name, top_k)
        );
        """)
        
        cur.execute("""
        CREATE INDEX IF NOT EXISTS recommendation_runs_cache_idx ON recommendation_runs (
            user_id, input_hash, model_type, model_name, top_k, created_at DESC
        );
        """)
        
        cur.execute("""
        CREATE TABLE IF NOT EXISTS recommendation_items (
            run_id BIGINT NOT NULL REFERENCES recommendation_runs(run_id) ON DELETE CASCADE,
            anime_id BIGINT NOT NULL,
            title TEXT NOT NULL,
            rank_position SMALLINT NOT NULL CONSTRAINT rank_pos_in_range CHECK (rank_position BETWEEN 1 AND 50),
            predicted_score REAL NOT NULL,
            PRIMARY KEY (run_id, anime_id),
            UNIQUE (run_id, rank_position)
        );
        """)
        
        cur.execute("""
        ALTER TABLE users ENABLE ROW LEVEL SECURITY;
        ALTER TABLE recommendation_runs ENABLE ROW LEVEL SECURITY;
        ALTER TABLE recommendation_items ENABLE ROW LEVEL SECURITY;
        """)
        

#Routes
@app.route('/')
def home():
    return render_template("index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, 
            debug=False, use_reloader=False)
