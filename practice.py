import os
import psycopg
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv(".env")


conn = psycopg.connect(
    os.environ["DATABASE_URL_DEV"],
    prepare_threshold=None,
)
cur = conn.cursor()

cur.execute("""
            DROP TABLE users
""")

conn.commit()

cur.close()
conn.close()


