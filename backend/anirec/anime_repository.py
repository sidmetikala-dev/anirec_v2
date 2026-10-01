import psycopg
import os


def search_anime(
    anime_names: str | list[str],
    limit: int = 10,
) -> list[dict]:
    database_url = os.getenv("DATABASE_URL_PROD")
    if not database_url:
        raise RuntimeError("DATABASE_URL_PROD is not set.")

    if isinstance(anime_names, str):
        anime_names = [anime_names]
    anime_names = [name.strip() for name in anime_names if name.strip()]
    if not anime_names:
        return []

    limit = max(1, min(limit, 50))
    formatted_names = [f"%{name}%" for name in anime_names]
    exact_names = [name.lower() for name in anime_names]

    with psycopg.connect(database_url, sslmode="require") as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT anime_id, anime_name
                FROM anime_cache
                WHERE anime_name ILIKE ANY(%s)
                ORDER BY
                    LOWER(anime_name) = ANY(%s) DESC,
                    LENGTH(anime_name),
                    anime_name
                LIMIT %s
            """, (formatted_names, exact_names, limit))

            return [
                {"anime_id": anime_id, "anime_name": anime_name}
                for anime_id, anime_name in cur.fetchall()
            ]
