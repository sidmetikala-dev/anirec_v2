import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from requests.exceptions import RequestException


class AnimeDataClient:
    CACHE_FIELDS = (
        "id,title,synopsis,mean,popularity,genres,statistics,main_picture,related_anime"
    )
    FULL_FIELDS = (
        "id,title,synopsis,mean,rank,popularity,num_list_users,"
        "num_scoring_users,media_type,status,genres,num_episodes,rating,"
        "recommendations,studios,statistics,main_picture,related_anime"
    )

    def __init__(self, client_id, cache_file="data/anime_cache.json", caching_mode=True):
        self.cache_path = Path(cache_file)
        self.headers = {
            "X-MAL-CLIENT-ID": client_id
        }
        self.caching_mode = caching_mode
        self.params = {
            "fields": self.CACHE_FIELDS if caching_mode else self.FULL_FIELDS
        }

    def _load_cache(self):
        if not self.cache_path.exists():
            return {}

        with self.cache_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _save_cache(self, anime_cache):
        with self.cache_path.open("w", encoding="utf-8") as file:
            json.dump(anime_cache, file, ensure_ascii=False, indent=2)

    def _get_page(self, url, params=None):
        response = requests.get(url, headers=self.headers, params=params, timeout=15)
        if response.status_code != 200:
            if response.status_code >= 500:
                raise RuntimeError(
                    "MyAnimeList is temporarily unavailable. Please try again."
                )
            raise RuntimeError(
                f"MyAnimeList request failed with status {response.status_code}."
            )
        return response.json()

    def _get_uncached_ranked_anime_ids(self, ranking_type, total=1000, page_limit=500):
        anime_cache = self._load_cache()
        if page_limit > 500:
            raise ValueError("MAL ranking requests support a maximum limit of 500.")

        anime_ids = []
        url = "https://api.myanimelist.net/v2/anime/ranking"
        params = {
            "fields": "id",
            "ranking_type": ranking_type,
            "limit": page_limit,
        }
        next_url = url
        seen_urls = set()
        seen_ids = set()

        while next_url and len(anime_ids) < total:
            if next_url in seen_urls:
                raise RuntimeError("Repeated paging URL detected while fetching ranked anime.")
            seen_urls.add(next_url)

            page = self._get_page(next_url, params=params)

            for item in page.get("data", []):
                node = item.get("node", {})
                anime_id = node.get("id")
                if not anime_id or anime_id in seen_ids or str(anime_id) in anime_cache:
                    continue
                
                seen_ids.add(anime_id)
                anime_ids.append(anime_id)

                if len(anime_ids) >= total:
                    break

            next_url = page.get("paging", {}).get("next")
            params = None

            print(
                f"{ranking_type}: page fetched, "
                f"collected {len(anime_ids)} uncached IDs so far"
            )
        return anime_ids

    def cache_ranked_animes(self, ranking_type, total=1000, page_limit=500, workers=3):
        anime_ids = self._get_uncached_ranked_anime_ids(
            ranking_type=ranking_type,
            total=total,
            page_limit=page_limit,
        )
        return self.get_anime_data(anime_ids, max_workers=workers)

    def _fetch_anime_detail(self, anime_id):
        url = f"https://api.myanimelist.net/v2/anime/{anime_id}"
        try:
            response = requests.get(
                url,
                headers=self.headers,
                params=self.params,
                timeout=15
            )
        except RequestException as error:
            return anime_id, None, str(error)

        if response.status_code != 200:
            if response.status_code == 404:
                return anime_id, None, "Anime not found on MyAnimeList."
            if response.status_code >= 500:
                return (
                    anime_id,
                    None,
                    "MyAnimeList is temporarily unavailable. Please try again.",
                )
            return (
                anime_id,
                None,
                f"MyAnimeList request failed with status {response.status_code}.",
            )

        return anime_id, response.json(), None

    def get_anime_data(self, anime_ids, max_workers=5):
        self.last_rate_limited = False
        anime_cache = self._load_cache()

        anime_data = {}
        cache_changed = False
        new_fetches = 0
        save_every = 50
        uncached_ids = []

        print(f"Fetching details for {len(anime_ids)} anime...")

        for anime_id in anime_ids:
            cache_key = str(anime_id)

            if cache_key in anime_cache:
                anime_data[anime_id] = anime_cache[cache_key]
            else:
                uncached_ids.append(anime_id)

        if uncached_ids:
            worker_count = max(1, min(max_workers, len(uncached_ids)))
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(self._fetch_anime_detail, anime_id): anime_id
                    for anime_id in uncached_ids
                }

                for i, future in enumerate(as_completed(futures), start=1):
                    anime_id, data, error = future.result()
                    if data is None:
                        if str(error).startswith("429"):
                            self.last_rate_limited = True
                            print(
                                "MAL rate limit hit while fetching anime "
                                f"ID {anime_id}: {error}"
                            )
                            for pending_future in futures:
                                pending_future.cancel()
                            if cache_changed:
                                print("Saving cache before returning...")
                                self._save_cache(anime_cache)
                            return anime_data

                        print(f"Skipping anime ID {anime_id}: {error}")
                        continue

                    cache_key = str(anime_id)
                    anime_data[anime_id] = data
                    anime_cache[cache_key] = data
                    cache_changed = True
                    new_fetches += 1

                    if i % 25 == 0 or i == len(uncached_ids):
                        print(
                            f"Fetched {i}/{len(uncached_ids)} uncached "
                            "anime details"
                        )

                    if new_fetches > 0 and new_fetches % save_every == 0:
                        print(f"Saving cache after {new_fetches} new anime...")
                        self._save_cache(anime_cache)
        else:
            print("All requested anime were already cached.")

        if cache_changed:
            print("Saving final cache...")
            self._save_cache(anime_cache)

        return anime_data

    def refresh(self, max_workers=3, save_every=50):
        self.last_rate_limited = False
        anime_cache = self._load_cache()
        anime_ids = [
            int(anime_id)
            for anime_id, anime in anime_cache.items()
            if "related_anime" not in anime
        ]

        if not anime_ids:
            print("All cached anime already include related_anime.")
            return anime_cache

        print(f"Refreshing {len(anime_ids)} cached anime missing related_anime...")

        refreshed = 0
        skipped = 0
        consecutive_skipped = 0
        worker_count = max(1, min(max_workers, len(anime_ids)))

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(self._fetch_anime_detail, anime_id): anime_id
                for anime_id in anime_ids
            }

            for i, future in enumerate(as_completed(futures), start=1):
                anime_id, data, error = future.result()
                cache_key = str(anime_id)

                if data is None:
                    skipped += 1
                    consecutive_skipped += 1
                    if str(error).startswith("429"):
                        self.last_rate_limited = True
                        print(
                            "MAL rate limit hit while refreshing anime "
                            f"ID {anime_id}: {error}"
                        )
                        for pending_future in futures:
                            pending_future.cancel()
                        print("Saving refreshed cache before returning...")
                        self._save_cache(anime_cache)
                        return anime_cache

                    continue

                anime_cache[cache_key] = data
                refreshed += 1
                consecutive_skipped = 0

                if i % 25 == 0 or i == len(anime_ids):
                    print(
                        f"Processed {i}/{len(anime_ids)} cached anime "
                        f"({refreshed} refreshed, {skipped} skipped, "
                        f"{consecutive_skipped} consecutive skipped)"
                    )

                if refreshed > 0 and refreshed % save_every == 0:
                    print(f"Saving cache after {refreshed} refreshed anime...")
                    self._save_cache(anime_cache)

        print(
            "Saving final refreshed cache... "
            f"({refreshed} refreshed, {skipped} skipped)"
        )
        self._save_cache(anime_cache)
        return anime_cache

    def get_rated_items(
        self,
        user_scores,
        anime_data,
        builder,
        recommender,
        anime_df=None,
        anime_vectors=None,
        retrieve_missing=True,
    ):
        anime_vectors = anime_vectors or getattr(recommender, "anime_vectors", None)
        if anime_vectors is None:
            raise ValueError("Create anime vectors before getting rated items.")

        valid_user_scores = {
            anime_id: score
            for anime_id, score in user_scores.items()
            if score not in (None, 0, "-")
        }

        missing_rated_ids = [
            anime_id
            for anime_id in valid_user_scores
            if anime_id not in anime_vectors
        ]

        if missing_rated_ids and retrieve_missing:
            fetched_anime_data = self.get_anime_data(missing_rated_ids)
            if fetched_anime_data:
                anime_data.update({
                    str(anime_id): anime
                    for anime_id, anime in fetched_anime_data.items()
                })

                builder.anime_data = anime_data
                anime_df = builder.build_features()
                anime_vectors = recommender.create_anime_vectors(anime_df)
        elif missing_rated_ids:
            print(
                "Skipping retrieval for "
                f"{len(missing_rated_ids)} missing rated anime."
            )

        rated_items = [
            (anime_id, anime_vectors[anime_id], score)
            for anime_id, score in valid_user_scores.items()
            if anime_id in anime_vectors
        ]

        return (
            rated_items,
            anime_df,
            anime_vectors,
            getattr(recommender, "anime_df_scaled", None),
            builder,
        )
    
    def print_cache_summary(self):
        anime_cache = self._load_cache()
        print(f"Cache contains {len(anime_cache)} anime entries.")
    
    def get_cache(self):
        return self._load_cache()
