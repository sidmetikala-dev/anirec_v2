import json
from pathlib import Path

import requests
from requests.exceptions import RequestException


class AnimeDataClient:
    def __init__(self, client_id, cache_file="anime_cache.json"):
        self.cache_path = Path(cache_file)
        self.headers = {
            "X-MAL-CLIENT-ID": client_id
        }
        self.params = {
            "fields": "id, title, synopsis, mean, rank, popularity, num_list_users, num_scoring_users, media_type, status, genres, num_episodes, rating, recommendations, studios, statistics"
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
        response = requests.get(url, headers=self.headers, params=params, timeout=5)
        if response.status_code != 200:
            raise RuntimeError(f"Error {response.status_code}: {response.text}")
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
            "limit": min(page_limit, total),
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

    def cache_ranked_animes(self, ranking_type, total=1000, page_limit=500):
        anime_ids = self._get_uncached_ranked_anime_ids(
            ranking_type=ranking_type,
            total=total,
            page_limit=page_limit,
        )
        return self.get_anime_data(anime_ids)

    def get_anime_data(self, anime_ids):
        anime_cache = self._load_cache()

        anime_data = {}
        cache_changed = False
        new_fetches = 0
        save_every = 50

        print(f"Fetching details for {len(anime_ids)} anime...")

        for i, anime_id in enumerate(anime_ids, start=1):
            cache_key = str(anime_id)

            if cache_key in anime_cache:
                anime_data[anime_id] = anime_cache[cache_key]
                continue

            url = f"https://api.myanimelist.net/v2/anime/{anime_id}"
            try:
                response = requests.get(
                    url,
                    headers=self.headers,
                    params=self.params,
                    timeout=15
                )
            except RequestException as error:
                print(f"Skipping anime ID {anime_id}: {error}")
                continue

            if response.status_code == 200:
                data = response.json()
                anime_data[anime_id] = data
                anime_cache[cache_key] = data
                cache_changed = True
                new_fetches += 1
            else:
                print(f"Skipping anime ID {anime_id}: {response.status_code}")

            if i % 25 == 0 or i == len(anime_ids):
                print(f"Fetched {i}/{len(anime_ids)} anime details")

            if new_fetches > 0 and new_fetches % save_every == 0:
                print(f"Saving cache after {new_fetches} new anime...")
                self._save_cache(anime_cache)

        if cache_changed:
            print("Saving final cache...")
            self._save_cache(anime_cache)

        return anime_data
    
    def print_cache_summary(self):
        anime_cache = self._load_cache()
        print(f"Cache contains {len(anime_cache)} anime entries.")
    
    def get_cache(self):
        return self._load_cache()
