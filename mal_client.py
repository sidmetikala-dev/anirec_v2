import requests

class MALClient:
    def __init__(self, client_id, username):
        self.url = f"https://api.myanimelist.net/v2/users/{username}/animelist"
        self.headers = {
            "X-MAL-CLIENT-ID": client_id
        }
        self.params = {
            "sort": "list_score",
            "limit": 1000,
            "fields": "list_status, id, title, num_episodes, mean, media_type",
        }

    def _get_page(self, url, params=None):
        response = requests.get(url, headers=self.headers, params=params, timeout=15)
        if response.status_code != 200:
            raise RuntimeError(f"Error {response.status_code}: {response.text}")
        return response.json()

    def get_user_data(self):
        page = self._get_page(self.url, params=self.params)
        all_data = list(page.get("data", []))
        next_url = page.get("paging", {}).get("next")
        seen_urls = set()

        while next_url:
            if next_url in seen_urls:
                raise RuntimeError("Repeated paging URL detected while fetching user data.")
            seen_urls.add(next_url)
            page = self._get_page(next_url)
            all_data.extend(page.get("data", []))
            next_url = page.get("paging", {}).get("next")

        page["data"] = all_data
        page["paging"] = {}
        return page

    def get_scores(self, user_data):
        scores = {}
        for anime in user_data.get("data", []):
            node = anime.get("node", {})
            list_status = anime.get("list_status", {})

            if not node:
                continue
            if node.get("media_type") != "tv":
                continue
            if list_status.get("status") != "completed":
                continue

            score = list_status.get("score", 0)
            if score not in (0, "-"):
                scores[node["id"]] = score

        return scores
