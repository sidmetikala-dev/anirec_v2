import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


class AnimeFeatureBuilder:
    def __init__(self, anime_data, max_tfidf_features=3000, n_svd_components=300):
        self.anime_data = anime_data
        self.max_tfidf_features = max_tfidf_features
        self.n_svd_components = n_svd_components
        self.tfidf = None
        self.svd = None
        self.svd_explained_variance = None

    def build_num_features(self):
        anime_df_num = pd.DataFrame(self.anime_data.values())
        anime_df_num = anime_df_num.drop(
            columns=[
                "main_picture",
                "title",
                "synopsis",
                "media_type",
                "status",
                "genres",
                "rating",
                "recommendations",
                "studios",
            ],
            errors="ignore",
        )

        statistics_df = pd.json_normalize(anime_df_num["statistics"])
        statistics_df = statistics_df.rename(columns={
            "num_list_users": "statistics_num_list_users",
            "status.watching": "watching",
            "status.completed": "completed",
            "status.on_hold": "on_hold",
            "status.dropped": "dropped",
            "status.plan_to_watch": "plan_to_watch",
        })
        statistics_df = statistics_df.apply(pd.to_numeric, errors="coerce")

        anime_df_num = pd.concat([anime_df_num.drop(columns=["statistics"]), statistics_df], axis=1)

        anime_df_num = anime_df_num.drop(
            columns=[
                "statistics_num_list_users",
                "watching",
                "completed",
                "on_hold",
                "dropped",
                "plan_to_watch",
                "num_list_users",
                "rank",
            ],
            errors="ignore",
        )

        return anime_df_num

    def build_genre_features(self):
        rows = []

        for anime in self.anime_data.values():
            rows.append({
                "anime_id": anime["id"],
                "genres": [genre["name"] for genre in anime.get("genres", [])]
            })

        anime_genres_df = pd.DataFrame(rows)
        genre_features = anime_genres_df["genres"].str.join("|").str.get_dummies()

        return pd.concat([anime_genres_df.drop(columns=["genres"]), genre_features], axis=1)

    def build_synopsis_features(self):
        rows = []

        for anime in self.anime_data.values():
            rows.append({
                "anime_id" : anime["id"], 
                "title" : anime["title"],
                "synopsis" : anime.get("synopsis", "")
            })

        anime_synopsis_df = pd.DataFrame(rows)

        anime_synopsis_df["synopsis"] = anime_synopsis_df["synopsis"].fillna("")

        self.tfidf = TfidfVectorizer(
            max_features=self.max_tfidf_features,
            stop_words="english",
        )
        synopsis_tfidf = self.tfidf.fit_transform(anime_synopsis_df["synopsis"])

        synopsis_features = pd.DataFrame(
            synopsis_tfidf.toarray(),
            columns=self.tfidf.get_feature_names_out(),
            index=anime_synopsis_df["anime_id"]
        )

        return synopsis_tfidf, synopsis_features

    def apply_svd(self, synopsis_tfidf, anime_ids):
        n_components = min(
            self.n_svd_components,
            synopsis_tfidf.shape[0] - 1,
            synopsis_tfidf.shape[1] - 1,
        )

        self.svd = TruncatedSVD(
            n_components=n_components,
            random_state=42
        )

        synopsis_svd = self.svd.fit_transform(synopsis_tfidf)
        self.svd_explained_variance = self.svd.explained_variance_ratio_.sum()

        synopsis_svd_df = pd.DataFrame(
            synopsis_svd,
            columns=[f"synopsis_svd_{i}" for i in range(n_components)],
            index=anime_ids
        )

        return synopsis_svd_df

    def combine_features(self):
        anime_df_num = self.build_num_features()
        anime_genres_df = self.build_genre_features()
        synopsis_tfidf, synopsis_features = self.build_synopsis_features()
        synopsis_svd_df = self.apply_svd(synopsis_tfidf, synopsis_features.index)

        anime_df = pd.concat([anime_df_num.set_index("id"), 
                              anime_genres_df.set_index("anime_id"), 
                              synopsis_svd_df], axis=1)

        return anime_df.dropna()

    def build_features(self):
        return self.combine_features()
