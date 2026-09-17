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
        self.synopsis_svd_columns = None

    def build_num_features(self, anime_data=None):
        anime_data = self.anime_data if anime_data is None else anime_data
        anime_df_num = pd.DataFrame(anime_data.values())
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
                "related_anime",
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
                "completed",
                "on_hold",
                "statistics_num_list_users",
                # "watching",
                "dropped",
                "plan_to_watch",
                "num_list_users",
                "num_scoring_users",
                "num_episodes",
                "rank",
                # "popularity",
                # "mean",
            ],
            errors="ignore",
        )

        return anime_df_num

    def build_genre_features(self, anime_data=None):
        anime_data = self.anime_data if anime_data is None else anime_data
        rows = []

        for anime in anime_data.values():
            rows.append({
                "anime_id": anime["id"],
                "genres": [genre["name"] for genre in anime.get("genres", [])]
            })

        anime_genres_df = pd.DataFrame(rows)
        genre_features = anime_genres_df["genres"].str.join("|").str.get_dummies()

        return pd.concat([anime_genres_df.drop(columns=["genres"]), genre_features], axis=1)

    def build_studio_features(self):
        rows = []

        for anime in self.anime_data.values():
            rows.append({
                "anime_id": anime["id"],
                "studios": [
                    studio["name"]
                    for studio in anime.get("studios", [])
                    if studio.get("name")
                ],
            })

        anime_studios_df = pd.DataFrame(rows)
        studio_features = (
            anime_studios_df["studios"]
            .str.join("|")
            .str.get_dummies()
            .add_prefix("studio_")
        )

        return pd.concat(
            [anime_studios_df.drop(columns=["studios"]), studio_features],
            axis=1,
        )

    def build_synopsis_features(self, anime_data=None, fit_tfidf=True):
        anime_data = self.anime_data if anime_data is None else anime_data
        rows = []

        for anime in anime_data.values():
            rows.append({
                "anime_id" : anime["id"], 
                "title" : anime["title"],
                "synopsis" : anime.get("synopsis", "")
            })

        anime_synopsis_df = pd.DataFrame(rows)

        anime_synopsis_df["synopsis"] = anime_synopsis_df["synopsis"].fillna("")

        if fit_tfidf:
            self.tfidf = TfidfVectorizer(
                max_features=self.max_tfidf_features,
                stop_words="english",
            )
            synopsis_tfidf = self.tfidf.fit_transform(anime_synopsis_df["synopsis"])
        else:
            if not fit_tfidf: return (None, None)
            if self.tfidf is None:
                raise ValueError(
                    "Call build_features with fit_tfidf=True before reusing TF-IDF."
                )
            synopsis_tfidf = self.tfidf.transform(anime_synopsis_df["synopsis"])

        return synopsis_tfidf, anime_synopsis_df["anime_id"]

    def apply_svd(self, synopsis_tfidf, anime_ids, fit_svd=True):
        if fit_svd:
            n_components = min(
                self.n_svd_components,
                synopsis_tfidf.shape[0] - 1,
                synopsis_tfidf.shape[1] - 1,
            )

            self.svd = TruncatedSVD(
                n_components=n_components,
                algorithm='randomized',
                n_iter=2,
                random_state=42
            )

            synopsis_svd = self.svd.fit_transform(synopsis_tfidf)
            self.svd_explained_variance = self.svd.explained_variance_ratio_.sum()
            self.synopsis_svd_columns = [
                f"synopsis_svd_{i}"
                for i in range(n_components)
            ]
        else:
            if not fit_svd: return None
            if self.svd is None or self.synopsis_svd_columns is None:
                raise ValueError(
                    "Call build_features with fit_svd=True before reusing SVD."
                )
            synopsis_svd = self.svd.transform(synopsis_tfidf)

        synopsis_svd_df = pd.DataFrame(
            synopsis_svd,
            columns=self.synopsis_svd_columns,
            index=anime_ids
        )

        return synopsis_svd_df

    def combine_features(
        self,
        anime_data=None,
        fit_tfidf=True,
        fit_svd=True,
    ):
        anime_data = self.anime_data if anime_data is None else anime_data
        anime_df_num = self.build_num_features(anime_data)
        anime_genres_df = self.build_genre_features(anime_data)
        synopsis_tfidf, anime_ids = self.build_synopsis_features(
            anime_data,
            fit_tfidf=fit_tfidf,
        )
        synopsis_svd_df = self.apply_svd(
            synopsis_tfidf,
            anime_ids,
            fit_svd=fit_svd,
        )
        if fit_tfidf and fit_svd:
            anime_df = pd.concat([anime_df_num.set_index("id"), 
                                anime_genres_df.set_index("anime_id"), 
                                synopsis_svd_df], axis=1)
        else:
            anime_df = pd.concat([anime_df_num.set_index("id"), 
                                anime_genres_df.set_index("anime_id")], 
                                axis=1)

        return anime_df.dropna()

    def build_features(self, fit_tfidf=True, fit_svd=True):
        return self.combine_features(
            fit_tfidf=fit_tfidf,
            fit_svd=fit_svd,
        )
