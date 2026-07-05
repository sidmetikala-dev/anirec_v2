import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


class SimilarityRecommender:
    def __init__(self):
        self.raw_numeric_columns = [
            "mean",
            "rank",
            "popularity",
            "num_list_users",
            "num_scoring_users",
            "num_episodes",
            "statistics_num_list_users",
            "watching",
            "completed",
            "on_hold",
            "dropped",
            "plan_to_watch",
        ]
        self.preprocessor = None
        self.vector_columns = None
        self.candidate_feature_columns = None
        self.anime_df_scaled = None
        self.anime_vectors = None

    def create_anime_vectors(self, anime_df):
        numeric_cols = [
            column
            for column in self.raw_numeric_columns
            if column in anime_df.columns
        ]
        svd_cols = [
            column
            for column in anime_df.columns
            if str(column).startswith("synopsis_svd_")
        ]
        passthrough_cols = [
            column
            for column in anime_df.columns
            if column not in numeric_cols + svd_cols
        ]

        transformers = []
        if numeric_cols:
            transformers.append(("numeric", StandardScaler(), numeric_cols))
        if passthrough_cols:
            transformers.append(("passthrough", "passthrough", passthrough_cols))
        if svd_cols:
            transformers.append(("svd", "passthrough", svd_cols))

        if not transformers:
            raise ValueError("anime_df must contain at least one feature column.")

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            sparse_threshold=0,
        )

        self.vector_columns = numeric_cols + passthrough_cols + svd_cols
        self.candidate_feature_columns = passthrough_cols + svd_cols
        self.preprocessor = preprocessor
        scaled_array = preprocessor.fit_transform(anime_df)
        self.anime_df_scaled = pd.DataFrame(
            scaled_array,
            index=anime_df.index,
            columns=self.vector_columns,
        )
        self.anime_vectors = dict(
            zip(self.anime_df_scaled.index, self.anime_df_scaled.to_numpy().tolist())
        )
        return self.anime_vectors

    def create_user_vec(self, scores, anime_vectors=None, baseline_score=6):
        anime_vectors = anime_vectors or self.anime_vectors
        if anime_vectors is None:
            raise ValueError("Create anime vectors before creating a user vector.")

        total_absolute_weight = 0
        user_anime_vecs = []
        for anime_id, score in scores.items():
            weight = max((score - baseline_score), 0)
            if weight == 0:
                continue

            user_anime_vecs.append(np.array(anime_vectors[anime_id]) * weight)
            total_absolute_weight += abs(weight)

        if not user_anime_vecs:
            return np.zeros(len(next(iter(anime_vectors.values()))))

        return np.sum(user_anime_vecs, axis=0) / total_absolute_weight

    def generate_candidates(self, scores, top_k=400, user_vec=None):
        if self.anime_df_scaled is None:
            raise ValueError("Create anime vectors before generating candidates.")

        if user_vec is None:
            user_vec = self.create_user_vec(scores)

        user_norm = np.linalg.norm(user_vec)
        if user_norm == 0:
            return []

        anime_matrix = self.anime_df_scaled.to_numpy()
        anime_ids = self.anime_df_scaled.index.to_numpy()
        anime_norms = np.linalg.norm(anime_matrix, axis=1)
        unseen_mask = ~np.isin(anime_ids, list(scores.keys()))
        valid_mask = unseen_mask & (anime_norms > 0)

        if not np.any(valid_mask):
            return []

        cosine_sims = np.full(len(anime_ids), -np.inf)
        cosine_sims[valid_mask] = (
            anime_matrix[valid_mask] @ user_vec
        ) / (anime_norms[valid_mask] * user_norm)

        candidate_count = min(top_k, int(np.sum(valid_mask)))
        candidate_idx = np.argpartition(cosine_sims, -candidate_count)[-candidate_count:]
        candidate_idx = candidate_idx[np.argsort(cosine_sims[candidate_idx])[::-1]]

        return list(zip(anime_ids[candidate_idx].tolist(), cosine_sims[candidate_idx].tolist()))


class BayesianRidgeRecommender:
    def __init__(
        self,
        anime_data_client=None,
        uncertainty_weight=20,
        score_min=1,
        score_max=10,
        user_scores=None,
        anime_data=None,
        builder=None,
        recommender=None,
        anime_df=None,
        anime_df_scaled=None,
        anime_vectors=None,
        clip_predictions=False,
    ):
        self.uncertainty_weight = uncertainty_weight
        self.score_min = score_min
        self.score_max = score_max
        self.model = BayesianRidge()
        self.user_scores = user_scores
        self.anime_data = anime_data
        self.anime_df = anime_df
        self.anime_df_scaled = anime_df_scaled
        self.recommender = recommender
        self.rated_items = []
        self.rated_ids = set()
        self.clip_predictions = clip_predictions

        if anime_data_client is not None:
            if (
                user_scores is None
                or anime_data is None
                or builder is None
                or recommender is None
            ):
                raise ValueError(
                    "user_scores, anime_data, builder, and recommender are "
                    "required when anime_data_client is provided."
                )

            (
                self.rated_items,
                self.anime_df,
                anime_vectors,
                self.anime_df_scaled,
                _,
            ) = anime_data_client.get_rated_items(
                user_scores=user_scores,
                anime_data=anime_data,
                builder=builder,
                recommender=recommender,
                anime_df=anime_df,
                anime_vectors=(
                    anime_vectors
                    or getattr(recommender, "anime_vectors", None)
                ),
            )

        self.anime_vectors = anime_vectors
        if self.anime_df_scaled is None and recommender is not None:
            self.anime_df_scaled = getattr(recommender, "anime_df_scaled", None)

        if not self.rated_items and self.user_scores is not None:
            self.rated_items = self._build_rated_items(self.user_scores)

    def _build_rated_items(self, scores):
        if self.anime_df_scaled is None:
            raise ValueError(
                "anime_df_scaled is required to build rated training items."
            )

        return [
            (anime_id, self.anime_df_scaled.loc[anime_id].to_numpy(), score)
            for anime_id, score in scores.items()
            if (
                anime_id in self.anime_df_scaled.index
                and score not in (None, 0, "-")
            )
        ]

    @staticmethod
    def _scale_uncertainty(uncertainty, lower_quantile=0.05, upper_quantile=0.95):
        uncertainty = np.asarray(uncertainty, dtype=float)
        if uncertainty.size == 0:
            return uncertainty

        lower, upper = np.quantile(
            uncertainty,
            [lower_quantile, upper_quantile],
        )
        clipped = np.clip(uncertainty, lower, upper)

        if np.isclose(clipped.max(), clipped.min()):
            return np.zeros_like(clipped)

        return MinMaxScaler().fit_transform(
            clipped.reshape(-1, 1)
        ).ravel()

    def fit(self):
        if self.anime_df_scaled is None:
            raise ValueError("anime_df_scaled is required to fit the reranker.")

        if not self.rated_items:
            raise ValueError("Need at least one scored anime to fit the reranker.")

        rated_ids = [anime_id for anime_id, _, _ in self.rated_items]
        y_train = np.array([score for _, _, score in self.rated_items], dtype=float)
        X_train = self.anime_df_scaled.loc[rated_ids].to_numpy()

        self.rated_ids = set(rated_ids)
        self.model.fit(X_train, y_train)
        return self

    def predict(self, anime_ids):
        if self.anime_df_scaled is None:
            raise ValueError("Fit the reranker before predicting.")

        X_candidates = self.anime_df_scaled.loc[anime_ids].to_numpy()
        predicted_score, uncertainty = self.model.predict(X_candidates, return_std=True)
        if self.clip_predictions:
            predicted_score = np.clip(predicted_score, self.score_min, self.score_max)
        return predicted_score, uncertainty

    def rank_candidates(
        self,
        candidate_ids=None,
        uncertainty_weight=None,
    ):
        if self.anime_df_scaled is None:
            raise ValueError("Fit the reranker before ranking candidates.")

        if candidate_ids is None:
            candidate_ids = [
                anime_id
                for anime_id in self.anime_df_scaled.index
                if anime_id not in self.rated_ids
            ]

        uncertainty_weight = (
            self.uncertainty_weight
            if uncertainty_weight is None
            else uncertainty_weight
        )
        predicted_score, uncertainty = self.predict(candidate_ids)

        recommendations = pd.DataFrame({
            "anime_id": candidate_ids,
            "predicted_score": predicted_score,
            "uncertainty_raw": uncertainty,
        })
        recommendations["uncertainty"] = self._scale_uncertainty(
            recommendations["uncertainty_raw"]
        )
        recommendations["ranking_score"] = (
            recommendations["predicted_score"]
            - uncertainty_weight * recommendations["uncertainty"]
        )

        return recommendations.sort_values("ranking_score", ascending=False)

    def _get_title(self, anime_id, anime_data):
        anime = anime_data.get(str(anime_id))
        if anime is None:
            return "Unknown"
        return anime.get("title", "Unknown")
    
    def _valid_user_score_count(self):
        if self.user_scores is None:
            return 0

        return sum(
            1
            for score in self.user_scores.values()
            if score not in (None, 0, "-")
        )

    def _similarity_recs_for_small_profile(self, top_k, score_count_threshold=100):
        if (
            self.user_scores is None
            or self.recommender is None
            or self._valid_user_score_count() >= score_count_threshold
        ):
            return None

        generated_candidates = self.recommender.generate_candidates(
            self.user_scores,
            top_k=top_k,
        )
        rows = [
            {
                "anime_id": anime_id,
                "predicted_score": similarity,
                "uncertainty_raw": 0.0,
                "uncertainty": 0.0,
                "ranking_score": similarity,
            }
            for anime_id, similarity in generated_candidates
            if anime_id in self.anime_df_scaled.index
            and anime_id not in self.rated_ids
        ]
        if not rows:
            return None

        recs = pd.DataFrame(rows)
        if self.anime_data is None:
            recs["title"] = "Unknown"
            return recs

        recs["title"] = recs["anime_id"].apply(
            lambda anime_id: self._get_title(anime_id, self.anime_data)
        )
        return recs

    def get_unfiltered_recs(self):
        ranked_df = self.rank_candidates().copy()

        if self.anime_data is None:
            ranked_df["title"] = "Unknown"
            return ranked_df

        ranked_df["title"] = ranked_df["anime_id"].apply(
            lambda anime_id: self._get_title(anime_id, self.anime_data)
        )
        return ranked_df
    
    def get_recs(self, top_k=5, filter_unwatched_prequels=True):
        recs = (
            self._similarity_recs_for_small_profile(top_k)
            if self._valid_user_score_count() < 100
            else None
        )
        if recs is None:
            recs = self.get_unfiltered_recs()

        def check_prequel(anime_id):
            if self.anime_data is None or self.user_scores is None:
                return True

            anime = self.anime_data.get(str(anime_id), {})
            related_animes = anime.get("related_anime", [])

            prequel_ids = [
                related_anime["node"]["id"]
                for related_anime in related_animes
                if related_anime.get("relation_type") == "prequel"
            ]

            if not prequel_ids:
                return True

            return all(prequel_id in self.user_scores for prequel_id in prequel_ids)

        if not filter_unwatched_prequels:
            return recs.head(top_k).reset_index(drop=True)

        valid_indices = []
        for index, row in recs.iterrows():
            if len(valid_indices) >= top_k:
                break
            if check_prequel(row["anime_id"]):
                valid_indices.append(index)

        return recs.loc[valid_indices].reset_index(drop=True)
