import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler


class SimilarityRecommender:
    def __init__(self):
        self.scaler = StandardScaler().set_output(transform="pandas")
        self.anime_df_scaled = None
        self.anime_vectors = None

    def create_anime_vectors(self, anime_df):
        self.anime_df_scaled = self.scaler.fit_transform(anime_df)
        self.anime_vectors = dict(
            zip(self.anime_df_scaled.index, self.anime_df_scaled.to_numpy().tolist())
        )
        return self.anime_vectors

    def create_user_vec(self, scores, anime_vectors=None, baseline_score=6):
        anime_vectors = anime_vectors or self.anime_vectors
        if anime_vectors is None:
            raise ValueError("Create anime vectors before creating a user vector.")

        user_anime_vecs = [
            np.array(anime_vectors[anime_id]) * max(score - baseline_score, 0)
            for anime_id, score in scores.items()
            if anime_id in anime_vectors
        ]

        if not user_anime_vecs:
            return np.zeros(len(next(iter(anime_vectors.values()))))

        return np.sum(user_anime_vecs, axis=0)

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
    def __init__(self, uncertainty_weight=4.5, score_min=1, score_max=10):
        self.uncertainty_weight = uncertainty_weight
        self.score_min = score_min
        self.score_max = score_max
        self.model = BayesianRidge()
        self.anime_df_scaled = None
        self.rated_ids = set()

    def fit(self, anime_df_scaled, scores):
        rated_items = [
            (anime_id, score)
            for anime_id, score in scores.items()
            if anime_id in anime_df_scaled.index and score not in (None, 0, "-")
        ]
        if not rated_items:
            raise ValueError("Need at least one scored anime to fit the reranker.")

        rated_ids = [anime_id for anime_id, _ in rated_items]
        y_train = np.array([score for _, score in rated_items], dtype=float)
        X_train = anime_df_scaled.loc[rated_ids].to_numpy()

        self.anime_df_scaled = anime_df_scaled
        self.rated_ids = set(rated_ids)
        self.model.fit(X_train, y_train)
        return self

    def predict(self, anime_ids):
        if self.anime_df_scaled is None:
            raise ValueError("Fit the reranker before predicting.")

        X_candidates = self.anime_df_scaled.loc[anime_ids].to_numpy()
        predicted_score, uncertainty = self.model.predict(X_candidates, return_std=True)
        predicted_score = np.clip(predicted_score, self.score_min, self.score_max)
        return predicted_score, uncertainty

    def rank_candidates(
        self,
        candidate_ids=None,
        uncertainty_weight=None,
        title_by_id=None,
        actual_scores=None,
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
            "uncertainty": uncertainty,
        })
        recommendations["ranking_score"] = (
            recommendations["predicted_score"]
            - uncertainty_weight * recommendations["uncertainty"]
        )

        if title_by_id is not None:
            recommendations["title"] = [
                title_by_id.get(int(anime_id), "Unknown")
                for anime_id in candidate_ids
            ]

        if actual_scores is not None:
            recommendations["actual_score"] = [
                actual_scores.get(int(anime_id))
                for anime_id in candidate_ids
            ]

        return recommendations.sort_values("ranking_score", ascending=False)
