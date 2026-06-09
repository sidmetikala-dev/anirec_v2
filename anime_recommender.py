import numpy as np
from sklearn.preprocessing import StandardScaler


class ContentBasedRecommender:
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
