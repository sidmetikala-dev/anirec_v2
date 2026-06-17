import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge, ElasticNetCV, LassoCV, RidgeCV


class HitRateEvaluator:
    def __init__(
        self,
        anime_df_scaled,
        scores,
        anime_df=None,
        like_threshold=None,
        heldout_fraction=0.25,
    ):
        self.anime_df_scaled = anime_df_scaled
        self.anime_df = anime_df
        self.scores = scores
        self.heldout_fraction = heldout_fraction

        self.rated_eval = self._build_rated_eval()
        if self.rated_eval.empty:
            raise ValueError("Need scored anime that exist in anime_df_scaled.")

        score_values = self.rated_eval["score"].to_numpy(dtype=float)
        self.like_threshold = (
            np.median(score_values) + 0.5 * np.std(score_values)
            if like_threshold is None
            else like_threshold
        )
        self.liked_eval = self.rated_eval[
            self.rated_eval["score"] >= self.like_threshold
        ]
        if len(self.liked_eval) < 2:
            raise ValueError("Need at least 2 liked anime for holdout evaluation.")

    def _build_rated_eval(self):
        rows = [
            {"anime_id": anime_id, "score": score}
            for anime_id, score in self.scores.items()
            if anime_id in self.anime_df_scaled.index and score not in (None, 0, "-")
        ]
        return pd.DataFrame(rows)

    def _sample_split(self, random_state=None):
        heldout_liked = self.liked_eval.sample(
            frac=self.heldout_fraction,
            random_state=random_state,
        )
        heldout_ids = set(heldout_liked["anime_id"])

        train_eval = self.rated_eval[
            ~self.rated_eval["anime_id"].isin(heldout_ids)
        ]
        train_ids = train_eval["anime_id"].tolist()
        candidate_ids = [
            anime_id
            for anime_id in self.anime_df_scaled.index
            if anime_id not in set(train_ids)
        ]

        return train_eval, train_ids, candidate_ids, heldout_ids

    def _score_top_k(self, recommendations, top_ks, heldout_ids):
        rows = []
        for k in top_ks:
            top_k = recommendations.head(k)
            hits = int(top_k["is_hidden_like"].sum())
            rows.append({
                "k": k,
                "hits": hits,
                "heldout_likes": len(heldout_ids),
                "hit_rate": hits / len(heldout_ids),
                "precision_at_k": hits / k,
            })
        return rows

    @staticmethod
    def summarize(results, group_cols):
        return (
            results
            .groupby(group_cols)
            .agg(
                avg_precision_at_k=("precision_at_k", "mean"),
                std_precision_at_k=("precision_at_k", "std"),
                avg_hit_rate=("hit_rate", "mean"),
                std_hit_rate=("hit_rate", "std"),
                avg_hits=("hits", "mean"),
            )
            .reset_index()
            .sort_values(
                [group_cols[-1], "avg_precision_at_k"],
                ascending=[True, False],
            )
        )

    def evaluate_bayesian_once(
        self,
        uncertainty_weight=4.5,
        top_ks=(5, 10, 20, 50, 100),
        random_state=None,
    ):
        train_eval, train_ids, candidate_ids, heldout_ids = self._sample_split(
            random_state=random_state
        )

        X_train = self.anime_df_scaled.loc[train_ids].to_numpy()
        y_train = train_eval["score"].to_numpy(dtype=float)
        X_candidates = self.anime_df_scaled.loc[candidate_ids].to_numpy()

        model = BayesianRidge()
        model.fit(X_train, y_train)
        predicted_score, uncertainty = model.predict(
            X_candidates,
            return_std=True,
        )
        predicted_score = np.clip(predicted_score, 1, 10)

        recommendations = pd.DataFrame({
            "anime_id": candidate_ids,
            "predicted_score": predicted_score,
            "uncertainty": uncertainty,
        })
        recommendations["ranking_score"] = (
            recommendations["predicted_score"]
            - uncertainty_weight * recommendations["uncertainty"]
        )
        recommendations["is_hidden_like"] = recommendations["anime_id"].isin(
            heldout_ids
        )
        recommendations = recommendations.sort_values(
            "ranking_score",
            ascending=False,
        )

        return pd.DataFrame(
            self._score_top_k(recommendations, top_ks, heldout_ids)
        )

    def tune_bayesian_uncertainty(
        self,
        weights,
        n_runs=50,
        top_ks=(5, 10),
        random_state=None,
    ):
        rng = np.random.default_rng(random_state)
        rows = []
        baseline_rows = []

        for run in range(n_runs):
            split_seed = None if random_state is None else int(rng.integers(0, 2**32 - 1))
            train_eval, train_ids, candidate_ids, heldout_ids = self._sample_split(
                random_state=split_seed
            )

            X_train = self.anime_df_scaled.loc[train_ids].to_numpy()
            y_train = train_eval["score"].to_numpy(dtype=float)
            X_candidates = self.anime_df_scaled.loc[candidate_ids].to_numpy()

            model = BayesianRidge()
            model.fit(X_train, y_train)
            predicted_score, uncertainty = model.predict(
                X_candidates,
                return_std=True,
            )
            predicted_score = np.clip(predicted_score, 1, 10)

            base_recommendations = pd.DataFrame({
                "anime_id": candidate_ids,
                "is_hidden_like": pd.Series(candidate_ids)
                .isin(heldout_ids)
                .to_numpy(),
                "predicted_score": predicted_score,
                "uncertainty": uncertainty,
            })

            if self.anime_df is not None and "mean" in self.anime_df.columns:
                baseline_recommendations = base_recommendations[
                    ["anime_id", "is_hidden_like"]
                ].copy()
                baseline_recommendations["baseline_score"] = self.anime_df.loc[
                    baseline_recommendations["anime_id"],
                    "mean",
                ].to_numpy()
                baseline_recommendations = baseline_recommendations.sort_values(
                    "baseline_score",
                    ascending=False,
                )

                for row in self._score_top_k(
                    baseline_recommendations,
                    top_ks,
                    heldout_ids,
                ):
                    row["run"] = run + 1
                    baseline_rows.append(row)

            for weight in weights:
                recommendations = base_recommendations.copy()
                recommendations["ranking_score"] = (
                    recommendations["predicted_score"]
                    - weight * recommendations["uncertainty"]
                )
                recommendations = recommendations.sort_values(
                    "ranking_score",
                    ascending=False,
                )

                for row in self._score_top_k(recommendations, top_ks, heldout_ids):
                    row["run"] = run + 1
                    row["uncertainty_weight"] = weight
                    rows.append(row)

        results = pd.DataFrame(rows)
        summary = self.summarize(results, ["uncertainty_weight", "k"])
        best_weights = (
            summary
            .sort_values(
                ["k", "avg_precision_at_k", "avg_hit_rate"],
                ascending=[True, False, False],
            )
            .groupby("k")
            .head(1)
        )

        baseline_results = pd.DataFrame(baseline_rows)
        baseline_summary = None
        if not baseline_results.empty:
            baseline_summary = (
                baseline_results
                .groupby("k")
                .agg(
                    baseline_avg_precision_at_k=("precision_at_k", "mean"),
                    baseline_std_precision_at_k=("precision_at_k", "std"),
                    baseline_avg_hit_rate=("hit_rate", "mean"),
                    baseline_std_hit_rate=("hit_rate", "std"),
                    baseline_avg_hits=("hits", "mean"),
                )
                .reset_index()
            )
            best_weights = best_weights.merge(
                baseline_summary,
                on="k",
                how="left",
            )

        return results, summary, best_weights, baseline_results, baseline_summary

    def compare_bayesian_ridge(
        self,
        uncertainty_weight=4.5,
        n_runs=50,
        top_ks=(5, 10),
        ridge_alphas=None,
        include_lasso=False,
        include_elastic_net=False,
        lasso_alphas=None,
        elastic_alphas=None,
        elastic_l1_ratios=None,
        cv=3,
        random_state=None,
    ):
        ridge_alphas = (
            np.logspace(-2, 3, 100)
            if ridge_alphas is None
            else ridge_alphas
        )
        lasso_alphas = (
            np.logspace(-4, 1, 50)
            if lasso_alphas is None
            else lasso_alphas
        )
        elastic_alphas = (
            np.logspace(-4, 1, 50)
            if elastic_alphas is None
            else elastic_alphas
        )
        elastic_l1_ratios = (
            [0.1, 0.3, 0.5, 0.7, 0.9]
            if elastic_l1_ratios is None
            else elastic_l1_ratios
        )
        rng = np.random.default_rng(random_state)
        rows = []

        for run in range(n_runs):
            split_seed = None if random_state is None else int(rng.integers(0, 2**32 - 1))
            train_eval, train_ids, candidate_ids, heldout_ids = self._sample_split(
                random_state=split_seed
            )

            X_train = self.anime_df_scaled.loc[train_ids].to_numpy()
            y_train = train_eval["score"].to_numpy(dtype=float)
            X_candidates = self.anime_df_scaled.loc[candidate_ids].to_numpy()

            bayes_model = BayesianRidge()
            bayes_model.fit(X_train, y_train)
            bayes_pred, bayes_std = bayes_model.predict(
                X_candidates,
                return_std=True,
            )
            bayes_score = np.clip(bayes_pred, 1, 10) - uncertainty_weight * bayes_std

            ridge_model = RidgeCV(alphas=ridge_alphas)
            ridge_model.fit(X_train, y_train)
            ridge_score = np.clip(ridge_model.predict(X_candidates), 1, 10)

            lasso_model = None
            lasso_score = None
            if include_lasso:
                lasso_model = LassoCV(
                    alphas=lasso_alphas,
                    cv=cv,
                    max_iter=10000,
                    random_state=42,
                )
                lasso_model.fit(X_train, y_train)
                lasso_score = np.clip(lasso_model.predict(X_candidates), 1, 10)

            elastic_model = None
            elastic_score = None
            if include_elastic_net:
                elastic_model = ElasticNetCV(
                    alphas=elastic_alphas,
                    l1_ratio=elastic_l1_ratios,
                    cv=cv,
                    max_iter=10000,
                    random_state=42,
                )
                elastic_model.fit(X_train, y_train)
                elastic_score = np.clip(elastic_model.predict(X_candidates), 1, 10)

            run_recommendations = pd.DataFrame({
                "anime_id": candidate_ids,
                "is_hidden_like": pd.Series(candidate_ids)
                .isin(heldout_ids)
                .to_numpy(),
                "bayesian_ridge": bayes_score,
                "ridge_cv": ridge_score,
            })

            model_names = ["bayesian_ridge", "ridge_cv"]
            if include_lasso:
                run_recommendations["lasso_cv"] = lasso_score
                model_names.append("lasso_cv")
            if include_elastic_net:
                run_recommendations["elastic_net_cv"] = elastic_score
                model_names.append("elastic_net_cv")
            if self.anime_df is not None and "mean" in self.anime_df.columns:
                run_recommendations["global_mean"] = self.anime_df.loc[
                    candidate_ids,
                    "mean",
                ].to_numpy()
                model_names.append("global_mean")

            for model_name in model_names:
                recommendations = run_recommendations.sort_values(
                    model_name,
                    ascending=False,
                )
                for row in self._score_top_k(recommendations, top_ks, heldout_ids):
                    row["run"] = run + 1
                    row["model"] = model_name
                    row["ridge_alpha"] = (
                        ridge_model.alpha_
                        if model_name == "ridge_cv"
                        else np.nan
                    )
                    row["lasso_alpha"] = (
                        lasso_model.alpha_
                        if model_name == "lasso_cv"
                        else np.nan
                    )
                    row["elastic_alpha"] = (
                        elastic_model.alpha_
                        if model_name == "elastic_net_cv"
                        else np.nan
                    )
                    row["elastic_l1_ratio"] = (
                        elastic_model.l1_ratio_
                        if model_name == "elastic_net_cv"
                        else np.nan
                    )
                    rows.append(row)

        results = pd.DataFrame(rows)
        summary = self.summarize(results, ["model", "k"])
        return results, summary
