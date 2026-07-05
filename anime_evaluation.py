import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge, ElasticNetCV, LassoCV, RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler


class HitRateEvaluator:
    def __init__(
        self,
        anime_df_scaled,
        scores,
        anime_df=None,
        anime_data_client=None,
        anime_data=None,
        builder=None,
        recommender=None,
        like_threshold=None,
        heldout_fraction=0.25,
    ):
        self.anime_df_scaled = anime_df_scaled
        self.anime_df = anime_df
        self.scores = scores
        self.recommender = recommender
        self.heldout_fraction = heldout_fraction

        if anime_data_client is not None:
            if anime_data is None or builder is None or recommender is None:
                raise ValueError(
                    "anime_data, builder, and recommender are required when "
                    "anime_data_client is provided."
                )

            (
                _,
                self.anime_df,
                _,
                self.anime_df_scaled,
                _,
            ) = anime_data_client.get_rated_items(
                user_scores=self.scores,
                anime_data=anime_data,
                builder=builder,
                recommender=recommender,
                anime_df=self.anime_df,
                anime_vectors=getattr(recommender, "anime_vectors", None),
            )

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

    def _generated_candidate_ids(
        self,
        train_eval,
        train_ids,
        candidate_top_k=None,
        use_candidate_generation=None,
    ):
        if use_candidate_generation is None:
            use_candidate_generation = candidate_top_k is not None

        if not use_candidate_generation:
            return [
                anime_id
                for anime_id in self.anime_df_scaled.index
                if anime_id not in set(train_ids)
            ]

        if self.recommender is None:
            raise ValueError(
                "recommender is required when candidate generation is enabled."
            )

        if candidate_top_k is None:
            candidate_top_k = len(self.anime_df_scaled)

        train_scores = dict(
            zip(
                train_eval["anime_id"].tolist(),
                train_eval["score"].tolist(),
            )
        )
        generated_candidates = self.recommender.generate_candidates(
            train_scores,
            top_k=candidate_top_k,
        )
        train_id_set = set(train_ids)
        return [
            anime_id
            for anime_id, _ in generated_candidates
            if anime_id in self.anime_df_scaled.index
            and anime_id not in train_id_set
        ]

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
    def _sort_with_anime_tiebreakers(recommendations, score_column):
        sort_columns = [score_column]
        ascending = [False]

        if "num_scoring_users" in recommendations.columns:
            sort_columns.append("num_scoring_users")
            ascending.append(False)

        sort_columns.append("anime_id")
        ascending.append(True)

        return recommendations.sort_values(sort_columns, ascending=ascending)

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

    @staticmethod
    def _prepare_prediction_scores(predicted_score, clip_predictions=False):
        predicted_score = np.asarray(predicted_score, dtype=float)
        if clip_predictions:
            return np.clip(predicted_score, 1, 10)
        return predicted_score

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
        uncertainty_weight=8.5,
        top_ks=(5, 10, 20, 50, 100),
        random_state=None,
        clip_predictions=False,
        candidate_top_k=None,
        use_candidate_generation=None,
    ):
        train_eval, train_ids, candidate_ids, heldout_ids = self._sample_split(
            random_state=random_state
        )
        candidate_ids = self._generated_candidate_ids(
            train_eval,
            train_ids,
            candidate_top_k,
            use_candidate_generation,
        )
        if not candidate_ids:
            raise ValueError("No candidates available for this split.")

        X_train = self.anime_df_scaled.loc[train_ids].to_numpy()
        y_train = train_eval["score"].to_numpy(dtype=float)
        X_candidates = self.anime_df_scaled.loc[candidate_ids].to_numpy()

        model = BayesianRidge()
        model.fit(X_train, y_train)
        predicted_score, uncertainty = model.predict(
            X_candidates,
            return_std=True,
        )
        predicted_score = self._prepare_prediction_scores(
            predicted_score,
            clip_predictions=clip_predictions,
        )

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
        similarity_weights=None,
        n_runs=50,
        top_ks=(5, 10),
        random_state=None,
        clip_predictions=False,
        candidate_top_k=None,
        use_candidate_generation=None,
    ):
        if similarity_weights is None:
            active_similarity_weights = [None]
        elif np.isscalar(similarity_weights):
            active_similarity_weights = [similarity_weights]
        else:
            active_similarity_weights = list(similarity_weights)

        use_similarity = similarity_weights is not None
        if use_similarity and use_candidate_generation is False:
            raise ValueError(
                "similarity_weights requires candidate generation to be enabled."
            )
        if use_similarity and self.recommender is None:
            raise ValueError(
                "recommender is required when similarity_weights is provided."
            )

        rng = np.random.default_rng(random_state)
        rows = []
        baseline_rows = []

        for run in range(n_runs):
            split_seed = None if random_state is None else int(rng.integers(0, 2**32 - 1))
            train_eval, train_ids, candidate_ids, heldout_ids = self._sample_split(
                random_state=split_seed
            )
            similarity_by_id = {}
            if use_similarity:
                active_candidate_top_k = (
                    len(self.anime_df_scaled)
                    if candidate_top_k is None
                    else candidate_top_k
                )

                train_scores = dict(
                    zip(
                        train_eval["anime_id"].tolist(),
                        train_eval["score"].tolist(),
                    )
                )
                generated_candidates = self.recommender.generate_candidates(
                    train_scores,
                    top_k=active_candidate_top_k,
                )
                train_id_set = set(train_ids)
                candidate_ids = [
                    anime_id
                    for anime_id, _ in generated_candidates
                    if anime_id in self.anime_df_scaled.index
                    and anime_id not in train_id_set
                ]
                similarity_by_id = dict(generated_candidates)
            else:
                candidate_ids = self._generated_candidate_ids(
                    train_eval,
                    train_ids,
                    candidate_top_k,
                    use_candidate_generation,
                )
            if not candidate_ids:
                continue

            X_train = self.anime_df_scaled.loc[train_ids].to_numpy()
            y_train = train_eval["score"].to_numpy(dtype=float)
            X_candidates = self.anime_df_scaled.loc[candidate_ids].to_numpy()

            model = BayesianRidge()
            model.fit(X_train, y_train)
            predicted_score, uncertainty = model.predict(
                X_candidates,
                return_std=True,
            )
            predicted_score = self._prepare_prediction_scores(
                predicted_score,
                clip_predictions=clip_predictions,
            )

            base_recommendations = pd.DataFrame({
                "anime_id": candidate_ids,
                "is_hidden_like": pd.Series(candidate_ids)
                .isin(heldout_ids)
                .to_numpy(),
                "predicted_score": predicted_score,
                "uncertainty_raw": uncertainty,
            })
            if use_similarity:
                base_recommendations["candidate_similarity"] = [
                    similarity_by_id.get(anime_id, 0.0)
                    for anime_id in candidate_ids
                ]
            base_recommendations["uncertainty"] = self._scale_uncertainty(
                base_recommendations["uncertainty_raw"]
            )

            if self.anime_df is not None and "mean" in self.anime_df.columns:
                baseline_recommendations = base_recommendations[
                    ["anime_id", "is_hidden_like"]
                ].copy()
                baseline_recommendations["baseline_score"] = self.anime_df.loc[
                    baseline_recommendations["anime_id"],
                    "mean",
                ].to_numpy()
                if "num_scoring_users" in self.anime_df.columns:
                    baseline_recommendations["num_scoring_users"] = self.anime_df.loc[
                        baseline_recommendations["anime_id"],
                        "num_scoring_users",
                    ].to_numpy()
                baseline_recommendations = self._sort_with_anime_tiebreakers(
                    baseline_recommendations,
                    "baseline_score",
                )

                for row in self._score_top_k(
                    baseline_recommendations,
                    top_ks,
                    heldout_ids,
                ):
                    row["run"] = run + 1
                    baseline_rows.append(row)

            for weight in weights:
                for similarity_weight in active_similarity_weights:
                    recommendations = base_recommendations.copy()
                    recommendations["ranking_score"] = (
                        recommendations["predicted_score"]
                        - weight * recommendations["uncertainty"]
                    )
                    if similarity_weight is not None:
                        recommendations["ranking_score"] += (
                            similarity_weight
                            * recommendations["candidate_similarity"]
                        )
                    recommendations = recommendations.sort_values(
                        "ranking_score",
                        ascending=False,
                    )

                    for row in self._score_top_k(
                        recommendations,
                        top_ks,
                        heldout_ids,
                    ):
                        row["run"] = run + 1
                        row["uncertainty_weight"] = weight
                        if similarity_weight is not None:
                            row["similarity_weight"] = similarity_weight
                        rows.append(row)

        results = pd.DataFrame(rows)
        group_cols = ["uncertainty_weight", "k"]
        if use_similarity:
            group_cols = ["uncertainty_weight", "similarity_weight", "k"]
        summary = self.summarize(results, group_cols)
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
        baseline_summary = pd.DataFrame(columns=[
            "k",
            "baseline_avg_precision_at_k",
            "baseline_std_precision_at_k",
            "baseline_avg_hit_rate",
            "baseline_std_hit_rate",
            "baseline_avg_hits",
        ])
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

    def compare_models(
        self,
        uncertainty_weight=8.5,
        n_runs=50,
        top_ks=(5, 10),
        ridge_alphas=None,
        include_ridge_cv=True,
        include_lasso=False,
        include_elastic_net=False,
        include_knn=True,
        include_hybrid_model=False,
        hybrid_global_weight=0.6,
        hybrid_bayesian_weight=0.4,
        include_gradient_boosting=True,
        knn_neighbors=20,
        knn_weights="distance",
        knn_metric="cosine",
        gradient_boosting_params=None,
        lasso_alphas=None,
        elastic_alphas=None,
        elastic_l1_ratios=None,
        cv=3,
        random_state=None,
        clip_predictions=False,
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
        gradient_boosting_params = (
            {
                "n_estimators": 100,
                "learning_rate": 0.05,
                "max_depth": 2,
                "random_state": 42,
            }
            if gradient_boosting_params is None
            else gradient_boosting_params
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
            bayes_pred = self._prepare_prediction_scores(
                bayes_pred,
                clip_predictions=clip_predictions,
            )
            bayes_score = (
                bayes_pred - uncertainty_weight * self._scale_uncertainty(bayes_std)
            )

            ridge_model = None
            ridge_score = None
            if include_ridge_cv:
                ridge_model = RidgeCV(alphas=ridge_alphas)
                ridge_model.fit(X_train, y_train)
                ridge_score = self._prepare_prediction_scores(
                    ridge_model.predict(X_candidates),
                    clip_predictions=clip_predictions,
                )

            knn_model = None
            knn_score = None
            if include_knn:
                knn_model = KNeighborsRegressor(
                    n_neighbors=min(knn_neighbors, len(train_ids)),
                    weights=knn_weights,
                    metric=knn_metric,
                )
                knn_model.fit(X_train, y_train)
                knn_score = self._prepare_prediction_scores(
                    knn_model.predict(X_candidates),
                    clip_predictions=clip_predictions,
                )

            gradient_model = None
            gradient_score = None
            if include_gradient_boosting:
                gradient_model = GradientBoostingRegressor(
                    **gradient_boosting_params
                )
                gradient_model.fit(X_train, y_train)
                gradient_score = self._prepare_prediction_scores(
                    gradient_model.predict(X_candidates),
                    clip_predictions=clip_predictions,
                )

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
                lasso_score = self._prepare_prediction_scores(
                    lasso_model.predict(X_candidates),
                    clip_predictions=clip_predictions,
                )

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
                elastic_score = self._prepare_prediction_scores(
                    elastic_model.predict(X_candidates),
                    clip_predictions=clip_predictions,
                )

            run_recommendations = pd.DataFrame({
                "anime_id": candidate_ids,
                "is_hidden_like": pd.Series(candidate_ids)
                .isin(heldout_ids)
                .to_numpy(),
                "bayesian_ridge": bayes_score,
            })

            model_names = ["bayesian_ridge"]
            if include_ridge_cv:
                run_recommendations["ridge_cv"] = ridge_score
                model_names.append("ridge_cv")
            if include_knn:
                run_recommendations["knn"] = knn_score
                model_names.append("knn")
            if include_gradient_boosting:
                run_recommendations["gradient_boosting"] = gradient_score
                model_names.append("gradient_boosting")
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
                if include_hybrid_model:
                    run_recommendations["hybrid_model"] = (
                        hybrid_global_weight * run_recommendations["global_mean"]
                        + hybrid_bayesian_weight * run_recommendations["bayesian_ridge"]
                    )
                if "num_scoring_users" in self.anime_df.columns:
                    run_recommendations["num_scoring_users"] = self.anime_df.loc[
                        candidate_ids,
                        "num_scoring_users",
                    ].to_numpy()
                if include_hybrid_model:
                    model_names.append("hybrid_model")
                model_names.append("global_mean")

            for model_name in model_names:
                if model_name in ("global_mean", "hybrid_model"):
                    recommendations = self._sort_with_anime_tiebreakers(
                        run_recommendations,
                        model_name,
                    )
                else:
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
                    row["knn_neighbors"] = (
                        knn_model.n_neighbors
                        if model_name == "knn"
                        else np.nan
                    )
                    rows.append(row)

        results = pd.DataFrame(rows)
        summary = self.summarize(results, ["model", "k"])
        return results, summary

    def tune_hybrid_weights(
        self,
        global_weights=None,
        uncertainty_weight=8.5,
        n_runs=50,
        top_ks=(5, 10),
        random_state=None,
        include_global_mean=True,
        include_bayesian=True,
        clip_predictions=False,
    ):
        if self.anime_df is None or "mean" not in self.anime_df.columns:
            raise ValueError("anime_df with a mean column is required for hybrid tuning.")

        global_weights = (
            np.linspace(0, 1, 11)
            if global_weights is None
            else np.asarray(global_weights, dtype=float)
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

            model = BayesianRidge()
            model.fit(X_train, y_train)
            bayes_pred, bayes_std = model.predict(
                X_candidates,
                return_std=True,
            )
            bayes_pred = self._prepare_prediction_scores(
                bayes_pred,
                clip_predictions=clip_predictions,
            )
            bayesian_score = (
                bayes_pred - uncertainty_weight * self._scale_uncertainty(bayes_std)
            )
            global_mean_score = self.anime_df.loc[candidate_ids, "mean"].to_numpy()

            recommendations = pd.DataFrame({
                "anime_id": candidate_ids,
                "is_hidden_like": pd.Series(candidate_ids)
                .isin(heldout_ids)
                .to_numpy(),
                "bayesian_ridge": bayes_pred,
                "global_mean": global_mean_score,
            })
            if "num_scoring_users" in self.anime_df.columns:
                recommendations["num_scoring_users"] = self.anime_df.loc[
                    candidate_ids,
                    "num_scoring_users",
                ].to_numpy()

            if include_bayesian:
                bayesian_recommendations = recommendations.sort_values(
                    "bayesian_ridge",
                    ascending=False,
                )
                for row in self._score_top_k(
                    bayesian_recommendations,
                    top_ks,
                    heldout_ids,
                ):
                    row["run"] = run + 1
                    row["model"] = "bayesian_ridge"
                    row["global_weight"] = 0.0
                    row["bayesian_weight"] = 1.0
                    rows.append(row)

            if include_global_mean:
                global_recommendations = self._sort_with_anime_tiebreakers(
                    recommendations,
                    "global_mean",
                )
                for row in self._score_top_k(
                    global_recommendations,
                    top_ks,
                    heldout_ids,
                ):
                    row["run"] = run + 1
                    row["model"] = "global_mean"
                    row["global_weight"] = 1.0
                    row["bayesian_weight"] = 0.0
                    rows.append(row)

            for global_weight in global_weights:
                bayesian_weight = 1 - global_weight
                hybrid_recommendations = recommendations.copy()
                hybrid_recommendations["hybrid_model"] = (
                    global_weight * hybrid_recommendations["global_mean"]
                    + bayesian_weight * hybrid_recommendations["bayesian_ridge"]
                )
                hybrid_recommendations = self._sort_with_anime_tiebreakers(
                    hybrid_recommendations,
                    "hybrid_model",
                )

                for row in self._score_top_k(
                    hybrid_recommendations,
                    top_ks,
                    heldout_ids,
                ):
                    row["run"] = run + 1
                    row["model"] = "hybrid_model"
                    row["global_weight"] = global_weight
                    row["bayesian_weight"] = bayesian_weight
                    rows.append(row)

        results = pd.DataFrame(rows)
        summary = self.summarize(results, ["model", "global_weight", "bayesian_weight", "k"])
        best_weights = (
            summary[summary["model"] == "hybrid_model"]
            .sort_values(
                ["k", "avg_precision_at_k", "avg_hit_rate"],
                ascending=[True, False, False],
            )
            .groupby("k")
            .head(1)
        )

        return results, summary, best_weights
    
class RankingMetricEvaluator(HitRateEvaluator):
    def __init__(
        self,
        anime_df_scaled,
        scores,
        anime_df=None,
        anime_data_client=None,
        anime_data=None,
        builder=None,
        recommender=None,
        like_threshold=None,
        relevance_threshold=6,
        heldout_fraction=0.25,
    ):
        super().__init__(
            anime_df_scaled=anime_df_scaled,
            anime_df=anime_df,
            scores=scores,
            anime_data_client=anime_data_client,
            anime_data=anime_data,
            builder=builder,
            recommender=recommender,
            like_threshold=like_threshold,
            heldout_fraction=heldout_fraction,
        )
        self.relevance_threshold = relevance_threshold
        self.rated_eval = self._add_relevance(self.rated_eval)

    def _add_relevance(self, rated_eval):
        rated_eval = rated_eval.copy()
        scores = rated_eval["score"].astype(float)
        rated_eval["relevance"] = np.select(
            [
                scores >= self.like_threshold,
                (scores > self.relevance_threshold)
                & (scores < self.like_threshold),
            ],
            [2, 1],
            default=0,
        )
        return rated_eval

    def _sample_ranking_split(self, random_state=None):
        test_eval = self.rated_eval.sample(
            frac=self.heldout_fraction,
            random_state=random_state,
        )
        test_ids = set(test_eval["anime_id"])

        train_eval = self.rated_eval[
            ~self.rated_eval["anime_id"].isin(test_ids)
        ]
        train_ids = train_eval["anime_id"].tolist()
        candidate_ids = [
            anime_id
            for anime_id in self.anime_df_scaled.index
            if anime_id not in set(train_ids)
        ]

        return train_eval, train_ids, candidate_ids, test_eval

    @staticmethod
    def dcg_at_k(relevance, k):
        relevance = np.asarray(relevance, dtype=float)[:k]
        if relevance.size == 0:
            return 0.0

        discounts = np.log2(np.arange(2, relevance.size + 2))
        gains = np.power(2, relevance) - 1
        return float(np.sum(gains / discounts))

    @classmethod
    def ndcg_at_k(cls, ranked_relevance, ideal_relevance, k):
        ideal_dcg = cls.dcg_at_k(sorted(ideal_relevance, reverse=True), k)
        if np.isclose(ideal_dcg, 0):
            return np.nan

        return cls.dcg_at_k(ranked_relevance, k) / ideal_dcg

    @staticmethod
    def mrr_at_k(ranked_relevance, k):
        for rank, relevance in enumerate(ranked_relevance[:k], start=1):
            if relevance > 0:
                return 1 / rank

        return 0.0

    def _score_ranking_top_k(self, recommendations, top_ks, test_eval):
        ranked_relevance = recommendations["relevance"].to_numpy()
        ideal_relevance = test_eval["relevance"].to_numpy()
        rows = []

        for k in top_ks:
            top_k_relevance = ranked_relevance[:k]
            rows.append({
                "k": k,
                "ndcg_at_k": self.ndcg_at_k(
                    ranked_relevance,
                    ideal_relevance,
                    k,
                ),
                "mrr_at_k": self.mrr_at_k(ranked_relevance, k),
                "relevant_hits_at_k": int(np.sum(top_k_relevance > 0)),
                "strong_hits_at_k": int(np.sum(top_k_relevance == 2)),
                "test_relevant": int(np.sum(ideal_relevance > 0)),
                "test_strong_relevant": int(np.sum(ideal_relevance == 2)),
            })

        return rows

    @staticmethod
    def summarize_ranking(results, group_cols):
        return (
            results
            .groupby(group_cols)
            .agg(
                avg_ndcg_at_k=("ndcg_at_k", "mean"),
                std_ndcg_at_k=("ndcg_at_k", "std"),
                avg_mrr_at_k=("mrr_at_k", "mean"),
                std_mrr_at_k=("mrr_at_k", "std"),
                avg_relevant_hits_at_k=("relevant_hits_at_k", "mean"),
                avg_strong_hits_at_k=("strong_hits_at_k", "mean"),
                avg_test_relevant=("test_relevant", "mean"),
                avg_test_strong_relevant=("test_strong_relevant", "mean"),
            )
            .reset_index()
            .sort_values(
                [group_cols[-1], "avg_ndcg_at_k"],
                ascending=[True, False],
            )
        )

    def evaluate_bayesian_once(
        self,
        uncertainty_weight=8.5,
        top_ks=(5, 10),
        random_state=None,
        include_global_mean=True,
        clip_predictions=False,
    ):
        train_eval, train_ids, candidate_ids, test_eval = (
            self._sample_ranking_split(random_state=random_state)
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
        predicted_score = self._prepare_prediction_scores(
            predicted_score,
            clip_predictions=clip_predictions,
        )
        uncertainty = self._scale_uncertainty(uncertainty)

        relevance_by_id = test_eval.set_index("anime_id")["relevance"].to_dict()
        base_recommendations = pd.DataFrame({
            "anime_id": candidate_ids,
            "predicted_score": predicted_score,
            "uncertainty": uncertainty,
        })
        base_recommendations["ranking_score"] = (
            base_recommendations["predicted_score"]
            - uncertainty_weight * base_recommendations["uncertainty"]
        )
        base_recommendations["relevance"] = (
            base_recommendations["anime_id"]
            .map(relevance_by_id)
            .fillna(0)
            .astype(int)
        )

        ranked_frames = {
            "bayesian_ridge": base_recommendations.sort_values(
                "ranking_score",
                ascending=False,
            )
        }

        if (
            include_global_mean
            and self.anime_df is not None
            and "mean" in self.anime_df.columns
        ):
            baseline_recommendations = base_recommendations[
                ["anime_id", "relevance"]
            ].copy()
            baseline_recommendations["global_mean"] = self.anime_df.loc[
                baseline_recommendations["anime_id"],
                "mean",
            ].to_numpy()
            if "num_scoring_users" in self.anime_df.columns:
                baseline_recommendations["num_scoring_users"] = self.anime_df.loc[
                    baseline_recommendations["anime_id"],
                    "num_scoring_users",
                ].to_numpy()
            ranked_frames["global_mean"] = self._sort_with_anime_tiebreakers(
                baseline_recommendations,
                "global_mean",
            )

        rows = []
        for model_name, recommendations in ranked_frames.items():
            for row in self._score_ranking_top_k(
                recommendations,
                top_ks,
                test_eval,
            ):
                row["model"] = model_name
                rows.append(row)

        return pd.DataFrame(rows)

    def tune_bayesian_uncertainty_ranking(
        self,
        weights,
        n_runs=50,
        top_ks=(5, 10),
        random_state=None,
        clip_predictions=False,
    ):
        rng = np.random.default_rng(random_state)
        rows = []

        for run in range(n_runs):
            split_seed = None if random_state is None else int(
                rng.integers(0, 2**32 - 1)
            )
            train_eval, train_ids, candidate_ids, test_eval = (
                self._sample_ranking_split(random_state=split_seed)
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
            predicted_score = self._prepare_prediction_scores(
                predicted_score,
                clip_predictions=clip_predictions,
            )
            uncertainty = self._scale_uncertainty(uncertainty)

            relevance_by_id = test_eval.set_index("anime_id")["relevance"].to_dict()
            base_recommendations = pd.DataFrame({
                "anime_id": candidate_ids,
                "predicted_score": predicted_score,
                "uncertainty": uncertainty,
            })
            base_recommendations["relevance"] = (
                base_recommendations["anime_id"]
                .map(relevance_by_id)
                .fillna(0)
                .astype(int)
            )

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

                for row in self._score_ranking_top_k(
                    recommendations,
                    top_ks,
                    test_eval,
                ):
                    row["run"] = run + 1
                    row["uncertainty_weight"] = weight
                    rows.append(row)

        results = pd.DataFrame(rows)
        summary = self.summarize_ranking(results, ["uncertainty_weight", "k"])
        return results, summary

    def evaluate_bayesian(
        self,
        uncertainty_weight=8.5,
        n_runs=50,
        top_ks=(5, 10),
        random_state=None,
        include_global_mean=True,
        clip_predictions=False,
    ):
        rng = np.random.default_rng(random_state)
        rows = []

        for run in range(n_runs):
            split_seed = None if random_state is None else int(
                rng.integers(0, 2**32 - 1)
            )
            run_results = self.evaluate_bayesian_once(
                uncertainty_weight=uncertainty_weight,
                top_ks=top_ks,
                random_state=split_seed,
                include_global_mean=include_global_mean,
                clip_predictions=clip_predictions,
            )
            run_results["run"] = run + 1
            rows.append(run_results)

        results = pd.concat(rows, ignore_index=True)
        summary = self.summarize_ranking(results, ["model", "k"])
        return results, summary
