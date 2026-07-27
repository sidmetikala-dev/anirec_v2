# AniRec v2

[Live Demo](https://anirec-app.vercel.app)

AniRec v2 is a personalized anime recommendation system built with MyAnimeList data. The project transforms user rating history and cached anime metadata into personalized recommendations through feature engineering, model comparison, offline evaluation, and full-stack deployment.

## Goals

- Build a stronger second version of the original AniRec project
- Generate personalized recommendations from a cached anime catalog
- Compare multiple recommendation approaches under the same evaluation protocol
- Evaluate recommendation quality on unseen users using ranking metrics
- Store recommendation history and user data in PostgreSQL
- Deploy the system as a usable web application

## Final Recommendation Approach

The final recommendation pipeline works as follows:

1. **Data collection and caching**
   - Fetch anime metadata and user rating histories from the MyAnimeList API
   - Cache anime records locally to reduce repeated API requests
   - Maintain a recommendation corpus containing more than 4,900 anime

2. **Feature engineering**
   - Represent each anime using numerical metadata and one-hot encoded genre indicators
   - Scale the feature matrix before calculating similarities
   - Exclude anime already rated by the user from the recommendation results

3. **Personalized recommendation**
   - Use distance-weighted cosine K-Nearest Neighbors regression
   - Treat the user's completed and scored anime as training examples
   - Use all available training ratings as neighbors
   - Give more influence to rated anime that are more similar to each candidate
   - Rank unseen anime by their predicted user rating

4. **Evaluation**
   - Randomly hold out 25% of each user's rated anime
   - Train each personalized model using the remaining ratings
   - Evaluate rankings using Precision@5, NDCG@5, and MRR@5
   - Compare KNN against Bayesian Ridge and a non-personalized global-mean baseline
   - Use the same held-out splits for all compared models

## Final Model

The strongest configuration was **distance-weighted cosine KNN** using **numerical metadata and genre indicators**.

| Setting | Value |
|---|---:|
| Final users evaluated | 69 |
| Candidate catalog | 4,800+ anime |
| Primary metric | Precision@5 |
| Precision@5 | 0.435 |
| NDCG@5 | 0.408 |
| Average relevant recommendations in top 5 | 2.17 |

## Model Comparison

The final models were evaluated on 69 users who were not used during model or feature selection.

| Model | Precision@5 | NDCG@5 |
|---|---:|---:|
| Distance-weighted cosine KNN | **0.435** | **0.408** |
| Bayesian Ridge | 0.289 | 0.266 |
| Global mean baseline | 0.159 | 0.147 |

KNN improved Precision@5 by approximately **50% relative to Bayesian Ridge** and **174% relative to the global baseline**.

## Performance by User History Size

Recommendation quality generally improved as users had more completed and scored anime.

| User score count | Users | KNN Precision@5 | Bayesian Precision@5 | Global Precision@5 |
|---|---:|---:|---:|---:|
| 50–99 | 11 | **0.261** | 0.188 | 0.100 |
| 100–199 | 27 | **0.387** | 0.259 | 0.130 |
| 200–399 | 16 | **0.469** | 0.258 | 0.169 |
| 400–799 | 13 | **0.599** | 0.432 | 0.228 |
| 800+ | 2 | **0.688** | 0.572 | 0.342 |

The 800+ group contains only two users, so its result is not treated as a stable population estimate. The more reliable result is KNN's consistent lead across the 50–799 score-count groups.

## Feature Engineering

### Final features

The final model uses:

#### Numerical metadata

- mean community score
- popularity
- number of users currently watching

#### Categorical metadata

- 77 one-hot encoded genre and theme indicators

### Features explored during development

The project also evaluated:

- synopsis TF-IDF features
- SVD-reduced synopsis embeddings
- studio indicators
- reduced numerical feature sets
- genre-only and numerical-only representations

Synopsis SVD features produced only a negligible Precision@5 improvement over numerical metadata and genres, so the simpler numerical-plus-genre feature set was selected.

## Models Evaluated

- Distance-weighted cosine KNN
- Bayesian Ridge
- Ridge regression
- Global-mean popularity baseline
- Multiple feature and uncertainty configurations

Bayesian Ridge substantially outperformed the global baseline but did not match KNN on the unseen-user evaluation.

## Data Source

The project uses the MyAnimeList API to:

- fetch user anime lists
- collect completed and scored TV anime
- retrieve anime metadata
- cache anime records locally
- build a recommendation pool from ranked and user-derived anime

## Backend and Data Storage

The backend is built with Flask and PostgreSQL.

Current backend responsibilities include:

- retrieving MyAnimeList user ratings
- loading the cached anime feature matrix
- generating personalized recommendations
- returning ranked recommendation results through an API
- storing recommendation history in PostgreSQL
- supporting reproducible recommendation and evaluation workflows

A bounded PostgreSQL connection pool is being added to replace a single shared global connection and improve connection reuse and concurrent-request reliability.

## Current Progress

Completed work includes:

- caching more than 4,900 anime records from MyAnimeList
- building reusable data-ingestion and caching workflows
- engineering numerical, genre, studio, and synopsis-based features
- implementing multiple personalized recommendation models
- tuning KNN neighborhood size and Bayesian uncertainty weighting
- evaluating feature combinations through controlled ablations
- selecting distance-weighted cosine KNN as the final model
- evaluating the final configuration on 69 unseen users
- building a Flask recommendation API
- deploying the application on Vercel
- creating a PostgreSQL-backed recommendation-history pipeline
- building Tableau visualizations for model comparison and tuning analysis

## Planned Improvements

- migrate the backend to a persistent Render web service
- replace the shared database connection with a bounded connection pool
- add user authentication and persistent user profiles
- add row-level security for user-owned data
- improve recommendation-history analysis and SQL views
- improve frontend integration and error handling
- explore vector-based candidate retrieval for larger anime catalogs

## Tech Stack

- Python
- Flask
- PostgreSQL
- scikit-learn
- Pandas
- NumPy
- MyAnimeList API
- Tableau
- Vercel

## Status

AniRec v2 has completed its first round of model selection and unseen-user evaluation. The current production model is distance-weighted cosine KNN using numerical metadata and genre indicators.

The project is still in active development. Current work includes:

- migrating the Flask backend from Vercel to Render
- containerizing the application with Docker
- testing the deployed recommendation pipeline
- improving backend reliability and frontend integration
- developing a two-tower retrieval model
- adding a separate reranking stage for retrieved candidates

The KNN system serves as the current evaluated baseline while the two-tower retrieval and reranking pipeline is being developed and compared against it.
