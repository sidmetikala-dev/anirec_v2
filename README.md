# AniRec v2

[Live Demo](https://anirec-app.vercel.app/)

AniRec v2 is a full-stack personalized anime recommendation application built with MyAnimeList data. The system combines a React/TypeScript frontend, Flask REST API, PostgreSQL persistence and caching, and a personalized recommendation pipeline evaluated across unseen users.

The application transforms user rating histories and cached anime metadata into ranked recommendations through feature engineering, model comparison, offline evaluation, and production deployment.

## Architecture

AniRec is split into separate frontend and backend applications:

- **Frontend:** React + TypeScript, deployed on Vercel
- **Backend:** Python + Flask REST API, deployed on Google Cloud Run
- **Database:** Supabase PostgreSQL
- **Recommendation engine:** scikit-learn
- **Data source:** MyAnimeList API

The frontend communicates with the Flask API to retrieve user data and generate personalized recommendation results.

## Goals

- Build a stronger second version of the original AniRec project
- Generate personalized recommendations from a cached anime catalog
- Compare multiple recommendation approaches under the same evaluation protocol
- Evaluate recommendation quality on unseen users using ranking metrics
- Build a production-ready backend API and persistence layer
- Develop a modern React/TypeScript frontend
- Deploy the frontend and backend independently
- Continue expanding the application beyond username-based recommendations

## Recommendation Pipeline

The current recommendation pipeline works as follows:

### 1. Data collection and caching

- Fetch anime metadata and user rating histories from the MyAnimeList API
- Maintain a cached recommendation corpus containing more than 4,900 anime
- Handle API pagination, request failures, and rate limits
- Checkpoint long-running collection jobs to avoid losing progress

### 2. Feature engineering

Each anime is represented using:

- normalized numerical metadata
- multi-hot genre and theme indicators

Anime already rated by the user are excluded from the candidate set before ranking.

### 3. Personalized recommendation

The production model uses distance-weighted cosine K-Nearest Neighbors regression.

For each user:

- completed and scored anime are used as training examples
- candidate anime are compared against the user's rated titles
- more similar rated anime receive greater influence
- predicted user scores are generated for unseen titles
- candidates are ranked by predicted preference

### 4. Evaluation

For each evaluation user:

- 25% of rated anime are randomly held out
- the model is trained using the remaining ratings
- recommendations are evaluated against the held-out titles
- identical splits are used across models for fair comparison

Ranking quality is measured using:

- Precision@5
- NDCG@5
- MRR@5

KNN was compared against Bayesian Ridge and a non-personalized global-mean baseline.

## Final Model

The strongest evaluated configuration was **distance-weighted cosine KNN** using numerical metadata and genre indicators.

| Setting | Value |
|---|---:|
| Final users evaluated | 69 |
| Candidate catalog | 4,900+ anime |
| Primary metric | Precision@5 |
| Precision@5 | 0.435 |
| NDCG@5 | 0.408 |
| MRR@5 | 0.682 |
| Average relevant recommendations in top 5 | 2.17 |

## Model Comparison

The final models were evaluated across 69 users who were not used during model or feature selection.

| Model | Precision@5 | NDCG@5 |
|---|---:|---:|
| Distance-weighted cosine KNN | **0.435** | **0.408** |
| Bayesian Ridge | 0.289 | 0.266 |
| Global mean baseline | 0.159 | 0.147 |

KNN improved Precision@5 by approximately **50% relative to Bayesian Ridge** and **174% relative to the global-mean baseline**.

## Performance by User History Size

Recommendation quality generally improved as more user rating history became available.

| User score count | Users | KNN Precision@5 | Bayesian Precision@5 | Global Precision@5 |
|---|---:|---:|---:|---:|
| 50–99 | 11 | **0.261** | 0.188 | 0.100 |
| 100–199 | 27 | **0.387** | 0.259 | 0.130 |
| 200–399 | 16 | **0.469** | 0.258 | 0.169 |
| 400–799 | 13 | **0.599** | 0.432 | 0.228 |
| 800+ | 2 | **0.688** | 0.572 | 0.342 |

The 800+ group contains only two users, so its result is not treated as a stable population estimate. The more reliable result is KNN's consistent lead across the 50–799 score-count groups.

## Feature Engineering

### Production features

The current model uses:

#### Numerical metadata

- mean community score
- popularity
- number of users currently watching

#### Categorical metadata

- 77 multi-hot encoded genre and theme indicators

### Features explored during development

The project also evaluated:

- synopsis TF-IDF features
- SVD-reduced synopsis embeddings
- studio indicators
- reduced numerical feature sets
- genre-only representations
- numerical-only representations

Synopsis SVD features produced only a negligible Precision@5 improvement over numerical metadata and genres, so the simpler numerical-plus-genre representation was selected.

## Models Evaluated

- Distance-weighted cosine KNN
- Bayesian Ridge
- Ridge regression
- Global-mean baseline
- Multiple feature and uncertainty configurations

Bayesian Ridge substantially outperformed the global baseline but did not match KNN on the final unseen-user evaluation.

## Backend

The backend is implemented as a Flask REST API and deployed independently on Google Cloud Run.

Its responsibilities include:

- retrieving MyAnimeList user rating histories
- loading cached anime metadata and features
- generating personalized recommendations
- filtering already-watched anime
- returning ranked results to the frontend
- storing recommendation history
- caching repeated recommendation requests

The project also includes a multithreaded ingestion pipeline that concurrently retrieves anime metadata while handling pagination, failures, rate limits, and checkpointing.

## PostgreSQL Persistence and Caching

AniRec uses Supabase PostgreSQL for persistence and recommendation caching.

The database layer includes:

- bounded connection pooling
- indexed input-hash lookups
- bulk inserts
- idempotent writes
- cached recommendation results

Automated benchmarks showed:

- **67% lower database connection latency**
- **57% lower repeat-request latency**

Repeated requests can reuse cached results instead of performing redundant recommendation inference.

## Frontend

The frontend was rebuilt using React and TypeScript and is deployed separately on Vercel.

It communicates with the Flask backend through REST API requests and provides the user-facing interface for:

- entering MyAnimeList usernames
- requesting recommendations
- displaying ranked anime results
- presenting recommendation metadata
- handling loading and API error states

Separating the frontend and backend allows both applications to be developed and deployed independently.

## Data Source

AniRec uses the MyAnimeList API to:

- fetch user anime lists
- collect completed and scored anime
- retrieve anime metadata
- build and maintain the recommendation corpus
- generate user-specific training data

## Project Structure

```text
AniRec/
├── backend/
│   ├── Flask API
│   ├── recommendation pipeline
│   ├── data ingestion
│   ├── feature engineering
│   ├── model evaluation
│   └── PostgreSQL persistence
│
└── frontend/
    └── React + TypeScript application
