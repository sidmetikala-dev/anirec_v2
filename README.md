# AniRec v2

[Live Demo](https://anirec-app.vercel.app)

AniRec v2 is a personalized anime recommendation system built with MyAnimeList data. The project focuses on turning user rating history and cached anime metadata into personalized recommendations through feature engineering, model evaluation, and backend deployment.

## Goals

- Build a stronger second version of the original AniRec project
- Generate personalized recommendations from a cached anime dataset
- Evaluate recommendation quality using offline ranking metrics
- Store recommendation history and user data in PostgreSQL
- Deploy the system as a usable web application

## Current Approach

The current recommendation pipeline works as follows:

1. Data collection and caching
   - Fetch anime metadata and user rating data from the MyAnimeList API
   - Cache anime records locally to reduce repeated API calls
   - Maintain a recommendation corpus built from ranked and user-derived anime

2. Feature engineering
   - Represent anime using numeric metadata, genre encodings, and synopsis-based text features
   - Apply TF-IDF to synopses and reduce dimensionality with SVD
   - Combine structured and text-derived features into a unified anime representation

3. Personalized recommendation
   - Fit a per-user Bayesian Ridge reranking model on anime the user has already scored
   - Score unseen anime from the cached dataset
   - Return the top-ranked personalized recommendations

4. Evaluation and tracking
   - Compare recommendation performance across models, uncertainty settings, and feature choices
   - Evaluate quality using ranking metrics such as Precision@5 and NDCG@5
   - Store recommendation history for analysis and reproducibility

## Features Used

### Numeric metadata
- mean score
- popularity
- number of episodes
- selected user-count statistics such as watching

### Categorical features
- genres
- related metadata from cached anime records

### Text features
- synopsis with TF-IDF
- dimensionality reduction with SVD

## Data Source

This project uses the MyAnimeList API to:
- fetch user anime lists
- collect user rating histories
- cache anime metadata locally
- build a recommendation pool from ranked and user-derived anime

## Current Progress

Current work completed includes:
- caching 4,900+ anime records from MyAnimeList
- building reusable data ingestion and caching workflows
- engineering combined metadata, genre, and synopsis-based feature sets
- evaluating multiple recommendation models with offline ranking metrics
- selecting a Bayesian Ridge reranking approach as the strongest current model
- building a Flask-based recommendation API
- deploying the application on Vercel
- creating a PostgreSQL-backed recommendation history pipeline
- building Tableau visualizations for model comparison and tuning analysis

## Planned Improvements

- user authentication
- row-level security and persistent user profiles
- better recommendation history analysis and SQL views
- improved frontend integration
- optional vector-based retrieval for larger anime corpora
- more robust deployment and production-ready backend structure

## Tech Stack

- Python
- Flask
- PostgreSQL
- Tableau
- Vercel
- MyAnimeList API
- scikit-learn
- Pandas

## Status

AniRec v2 is currently in active development.
