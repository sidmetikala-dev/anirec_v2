# AniRec v2

AniRec v2 is a personalized anime recommendation system built with MyAnimeList data. The project combines content-based candidate generation with personalized reranking to recommend anime based on a user's ratings, anime metadata, genres, and synopsis-based features.

## Goals

- Build a better version of the original AniRec project
- Generate personalized recommendations from a cached anime dataset
- Use content-based similarity to find strong candidates
- Use a per-user model to rerank candidates based on rating behavior
- Structure the project cleanly and document progress through Git commits

## Current Approach

The recommendation pipeline is being built in two stages:

1. Candidate generation
   - Represent anime using metadata, genre encoding, and synopsis features
   - Build a user profile from anime the user rated highly
   - Generate candidate anime from the cached dataset using similarity

2. Personalized reranking
   - Train a per-user regression model on anime the user has already scored
   - Predict scores for candidate anime
   - Return the highest-ranked unseen anime

## Features Being Explored

- Numeric metadata
  - mean score
  - popularity
  - number of episodes
  - selected statistics fields

- Categorical features
  - genres
  - studios
  - source and related metadata later

- Text features
  - synopsis with TF-IDF
  - dimensionality reduction with SVD

## Data Source

This project uses the MyAnimeList API to:
- fetch user anime lists
- cache anime metadata locally
- build a recommendation pool from ranked and user-derived anime

## Project Status

This project is currently in development.

Current progress includes:
- fetching and caching anime data from MyAnimeList
- collecting user rating data
- exploring feature usefulness with regression
- building a content-based recommendation pipeline

## Planned Improvements

- user authentication
- saved user profiles
- cached recommendation results
- recommendation dashboard and statistics
- deployment as a full-stack web app
