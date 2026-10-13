# AniRec v2

[Live Demo](https://anirec-app.vercel.app/)

AniRec v2 is a full-stack personalized anime recommendation application built with MyAnimeList data. The system combines a React/TypeScript frontend, Flask/Gunicorn backend, PostgreSQL persistence and caching, a personalized recommendation pipeline, and an LLM-powered tool-calling layer for natural-language recommendation requests.

The application transforms user rating histories and cached anime metadata into ranked recommendations through feature engineering, model comparison, offline evaluation, backend ranking, and agent-based orchestration.

## Architecture

AniRec is split into separate frontend and backend applications:

- **Frontend:** React + TypeScript, deployed on Vercel
- **Backend:** Python + Flask + Gunicorn, deployed on AWS EC2
- **Database:** Supabase PostgreSQL
- **Recommendation engine:** scikit-learn
- **LLM integration:** Gemini API with provider abstraction
- **Data source:** MyAnimeList API

The frontend sends either MyAnimeList usernames or natural-language recommendation prompts to the backend. The backend then handles recommendation generation, tool execution, filtering, ranking, metadata retrieval, caching, and structured response generation.

## Goals

- Generate personalized recommendations from MyAnimeList rating histories
- Support natural-language recommendation requests
- Compare multiple recommendation approaches under the same evaluation protocol
- Evaluate recommendation quality on unseen users using ranking metrics
- Keep ranking and filtering logic deterministic in Python
- Build a production-ready backend API and persistence layer
- Develop a React/TypeScript frontend
- Deploy the frontend and backend independently
- Evaluate both recommendation quality and agent behavior

## Recommendation System

### 1. Data collection and caching

- Fetch anime metadata and user rating histories from the MyAnimeList API
- Maintain a cached recommendation corpus containing more than 4,900 anime
- Handle API pagination, request failures, and rate limits
- Checkpoint long-running collection jobs to avoid losing progress

### 2. Feature engineering

Each anime is represented using:

- normalized numerical metadata
- multi-hot genre and theme indicators

Anime already rated by the user are excluded from personalized recommendation results.

### 3. Personalized recommendation

The production recommender uses distance-weighted cosine K-Nearest Neighbors regression.

For each user:

- completed and scored anime are used as training examples
- candidate anime are compared against the user's rated titles
- more similar rated anime receive greater influence
- predicted scores are generated for unseen titles
- candidates are ranked by predicted user preference

### 4. Offline evaluation

For each evaluation user:

- 25% of rated anime are randomly held out
- the model is trained using the remaining ratings
- recommendations are evaluated against the held-out titles
- identical splits are used across compared models

Ranking quality is measured using:

- Precision@5
- NDCG@5
- MRR@5

KNN was compared against Bayesian Ridge and a non-personalized global-mean baseline.

## Final Recommendation Model

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

## Agentic Recommendation Layer

AniRec also supports natural-language recommendation requests through a tool-calling agent.

Example requests include:

- "Find anime similar to Naruto"
- "Find something like Bleach but with fewer episodes"
- "Recommend anime based on my MyAnimeList profile"
- "Give me shows similar to two anime I liked"

The agent does not directly generate recommendation rankings. Instead, it orchestrates backend tools that operate on the existing recommendation system.

### Available agent operations

The agent can:

- resolve anime titles to MyAnimeList IDs
- retrieve personalized recommendations from a username
- find anime similar to one or more reference titles
- filter ranked candidates by episode count
- retrieve metadata for selected anime
- generate grounded explanations for final recommendations

Tool arguments are validated using Pydantic before execution.

## Agent Execution

The agent uses sequential tool execution.

For each request:

1. The model receives the user prompt and available tool schemas.
2. It requests at most one tool per turn.
3. The backend validates and executes the tool.
4. The result is returned to the model.
5. The model decides the next required tool.
6. Python stores the authoritative ranking.
7. The backend selects the top five eligible results.
8. Final metadata is fetched locally.
9. The LLM generates grounded explanations for the backend-selected recommendations.
10. The final response is validated against a structured schema before being returned to the frontend.

The LLM is not allowed to reorder, replace, invent, or remove backend-selected recommendations.

## Agent Reliability

The agent layer includes several safeguards:

- Pydantic validation for tool arguments
- sequential single-tool execution
- bounded model turns
- malformed tool-call recovery
- premature-answer recovery
- rejection of unresolved anime IDs
- rejection of filtering before a ranking exists
- backend-controlled ranking and top-five selection
- validation of final recommendation IDs
- structured JSON output
- provider error handling
- API key redaction in logs and evaluation output

The backend preserves ranking authority so that model output cannot silently change recommendation order.

## Agent Evaluation

AniRec includes an evaluation framework for live agent runs.

Evaluation traces can record:

- tools attempted
- raw and effective tool arguments
- rejected tool calls
- recovery attempts
- final responses
- request latency
- token usage

The evaluation framework is designed to measure:

- end-to-end task success
- tool-selection accuracy
- constraint satisfaction
- ranking preservation
- malformed-call rate
- recovery success rate
- latency
- token efficiency

## Backend

The backend is implemented using Flask and served with Gunicorn on AWS EC2.

Its responsibilities include:

- retrieving MyAnimeList user rating histories
- loading cached anime metadata and features
- generating personalized recommendations
- executing agent tools
- filtering ranked candidates
- enforcing deterministic ranking
- returning structured recommendation results
- storing recommendation history
- caching repeated recommendation requests
- handling LLM provider failures

The project also includes a multithreaded ingestion pipeline that retrieves anime metadata while handling pagination, failures, rate limits, and checkpointing.

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

The frontend is built with React and TypeScript and deployed on Vercel.

It supports:

- MyAnimeList username-based recommendations
- natural-language recommendation prompts
- ranked recommendation results
- anime images and metadata
- loading states
- backend and provider error messages

The frontend communicates with the AWS-hosted Flask API through REST requests.

## LLM Provider Support

The agent layer uses a provider-independent orchestration interface.

Current provider configuration supports:

- Gemini
- Groq
- local Ollama-based models

The provider can be selected through backend configuration without changing the frontend response format.

Gemini is used for the deployed recommendation-agent workflow.

## Data Source

AniRec uses the MyAnimeList API to:

- fetch user anime lists
- collect completed and scored anime
- retrieve anime metadata
- build and maintain the recommendation corpus
- generate user-specific training data
- resolve anime titles for natural-language recommendation requests

## Project Structure

```text
AniRec/
├── backend/
│   ├── Flask API
│   ├── Gunicorn server
│   ├── agent orchestration
│   ├── agent tools
│   ├── LLM provider client
│   ├── recommendation pipeline
│   ├── data ingestion
│   ├── feature engineering
│   ├── model evaluation
│   ├── agent evaluation
│   └── PostgreSQL persistence
│
└── frontend/
    └── React + TypeScript application
