# GraphRec: Graph-Based Book Recommendation System

GraphRec is a graph-based book recommendation project built around the Book-Crossing dataset. The repository contains the full pipeline from raw CSV ingestion and preprocessing, through bipartite graph construction and LightGCN training, to a FastAPI inference layer and a Streamlit frontend for demo-style recommendation browsing.

This README summarizes the current repository state end to end. Every status label below is based on files that exist in the codebase today.

## Project Status Overview

| Area | Status | Current State |
|---|---:|---|
| Dataset collection and source handling | ✅ Completed | Raw Book-Crossing CSV files are present in `data/raw-dataset-books/` and are loaded by the preprocessing pipeline. |
| Data cleaning and preprocessing | ✅ Completed | Duplicate removal, explicit-rating filtering, sparse-user/book filtering, summary stats, and plots are implemented. |
| Feature engineering | ✅ Completed | User/book mappings, node indices, edge tables, and graph-ready interaction structures are implemented. |
| Graph construction | ✅ Completed | Bipartite graph artifacts, edge weights, graph statistics, and LightGCN tensors are generated. |
| Recommendation model implementation | ✅ Completed | A PyTorch LightGCN implementation and an SVD baseline both exist. |
| Backend/API development | ✅ Completed | FastAPI exposes `/recommend/{user_id}` and loads the model on startup. |
| Frontend/UI implementation | 🚧 In Progress | Streamlit frontend exists and is functional, but the UI is still being iterated on for layout and visual design. |
| Testing and evaluation | 🚧 In Progress | Evaluation scripts and saved metrics exist, but there is no automated test suite in the repository. |
| Deployment / production hardening | ⏳ Pending | No deployment configuration, auth, or CI/CD setup is present in the repository. |

## 1. Project Overview and Objectives

### Goal
Build an end-to-end recommendation system that can:

- learn user-book interaction patterns from the Book-Crossing dataset,
- represent the data as a bipartite graph,
- train a LightGCN model for ranking books,
- compare the graph model against a matrix-factorization baseline,
- serve recommendations through an API,
- and display them in a Streamlit interface.

### Project Focus

- Graph-based collaborative filtering
- Personalized recommendations for known users
- Popularity fallback for unknown users
- Local demo workflow using FastAPI + Streamlit

## 2. Dataset Collection and Sources

### Data present in the repository

| File | Purpose | Status |
|---|---|---:|
| `data/raw-dataset-books/Books.csv` | Raw book metadata | ✅ Completed |
| `data/raw-dataset-books/Ratings.csv` | Raw user ratings | ✅ Completed |
| `data/raw-dataset-books/Users.csv` | Raw user metadata | ✅ Completed |

### Source notes

- The repository clearly uses the Book-Crossing dataset.
- The external download source is not documented in the codebase, so this README does not invent one.

## 3. Data Cleaning and Preprocessing Completed

The preprocessing pipeline is implemented in `src/book_reco/preprocessing.py` and exposed through `scripts/preprocess_book_crossing.py`.

### Completed preprocessing steps

- Loaded raw Books, Ratings, and Users CSV files.
- Reported dataset shapes and missing values.
- Removed duplicate rows.
- Kept explicit ratings only (`Book-Rating > 0`).
- Merged ratings with book metadata.
- Iteratively filtered sparse users and sparse books.
- Built summary statistics for the cleaned dataset.
- Generated preprocessing visualizations.
- Saved cleaned interactions to `data/processed/cleaned_ratings.csv`.

### Preprocessing outputs

- `data/processed/cleaned_ratings.csv`
- `data/processed/dataset_statistics_before_after.csv`
- Figures under `reports/figures/`

## 4. Feature Engineering

Feature engineering is handled as part of the graph preparation pipeline in `src/book_reco/graph_construction.py`.

### Implemented features

- Contiguous integer IDs for users.
- Contiguous integer IDs for books with offset separation from user nodes.
- Node-type labeling (`user` / `book`).
- Bidirectional edge construction for the bipartite graph.
- Edge weights derived from explicit ratings.
- Graph summary statistics:
  - number of users,
  - number of books,
  - number of nodes,
  - number of edges,
  - average degree,
  - graph density.

### Saved feature artifacts

- `data/processed/user_mapping.csv`
- `data/processed/book_mapping.csv`
- `data/processed/edge_index.csv`
- `data/processed/edge_weight.csv`
- `data/processed/lightgcn_graph.pt`

## 5. Graph Construction

The graph construction pipeline is complete and runnable through `scripts/build_book_graph.py`.

### What it does

- Loads cleaned ratings.
- Builds user and book mappings.
- Creates a symmetric edge list for the bipartite graph.
- Saves tabular graph outputs for downstream training.
- Optionally saves a PyTorch tensor artifact for LightGCN.
- Generates graph visualizations.

### Graph artifacts present in `data/processed/`

- `edge_index.csv`
- `edge_weight.csv`
- `user_mapping.csv`
- `book_mapping.csv`
- `lightgcn_graph.pt`

## 6. Recommendation Model Implementation

### LightGCN

The core recommender is implemented in `src/book_reco/lightgcn.py`.

#### Implemented model capabilities

- LightGCN embedding layer.
- Sparse adjacency construction.
- Graph propagation across layers.
- Bayesian Personalized Ranking (BPR) loss.
- Negative sampling.
- Per-user train/validation/test splitting.
- Ranking evaluation with `Precision@10`, `Recall@10`, and `NDCG@10`.
- Model checkpoint saving.
- Sample recommendation generation.

### Baseline model

The repository also includes a matrix-factorization baseline in `src/book_reco/baseline_svd.py`.

#### Baseline capabilities

- Per-user train/test split.
- FunkSVD-style matrix factorization.
- Recommendation generation.
- Metric computation including `Precision@10`, `Recall@10`, `NDCG@10`, and `RMSE`.
- Saved baseline metrics and sample recommendation plots.

### Training scripts

| Script | Status | Purpose |
|---|---:|---|
| `scripts/train_lightgcn.py` | ✅ Completed | Trains LightGCN experiments and compares results to the baseline. |
| `scripts/run_baseline_svd.py` | ✅ Completed | Trains and evaluates the baseline matrix-factorization model. |

## 7. Backend/API Development

The backend is implemented in `backend/main.py` and `backend/inference.py`.

### Implemented backend features

- FastAPI application with CORS enabled for the frontend.
- Startup model loading.
- `/recommend/{user_id}` endpoint.
- Structured response schema using Pydantic models.
- Known-user recommendation flow using LightGCN embeddings.
- Unknown-user fallback using popularity-based recommendations.
- Metadata enrichment from the Books CSV.

### API contract

| Endpoint | Method | Status | Purpose |
|---|---|---:|---|
| `/recommend/{user_id}?top_k=10` | `GET` | ✅ Completed | Returns top recommendations for a user. |

## 8. Frontend/UI Implementation and Current Progress

The frontend lives in `frontend/app.py` and is currently a working Streamlit demo UI.

### What is implemented

- Persona-based user switching.
- Three demo personas in the current UI code.
- Custom user ID input in an expander.
- Fetch button for custom recommendations.
- Active-user badge.
- Recommendation cards with images and metadata.
- Fallback banner for unknown users.
- Custom CSS styling.

### Current UI progress

| UI Area | Status | Notes |
|---|---:|---|
| Persona switching | ✅ Completed | Three demo personas are available in the sidebar. |
| Custom user ID input | ✅ Completed | Present in the advanced expander. |
| Recommendation grid | ✅ Completed | Cards render in a multi-column layout. |
| Visual redesign iteration | 🚧 In Progress | The UI is still being refined based on design feedback. |
| Search/filter controls | ❌ Not Started | No implemented search or filter UI is present in the codebase. |
| Saved favorites / interaction actions | ❌ Not Started | No persistence or user action tracking is implemented. |

## 9. Technologies and Libraries Used

### Core stack

| Technology | Status | Usage |
|---|---:|---|
| Python | ✅ Completed | Main language for the full project. |
| Pandas | ✅ Completed | Data loading, cleaning, merging, and metric tables. |
| NumPy | ✅ Completed | Array operations and evaluation helpers. |
| PyTorch | ✅ Completed | LightGCN model and tensor artifacts. |
| scikit-learn | ✅ Completed | Included in dependencies for ML support. |
| FastAPI | ✅ Completed | Backend inference API. |
| Uvicorn | ✅ Completed | ASGI server for FastAPI. |
| Pydantic | ✅ Completed | API schemas. |
| Streamlit | ✅ Completed | Frontend application. |
| Requests | ✅ Completed | Frontend-to-backend communication. |
| Matplotlib | ✅ Completed | Preprocessing and graph visualizations. |
| NetworkX | ✅ Completed | Sample graph visualization. |
| tqdm | ✅ Completed | Progress display in baseline training. |

## 10. Folder Structure

```text
recommendation-system-using-graph-theory/
├── backend/
│   ├── main.py
│   └── inference.py
├── data/
│   ├── raw-dataset-books/
│   │   ├── Books.csv
│   │   ├── Ratings.csv
│   │   └── Users.csv
│   └── processed/
│       ├── cleaned_ratings.csv
│       ├── edge_index.csv
│       ├── edge_weight.csv
│       ├── user_mapping.csv
│       ├── book_mapping.csv
│       ├── lightgcn_graph.pt
│       └── model_training/
│           ├── models/
│           ├── embeddings/
│           ├── metrics/
│           ├── plots/
│           ├── logs/
│           ├── recommendations/
│           ├── reports/
│           └── splits/
├── docs/
│   ├── DEMO_SCRIPT.md
│   ├── DIAGRAMS.md
│   ├── IMPLEMENTATION.md
│   ├── SYSTEM_ARCHITECTURE.md
│   └── USER_GUIDE.md
├── frontend/
│   └── app.py
├── notebooks/
│   └── 01_book_crossing_preprocessing.ipynb
├── reports/
│   └── figures/
├── scripts/
│   ├── build_book_graph.py
│   ├── preprocess_book_crossing.py
│   ├── run_baseline_svd.py
│   └── train_lightgcn.py
├── src/
│   └── book_reco/
│       ├── baseline_svd.py
│       ├── graph_construction.py
│       ├── lightgcn.py
│       └── preprocessing.py
├── requirements.txt
└── README.md
```

## 11. Features Completed

- ✅ Raw Book-Crossing datasets are included in the repository.
- ✅ Data cleaning and sparse filtering are implemented.
- ✅ Graph-ready user/book mappings are generated.
- ✅ Bipartite graph artifacts are created and saved.
- ✅ LightGCN model code is implemented in PyTorch.
- ✅ BPR loss and negative sampling are implemented.
- ✅ SVD baseline training and evaluation are implemented.
- ✅ FastAPI backend is implemented.
- ✅ Fallback recommendations for unknown users are implemented.
- ✅ Streamlit frontend is implemented.
- ✅ Documentation files for architecture and user guidance are present.

## 12. Features In Progress

- 🚧 Frontend visual redesign and interaction polish.
- 🚧 UI refinement for persona selection and recommendation presentation.
- 🚧 Ongoing improvements to the demo experience.

## 13. Pending Tasks

- ⏳ Automated test suite.
- ⏳ CI/CD pipeline.
- ⏳ Deployment configuration.
- ⏳ Authentication or user accounts.
- ⏳ Search and filtering UI.
- ⏳ Saved favorites / bookmarks.
- ⏳ Production monitoring and logging.
- ⏳ Model registry or experiment tracking integration.

## 14. Known Issues and Limitations

- ❌ No automated tests were found in the repository.
- ❌ No CI workflow or deployment manifest is present.
- ❌ The frontend depends on the backend running locally at `http://localhost:8000`.
- ❌ The repository does not document an external source URL for the Book-Crossing dataset.
- ❌ The current Streamlit UI is still being iterated on and is not yet a final polished product.
- ❌ The project currently uses local file-based artifacts rather than a managed data store.

## 15. Testing and Evaluation Status

### Evaluation artifacts present

| Artifact | Status | Purpose |
|---|---:|---|
| `data/processed/model_training/metrics/baseline_metrics.csv` | ✅ Completed | Baseline evaluation output. |
| `data/processed/model_training/metrics/lightgcn_metrics*.csv` | ✅ Completed | LightGCN evaluation outputs. |
| `data/processed/model_training/metrics/lightgcn_comparison.csv` | ✅ Completed | Comparison table between models. |
| `reports/figures/` | ✅ Completed | Saved charts and recommendation examples. |

### Testing status

| Test Area | Status | Notes |
|---|---:|---|
| Manual end-to-end flow | ✅ Completed | Backend + frontend are wired together through the API. |
| Model evaluation metrics | ✅ Completed | Precision, Recall, NDCG, and RMSE are computed in code. |
| Automated unit tests | ❌ Not Started | No test files were found. |
| Integration test suite | ❌ Not Started | No test framework setup was found. |

## 16. Project Milestones Achieved

1. ✅ Book-Crossing raw data was added to the repository.
2. ✅ Preprocessing pipeline was implemented and saved cleaned interactions.
3. ✅ Graph construction pipeline was implemented.
4. ✅ LightGCN and baseline SVD models were implemented.
5. ✅ Evaluation metrics and experiment artifacts were generated.
6. ✅ FastAPI backend was implemented for inference.
7. ✅ Streamlit frontend was implemented for demonstration.
8. ✅ Documentation files were added for architecture, diagrams, and user flow.

## 17. Next Steps

### Recommended next work items

- [ ] Finalize the new frontend UI direction.
- [ ] Add search and filtering controls to the demo.
- [ ] Add automated tests for preprocessing, graph building, and inference.
- [ ] Add deployment instructions and environment packaging.
- [ ] Add logging and error handling improvements.
- [ ] Add experiment tracking for model comparisons.
- [ ] Document the external dataset source if you want the README to include a citation.

## Run the Project Locally

### Install dependencies

```powershell
pip install -r requirements.txt
```

### Start the backend

```powershell
uvicorn backend.main:app --reload
```

### Start the frontend

```powershell
streamlit run frontend/app.py
```

## Documentation

- [Implementation roadmap](docs/IMPLEMENTATION.md)
- [Model training output structure](docs/MODEL_TRAINING_OUTPUTS.md)
- [System diagrams](docs/DIAGRAMS.md)
- [User guide](docs/USER_GUIDE.md)
- [Demo script](docs/DEMO_SCRIPT.md)
- [Model Explanation: Training and Inference](docs/MODEL_EXPLANATION_README.md)

## Notes

- This README is intentionally based only on files currently present in the repository.
- Where the repository does not document a detail, the README marks it as pending or not started instead of inventing information.
