# Implementation Roadmap and System Architecture

This document describes the high-level roadmap and the implementation details of the GraphRec platform.

## 1. Project Roadmap

### Phase 1: Research & ML Pipeline (Completed)
- **Data Preprocessing**: Cleaned the Book-Crossing dataset, removed sparse users/books.
- **SVD Baseline**: Established a Matrix Factorization baseline.
- **Graph Construction**: Mapped users and items to integer nodes, created bipartite graph edge indices.
- **LightGCN Model**: Implemented and trained a PyTorch-based Graph Neural Network using BPR loss.

### Phase 2: Production Inference API (Completed)
- **FastAPI Service**: Created `backend/main.py` serving as the REST API.
- **Inference Engine**: Created `backend/inference.py` to handle PyTorch model loading, embeddings precomputation, and fast ranking.
- **Popularity Fallback**: Implemented a fallback mechanism for unknown users.

### Phase 3: Frontend Application (Completed)
- **Streamlit App**: Built `frontend/app.py` for a modern, responsive user interface.
- **API Integration**: Connected the frontend directly to the FastAPI backend.

### Phase 4: Future Enhancements (Planned)
- **Multi-Relational Data**: Add clicks, carts, and views to the graph.
- **Session-Aware Modeling**: Use RNNs or Transformers to capture temporal context.
- **Two-Stage Serving**: Implement a lightweight candidate generator (e.g. Faiss) before the LightGCN ranker.
- **MLOps**: Kubernetes deployment, model registry, online A/B testing hooks.

## 2. System Architecture

- **Data Layer**: Cleaned interactions stored in CSV files, precomputed graph structures in `.pt` and `.csv`.
- **Model Layer**: LightGCN `.pt` weights stored in `models/`.
- **API Layer**: FastAPI handles HTTP GET requests, parses User IDs, and returns JSON recommendations.
- **Presentation Layer**: Streamlit consumes the REST API and renders results in an interactive card grid.
