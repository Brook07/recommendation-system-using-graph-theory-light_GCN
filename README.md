# Graph Theory + GNN Recommendation Platform

Production-grade design for an e-commerce recommendation system based on:
- User–Item interaction graphs (heterogeneous + temporal)
- Graph Neural Networks (GNNs) and link prediction
- Personalized ranking with real-time serving

This repo currently focuses on **complete system design** (research + engineering). Implementation can be layered on top of these specs.

## Current Project Structure
- [notebooks/01_book_crossing_preprocessing.ipynb](notebooks/01_book_crossing_preprocessing.ipynb) - preprocessing notebook for the Book-Crossing dataset
- [scripts/preprocess_book_crossing.py](scripts/preprocess_book_crossing.py) - command-line preprocessing entry point
- [src/book_reco/preprocessing.py](src/book_reco/preprocessing.py) - shared preprocessing helpers used by the notebook and script
- `data/raw-dataset-books/` - raw Book-Crossing CSV files
- `data/processed/cleaned_ratings.csv` - cleaned interaction table for downstream GNN work
- `reports/figures/` - saved preprocessing charts

## Documentation
- Full research + ML design: [docs/INDUSTRY_DESIGN.md](docs/INDUSTRY_DESIGN.md)
- System architecture, roadmap, codebase structure: [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md)
- Mermaid diagrams (architecture, dataflow, pipelines): [docs/DIAGRAMS.md](docs/DIAGRAMS.md)
- Coverage map (your 20 requirements → doc sections): [docs/SPEC_COVERAGE.md](docs/SPEC_COVERAGE.md)

## What this platform supports (target)
- Multi-relational user behavior: views, clicks, carts, purchases, favorites, ratings
- Session-aware and temporal modeling
- Two-stage serving (candidate generation + ranking) with caching + fallbacks
- Offline evaluation + online A/B testing hooks
- Monitoring, model registry, CI/CD, and Kubernetes deployment plan

## Book-Crossing Preprocessing
Run the preprocessing pipeline from the project root with:

```powershell
c:/python313/python.exe scripts/preprocess_book_crossing.py
```

That script will regenerate `data/processed/cleaned_ratings.csv` and the plots in `reports/figures/`.
