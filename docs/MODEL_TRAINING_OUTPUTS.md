# Model Training Output Structure

Training, validation, evaluation, and inference artifacts are separated from preprocessing outputs.

## Preprocessing Outputs

The root `data/processed/` directory remains the source for cleaned and graph-ready preprocessing artifacts:

- `cleaned_ratings.csv`
- `ratings_reduced_lightgcn.csv`
- `books_reduced_lightgcn.csv`
- `users_reduced_lightgcn.csv`
- `merged_reduced_lightgcn.csv`
- `user_mapping.csv`
- `book_mapping.csv`
- `edge_index.csv`
- `edge_weight.csv`
- `lightgcn_graph.pt`
- preprocessing statistics and manifests

These files are inputs to graph construction, LightGCN training, baseline training, and inference.

## Training Outputs

All training-generated outputs should be written under:

```text
data/processed/model_training/
├── checkpoints/
├── embeddings/
├── logs/
├── metrics/
├── models/
├── plots/
├── recommendations/
├── reports/
└── splits/
```

Folder purpose:

- `models/`: final trained model files, including `.pt` model weights.
- `checkpoints/`: intermediate model checkpoints if checkpointing is added.
- `embeddings/`: saved user, book, or full node embeddings.
- `metrics/`: evaluation metrics in `.csv` or `.json` format.
- `plots/`: training loss curves, metric comparisons, ranking plots, and related figures.
- `logs/`: training logs, loss tables, and run diagnostics.
- `recommendations/`: generated recommendation outputs for sample users or experiments.
- `reports/`: training summaries and experiment reports.
- `splits/`: train, validation, and test split files.

## Code Path Rules

- Preprocessing code reads from and writes to `data/processed/`.
- Model training code reads preprocessing inputs from `data/processed/`.
- Model training code writes generated outputs to `data/processed/model_training/`.
- Backend inference uses saved LightGCN embeddings from `data/processed/model_training/embeddings/` when available and falls back to popularity recommendations otherwise.

This keeps reusable preprocessing artifacts stable while preventing model experiments from filling the root processed-data directory.

## Training Command

Install the PyTorch Geometric dependency before running LightGCN training:

```powershell
pip install -r requirements.txt
```

Run the full baseline + LightGCN pipeline:

```powershell
python scripts/train_recommenders.py --epochs 50 --embedding-dim 64 --num-layers 3 --batch-size 2048
```

The runner creates train, validation, and test splits, evaluates the popularity baseline, trains PyG LightGCN, saves embeddings and model weights, writes metrics, and generates comparison plots.

## Reusable Recommendation Function

Use the reusable function from Python code or the Streamlit app:

```python
from src.book_reco.recommender_pipeline import recommend

recommendations = recommend(user_id=11676, top_k=10)
```

The function returns a list of dictionaries containing ISBN, title, author, publisher, image URL, and recommendation score. If LightGCN embeddings are unavailable or the user is unknown, it falls back to the popularity baseline.
