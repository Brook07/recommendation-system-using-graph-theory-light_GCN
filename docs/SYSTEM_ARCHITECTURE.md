# System Architecture Details

The GraphRec system follows a modular architecture separating data processing, model training, and model serving.

## Folder Structure

```text
graphrec/
├── backend/                  # FastAPI inference API
│   ├── main.py               # REST endpoints
│   └── inference.py          # Model loading and prediction logic
├── frontend/                 # Streamlit User Interface
│   └── app.py                # Web application
├── models/                   # Trained model weights
│   └── best_lightgcn_model.pt
├── docs/                     # Project documentation
├── data/                     # Dataset storage
│   ├── raw-dataset-books/    # Original CSV files
│   └── processed/            # Cleaned data, mappings, graph structures
├── src/                      # Core ML libraries
│   └── book_reco/            # Preprocessing, Graph, Baseline, LightGCN modules
└── scripts/                  # Executable ML pipeline scripts
```

## Data Flow Pipeline

1. **Ingestion**: Raw CSVs (`Books.csv`, `Ratings.csv`, `Users.csv`) are ingested by `src/book_reco/preprocessing.py`.
2. **Graph Construction**: The cleaned ratings are converted into a bipartite graph (`edge_index.csv`) using `graph_construction.py`.
3. **Training**: `train_lightgcn.py` loads the graph, trains the LightGCN model via BPR loss, and outputs `.pt` weights.
4. **Serving**: 
   - The FastAPI backend (`inference.py`) loads the `.pt` weights and precomputes user/item embeddings.
   - When a request is received, the backend performs a fast dot-product multiplication between the target user and all items.
   - Top K items are sorted and enriched with metadata from `Books.csv`.
5. **Consumption**: The Streamlit frontend sends requests to the backend and renders the final JSON payload as UI cards.
