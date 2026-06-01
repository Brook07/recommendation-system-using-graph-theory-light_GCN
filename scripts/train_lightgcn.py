"""Train LightGCN experiments and compare to the SVD baseline."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.book_reco.lightgcn import train_lightgcn, recommend_topk_from_model
import pandas as pd


def run_experiments():
    dims = [64, 128]
    results = []
    artifacts = {}
    for d in dims:
        print(f"Training LightGCN with emb dim = {d}")
        art = train_lightgcn(emb_dim=d, n_layers=3, epochs=8, batch_size=4096, lr=0.01)
        artifacts[d] = art
        metrics = art.metrics
        results.append({"Model": f"LightGCN-{d}", "Precision@10": metrics['Precision@10'], "Recall@10": metrics['Recall@10'], "NDCG@10": metrics['NDCG@10']})

    # load baseline
    baseline = pd.read_csv(Path('data/processed/baseline_metrics.csv')).iloc[0].to_dict()
    results.append({"Model": "SVD-baseline", "Precision@10": baseline['Precision@10'], "Recall@10": baseline['Recall@10'], "NDCG@10": baseline['NDCG@10']})

    results_df = pd.DataFrame(results)
    out = Path('data/processed/lightgcn_comparison.csv')
    out.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out, index=False)
    print(f"Saved comparison table to: {out.resolve()}")

    # show sample recommendations for first trained model
    sample_recs = recommend_topk_from_model(artifacts[dims[0]], topk=10)
    print("\nSample recommendations (5 users):")
    for u, recs in sample_recs.items():
        print(f"User {u}: {recs}")


if __name__ == '__main__':
    run_experiments()
