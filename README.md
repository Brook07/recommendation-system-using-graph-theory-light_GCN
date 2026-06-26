# GraphRec: End-to-End GNN Book Recommendation Platform

GraphRec is a complete, production-grade e-commerce recommendation system based on Graph Neural Networks (GNNs). 

It models user-book interactions as a bipartite graph, trains a highly efficient **LightGCN** model natively in PyTorch, and serves real-time personalized recommendations via a modern **FastAPI** backend and **Streamlit** frontend.

## 🚀 Features

- **Advanced ML Pipeline**: Custom LightGCN implementation trained with Bayesian Personalized Ranking (BPR) loss and negative sampling.
- **FastAPI Inference Engine**: Real-time REST API for serving model predictions.
- **Smart Fallback Mechanism**: Gracefully handles the "Cold Start" problem for unknown users by falling back to popularity-based recommendations.
- **Modern UI**: An interactive, card-based Streamlit web application.
- **Comprehensive Docs**: Detailed architecture and sequence diagrams.

## 📁 Project Structure

```text
graphrec/
├── backend/          # FastAPI inference API (main.py, inference.py)
├── frontend/         # Streamlit User Interface (app.py)
├── models/           # Trained PyTorch model weights (.pt files)
├── docs/             # Diagrams, implementation details, and demo script
├── data/             # Raw and processed datasets, graph mappings
├── src/              # Core ML libraries (preprocessing, graph, LightGCN)
└── scripts/          # ML pipeline runner scripts
```

## 🛠️ Installation & Setup

1. **Install Dependencies**:
   Ensure you have Python 3.10+ installed.
   ```powershell
   pip install -r requirements.txt
   ```

2. **Start the Backend (FastAPI)**:
   In your terminal, run the following from the project root:
   ```powershell
   uvicorn backend.main:app --reload
   ```
   *The API will be available at http://localhost:8000*

3. **Start the Frontend (Streamlit)**:
   Open a *second* terminal window and run:
   ```powershell
   streamlit run frontend/app.py
   ```
   *The UI will automatically open in your browser at http://localhost:8501*

## 📚 Documentation
- Architecture & Roadmap: `docs/IMPLEMENTATION.md`
- Sequence & Flow Diagrams: `docs/DIAGRAMS.md`
- User Guide: `docs/USER_GUIDE.md`
- Presentation Script: `docs/DEMO_SCRIPT.md`
