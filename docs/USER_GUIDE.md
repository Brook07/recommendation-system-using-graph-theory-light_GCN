# User Guide

This guide explains how to install dependencies and run the GraphRec End-to-End Application locally.

## 1. Installation

Ensure you have Python 3.10+ installed. Install the required dependencies:

```bash
pip install -r requirements.txt
```

*(Note: The `requirements.txt` includes FastAPI, Streamlit, PyTorch, Pandas, and other essential libraries.)*

## 2. Running the Backend Server (FastAPI)

Open a terminal and navigate to the project root directory. Start the FastAPI server using `uvicorn`:

```bash
uvicorn backend.main:app --reload
```

The backend server will start at `http://localhost:8000`. 
- You can access the automatic interactive API documentation at `http://localhost:8000/docs`.
- The server will automatically load the PyTorch LightGCN model into memory upon startup.

## 3. Running the Frontend UI (Streamlit)

Open a **second** terminal window, keeping the backend server running. Navigate to the project root and start the Streamlit application:

```bash
streamlit run frontend/app.py
```

Streamlit will launch your default web browser and navigate to `http://localhost:8501`.

## 4. Usage Instructions

1. Once the Streamlit interface loads, look at the **User Settings** sidebar.
2. Enter a **User ID**.
   - *Example Known User*: `276964` or `277042` (Will use LightGCN personalized embeddings).
   - *Example Unknown User*: `999999` (Will trigger the popularity-based fallback mechanism).
3. Click the **Generate Recommendations** button.
4. The system will query the FastAPI backend and display the Top-10 recommended books in an attractive card grid!

## 5. Troubleshooting

- **Connection Error**: If Streamlit displays a connection error, ensure the FastAPI server is running on port 8000.
- **Model Not Found**: Ensure LightGCN training has produced `data/processed/model_training/models/lightgcn_pyg.pt` and embeddings under `data/processed/model_training/embeddings/`.
