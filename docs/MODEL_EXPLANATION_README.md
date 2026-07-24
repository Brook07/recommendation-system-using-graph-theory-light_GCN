# How GraphRec Works: Training and Recommendations

This document explains the inner workings of the GraphRec system, specifically focusing on how the LightGCN model is trained and how recommendations are generated and displayed to the user.

---

## 1. How is the Model Trained?

The recommendation system uses **LightGCN** (Light Graph Convolutional Network), a state-of-the-art graph-based collaborative filtering model. The training process involves several key steps:

### A. Graph Construction
Instead of treating users and books as separate tables, the system models them as a **Bipartite Graph**:
- **Nodes**: Users and Books.
- **Edges**: An edge exists between a User node and a Book node if the user has rated that book.
- **Adjacency Matrix**: The graph is represented as a mathematical matrix. To ensure stable learning, this matrix is symmetrically normalized (meaning nodes with many connections don't overpower nodes with few connections).

### B. LightGCN Architecture
Unlike standard neural networks with heavy non-linear layers, LightGCN is intentionally simplified for recommendation tasks:
1. **Initial Embeddings**: Every user and book is assigned a random vector (embedding) of a fixed size (e.g., 64 dimensions).
2. **Graph Propagation**: The model updates a node's embedding by aggregating the embeddings of its neighbors. 
   - *Example*: A user's embedding is updated by averaging the embeddings of all the books they have read. A book's embedding is updated by averaging the embeddings of all users who read it.
3. **Layer Combination**: This propagation happens over multiple layers (usually 3). The final embedding for a user or book is the average of its embeddings at every layer (Layer 0 + Layer 1 + Layer 2 + Layer 3).

### C. Training with BPR Loss
The model is trained using **Bayesian Personalized Ranking (BPR) loss**, which is a pairwise learning method:
1. **Positive Sample**: A book the user *has* read.
2. **Negative Sample**: A book the user *has not* read (randomly sampled).
3. **Objective**: The model computes a score (dot product) between the user's embedding and the book's embedding. The loss function forces the model to score the positive book higher than the negative book.
4. **Optimization**: Through backpropagation (using the Adam optimizer), the embeddings are adjusted so that users are brought closer in the vector space to the books they like (and the books similar users like).

---

## 2. How are Recommendations Shown?

Once the model is trained, generating and displaying recommendations involves a smooth pipeline from the backend to the user interface.

### A. The Inference Engine (Backend)
When a recommendation is requested for a specific `user_id`:
1. **Embedding Lookup**: The system looks up the final learned embedding for that user.
2. **Score Calculation**: It calculates the inner product (similarity score) between the user's embedding and the embeddings of all available books.
3. **Ranking**: The books are sorted by their score in descending order, and the books the user has already read are filtered out.
4. **Fallback Mechanism**: If the `user_id` is unknown (a new user not present during training), the system falls back to a **popularity-based recommender**, suggesting the most highly-rated and interacted books across the entire dataset.

### B. The API Layer (FastAPI)
The backend exposes a REST API endpoint (`GET /recommend/{user_id}`). It receives the request, calls the inference engine, and formats the output. It enriches the raw book IDs with rich metadata from the dataset (Title, Author, Publisher, Year, and Cover Image URLs) and sends it as a structured JSON response.

### C. The Frontend Display (Streamlit)
The user interacts with a modern UI built in Streamlit (`frontend/app.py`).
1. **Persona Selection**: The user can pick predefined "personas" (e.g., a "Highly Active Reader" or a "Classic-Era Reader") or enter a custom user ID.
2. **Visual Presentation**: 
   - The UI fetches the JSON response from the API.
   - Books are rendered as visually appealing **Cards** in a grid layout.
   - Each card displays the **Book Cover** (fetched via URL), Title, Author, and tags for Year and Publisher.
3. **Explainability**: Each card includes a badge showing the **Similarity Score** and a short text explaining *why* the book was recommended (e.g., "Top-ranked recommendation for this user's learned reading profile" or "Popularity fallback").
