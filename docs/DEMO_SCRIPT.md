# GraphRec Presentation Demo Script

**Total Time:** 5-7 Minutes

## 1. Introduction (1 min)
- **Hook**: "Hello everyone! Have you ever wondered how Amazon or Goodreads knows exactly what book you want to read next? Today, I'm excited to present **GraphRec**, a book recommendation engine powered by Graph Neural Networks."
- **Problem**: "Traditional recommendation systems (like matrix factorization) often struggle with sparse data and fail to capture complex, multi-hop relationships between users and items."
- **Solution**: "To solve this, I modeled user-book interactions as a bipartite graph and implemented **LightGCN**—a state-of-the-art Graph Neural Network—entirely from scratch in PyTorch."

## 2. Architecture & Approach (1.5 mins)
*(Show the Architecture Diagram from `docs/DIAGRAMS.md`)*
- "Here is the high-level architecture of GraphRec."
- "First, the **Data Pipeline**: I cleaned the Book-Crossing dataset, filtered out sparse interactions, and constructed a graph where users and books are nodes, and ratings are edges."
- "Next, the **Model**: I built LightGCN. Unlike standard GCNs, LightGCN drops feature transformations and nonlinear activations, making it highly efficient for collaborative filtering. It learns embeddings by propagating them across the graph layers, optimizing via Bayesian Personalized Ranking (BPR) loss."
- "Finally, the **Production System**: I wrapped the trained model in a **FastAPI** backend for real-time inference, and built a modern **Streamlit** frontend for user interaction."

## 3. Live Demonstration (2.5 mins)
*(Switch to the Streamlit UI running locally)*
- "Let's see it in action!"
- **Scenario 1 (Known User)**: 
  - "I'll enter User ID `276964`."
  - *(Click Generate)*
  - "The FastAPI backend receives this ID, instantly computes the dot-product between this user's LightGCN embedding and all book embeddings, and returns the top 10 matches."
  - "As you can see, the UI renders these recommendations beautifully with book covers and metadata."
- **Scenario 2 (Cold Start / Unknown User)**:
  - "What happens if a brand new user joins? Let's enter ID `999999`."
  - *(Click Generate)*
  - "The system detects they aren't in the graph and automatically triggers our **Popularity Fallback** mechanism. It serves an alert and recommends our highest-rated, most popular books. This ensures we never return a blank page."

## 4. Results and Conclusion (1 min)
- "To prove this works, I evaluated LightGCN against a standard SVD Matrix Factorization baseline."
- "LightGCN outperformed SVD significantly on Precision@10, Recall@10, and NDCG@10, proving that capturing high-order graph connectivity yields better personalized recommendations."
- "In the future, I plan to add temporal dynamics to the graph for session-aware recommendations."
- "Thank you for listening! I'm happy to take any questions."
