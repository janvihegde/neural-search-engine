# 🧠 Neural Semantic Search Engine (RAG-Lite)

A high-performance search system that retrieves information by **semantic meaning** rather than simple keyword matching. 

### 🚀 Key Features
- **Semantic Retrieval:** Uses `all-MiniLM-L6-v2` BERT-based embeddings to understand query intent.
- **Ultra-Fast Indexing:** Powered by **FAISS** (Facebook AI Similarity Search) for sub-50ms retrieval.
- **Modern Stack:** Built with a **FastAPI** microservice and a **React + Tailwind CSS** frontend.

### 🛠️ Tech Stack
- **ML Model:** Sentence-Transformers (HuggingFace)
- **Vector DB:** FAISS
- **Backend:** Python (FastAPI, Uvicorn)
- **Frontend:** React.js, Tailwind CSS, Axios

### 🏁 Quick Start
1. **Prepare Data:** `python -m backend.app.prepare_data`
2. **Build Index:** `python -m backend.app.engine`
3. **Start API:** `python -m uvicorn backend.app.main:app --reload`
4. **Start UI:** `cd frontend && npm start`
