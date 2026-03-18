from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from .engine import VectorEngine
import os

app = FastAPI(title="Neural Search API")

# 1. Enable CORS for React (Running on port 3000 usually)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with your frontend URL
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Global variable to hold our Engine
# We initialize it as None and load it on startup
engine = None

@app.on_event("startup")
async def startup_event():
    global engine
    engine = VectorEngine()
    
    # Absolute paths to your saved assets
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    INDEX_PATH = os.path.join(BASE_DIR, "backend", "vector_storage", "tickets.index")
    META_PATH = os.path.join(BASE_DIR, "backend", "vector_storage", "metadata.pkl")
    
    if os.path.exists(INDEX_PATH):
        engine.load_assets(INDEX_PATH, META_PATH)
    else:
        print("❌ Error: Index files not found. Run engine.py first!")

@app.get("/search")
async def search_tickets(q: str = Query(..., min_length=3)):
    """
    The main search endpoint. 
    Input: ?q=how do I reset my password
    Output: List of JSON objects with text and similarity scores.
    """
    if not engine:
        return {"error": "Search engine not initialized"}
    
    results = engine.search(q, top_k=5)
    return {
        "query": q,
        "count": len(results),
        "results": results
    }

@app.get("/health")
def health_check():
    return {"status": "online", "model": "all-MiniLM-L6-v2"}