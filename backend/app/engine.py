import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class VectorEngine:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print("Loading Transformer Model...")
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.metadata = None 

    def build_index(self, csv_path):
        df = pd.read_csv(csv_path)
        self.metadata = df[['Ticket Subject', 'Ticket Description', 'Category', 'cleaned_text']].copy()
        
        print(f"Encoding {len(df)} entries...")
        embeddings = self.model.encode(df['cleaned_text'].tolist(), show_progress_bar=True)
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))

    def save_assets(self, folder):
        import os
        os.makedirs(folder, exist_ok=True)
        faiss.write_index(self.index, os.path.join(folder, "tickets.index"))
        self.metadata.to_pickle(os.path.join(folder, "metadata.pkl"))

    def load_assets(self, index_path, meta_path):
        """THIS WAS MISSING: Loads the saved brain from disk"""
        print(f"Loading Index from {index_path}...")
        self.index = faiss.read_index(index_path)
        self.metadata = pd.read_pickle(meta_path)
        print("Assets loaded successfully.")

    def search(self, query, top_k=5):
        """Converts query to vector and finds neighbors"""
        query_vector = self.model.encode([query]).astype('float32')
        distances, indices = self.index.search(query_vector, top_k)
        
        # Get the rows from our metadata
        results = self.metadata.iloc[indices[0]].copy()
        # Add the similarity score (Lower L2 distance = higher similarity)
        results['score'] = distances[0].tolist()
        return results.to_dict(orient='records')