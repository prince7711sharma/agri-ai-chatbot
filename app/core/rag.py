# app/core/rag.py

import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from app.core.config import TOP_K, MODEL_NAME
import os

model = None
index = None
chunks = None

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_db")

def load_rag_resources():
    global model, index, chunks

    model = SentenceTransformer(MODEL_NAME)

    index = faiss.read_index(
        os.path.join(VECTOR_DB_PATH, "agri_faiss.index")
    )

    with open(os.path.join(VECTOR_DB_PATH, "agri_chunks.pkl"), "rb") as f:
        chunks = pickle.load(f)

    print("✅ RAG resources loaded successfully")

def retrieve_context(query: str):
    if model is None or index is None or chunks is None:
        raise RuntimeError("RAG resources not loaded")

    query_vector = model.encode([query])
    distances, indices = index.search(np.array(query_vector), TOP_K)

    best_distance = distances[0][0]
    context = "\n".join([chunks[i] for i in indices[0]])

    return best_distance, context
