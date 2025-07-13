# chatbot/rag_chat.py

import os
import pickle
import streamlit as st

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

try:
    import faiss  # for Linux/macOS
except ImportError:
    import faiss_cpu as faiss  # for Windows/pip install

# Load environment variables from .env
load_dotenv()

# Load and check Gemini API key
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets["api_keys"]["GOOGLE_API_KEY"]
if not GEMINI_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found in environment variables or .env file.")

# Configure Gemini with your API key
genai.configure(api_key=GEMINI_API_KEY)

# Load Gemini model
try:
    model = genai.GenerativeModel("gemini-1.5-flash")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load Gemini model: {e}")

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index
faiss_index_path = "RAG-Loan-Chatbot/embeddings/faiss_index.index"
metadata_path = "RAG-Loan-Chatbot/embeddings/metadata.pkl"

if not os.path.exists(faiss_index_path) or not os.path.exists(metadata_path):
    raise FileNotFoundError("❌ FAISS index or metadata file not found. Make sure to run `build_vector_store.py` first.")

faiss_index = faiss.read_index(faiss_index_path)

with open(metadata_path, "rb") as f:
    text_chunks = pickle.load(f)

# --- Retrieval Function ---
def retrieve_chunks(query, top_k=5):
    query_embedding = embedding_model.encode([query])
    distances, indices = faiss_index.search(query_embedding, top_k)
    return [text_chunks[i] for i in indices[0]]

# --- Answer Generation ---
def generate_answer(query, context_chunks):
    context = "\n\n".join(context_chunks)

    prompt = f"""
You are a helpful assistant for analyzing loan application data.
Use the following context to answer the user's query.

Context:
{context}

Question: {query}

Answer:
"""

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Error generating response: {e}"

# --- Main RAG Function ---
def rag_chatbot(query):
    context_chunks = retrieve_chunks(query)
    return generate_answer(query, context_chunks)

# --- Run Locally ---
if __name__ == "__main__":
    sample_question = "What are common reasons a loan application is rejected?"
    print("🤖", rag_chatbot(sample_question))
