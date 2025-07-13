import os
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
import pickle

def row_to_text(row):
    return f"""
    📌 Loan ID: {row.get('Loan_ID', 'N/A')}.  
    👤 Gender: {row.get('Gender', 'Unknown')}, Married: {row.get('Married', 'Unknown')}, Dependents: {row.get('Dependents', 'Unknown')}.  
    🎓 Education: {row.get('Education', 'Unknown')}, Self-Employed: {row.get('Self_Employed', 'Unknown')}.  
    💰 Applicant Income: {row.get('ApplicantIncome', 'N/A')}, Coapplicant Income: {row.get('CoapplicantIncome', 'N/A')}.  
    🏦 Loan Amount: {row.get('LoanAmount', 'N/A')}k over {row.get('Loan_Amount_Term', 'N/A')} months.  
    ✅ Credit History: {'Meets guidelines' if row.get('Credit_History') == 1.0 else 'Does not meet guidelines'}.  
    📍 Property Area: {row.get('Property_Area', 'Unknown')}.  
    📝 Loan Status: {'Approved ✅' if row.get('Loan_Status') == 'Y' else 'Rejected ❌'}
    """.strip()

def generate_text_chunks(csv_path, save_path):
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    for col in df.select_dtypes(include='object').columns:
        df.fillna({col: "Unknown"}, inplace=True)
    for col in df.select_dtypes(include='number').columns:
        df.fillna({col: -1}, inplace=True)

    print(f"Loaded dataset with {len(df)} rows")

    text_chunks = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        text_chunks.append(row_to_text(row))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(text_chunks))

    print(f"Saved {len(text_chunks)} text chunks to {save_path}")

def embed_and_store(text_file_path, faiss_index_path, metadata_path):
    if not os.path.exists(text_file_path):
        print(f"❌ Text file not found: {text_file_path}")
        return

    with open(text_file_path, "r", encoding="utf-8") as f:
        chunks = f.read().split("\n\n")

    print(f"✅ Loaded {len(chunks)} chunks for embedding")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, show_progress_bar=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    os.makedirs(os.path.dirname(faiss_index_path), exist_ok=True)
    faiss.write_index(index, faiss_index_path)

    with open(metadata_path, "wb") as f:
        pickle.dump(chunks, f)

    print(f"✅ Saved FAISS index to {faiss_index_path}")
    print(f"✅ Saved metadata to {metadata_path}")

if __name__ == "__main__":
    # Paths adjusted based on current script location inside retriever/
    base_dir = os.path.dirname(os.path.dirname(__file__))  # Goes up to RAG-Loan-Chatbot

    csv_path = os.path.join(base_dir, "data", "Training Dataset.csv")
    text_path = os.path.join(base_dir, "data", "text_chunks.txt")
    faiss_path = os.path.join(base_dir, "embeddings", "faiss_index.index")
    metadata_path = os.path.join(base_dir, "embeddings", "metadata.pkl")

    generate_text_chunks(csv_path, text_path)
    embed_and_store(text_path, faiss_path, metadata_path)
