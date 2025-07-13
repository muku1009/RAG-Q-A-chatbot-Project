import streamlit as st
import sys
import os

# Fix module path to import chatbot
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from chatbot.rag_chat import rag_chatbot

# Streamlit page setup
st.set_page_config(
    page_title="📊 Loan Approval RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Custom CSS styling for better UI
st.markdown("""
    <style>
        body {
            background-color: #f8f9fa;
        }
        .stTextInput>div>div>input {
            padding: 10px;
            font-size: 16px;
            border-radius: 8px;
            border: 1px solid #ccc;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            font-size: 16px;
            border: none;
            border-radius: 10px;
            cursor: pointer;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stMarkdown h1 {
            color: #2c3e50;
        }
        .stMarkdown h3 {
            margin-top: 20px;
        }
        .stSuccess {
            background-color: #e8f5e9;
            color: #2e7d32;
        }
    </style>
""", unsafe_allow_html=True)

# App Title & Subtitle
st.title("🤖 Loan Approval Q&A Chatbot")
st.markdown("🔍 **Ask questions about loan approvals. Powered by Retrieval-Augmented Generation (RAG) using FAISS & Gemini Pro.**")

# Input box for user query
query = st.text_input("💬 Ask a question:", placeholder="e.g., Why was my loan rejected?")

# Generate answer on button click
if st.button("🔍 Get Answer") and query:
    with st.spinner("⏳ Thinking..."):
        response = rag_chatbot(query)
        st.success("✅ Answer:")
        st.markdown(f"📝 **{response}**")

# Sample questions for guidance
st.markdown("---")
st.subheader("🎯 Sample Questions You Can Try")
col1, col2 = st.columns(2)

with col1:
    st.markdown("- 📌 What factors lead to loan rejection?")
    st.markdown("- 📌 Who mostly gets their loan approved?")

with col2:
    st.markdown("- 📌 Tell me about applicants from rural areas.")
    st.markdown("- 📌 Do self-employed people get loans easily?")

# Footer
st.markdown("---")
st.markdown("🛠️ *Developed as part of Celebal Technologies Internship – Week 8 Project*")
