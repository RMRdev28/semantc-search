import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import google.generativeai as genai
import os

load_dotenv()

os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = Flask(__name__)

# PDF Processing Functions
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def create_vector_store(pdf_docs):
    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# Streamlit UI for Preprocessing (Optional)
def main_ui():
    st.set_page_config("PDF Preprocessing")
    st.header("Process PDFs for Semantic Search")
    pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
    if st.button("Process PDFs"):
        with st.spinner("Processing..."):
            create_vector_store(pdf_docs)
            st.success("PDFs processed and index created!")

# Load the vector store (this assumes you've already processed the PDFs)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = FAISS.load_local("faiss_index", embeddings)

# API Endpoint
@app.route('/semantic_search', methods=['POST'])
def semantic_search():
    query = request.json.get('query')
    if not query:
        return jsonify({"error": "Query is required"}), 400

    docs_and_scores = vector_store.similarity_search_with_score(query) 

    # Check if any relevant documents were found
    if len(docs_and_scores) > 0:  # Check if the list has any items
        # Sort documents by similarity score (ascending order, where 0 is most similar)
        sorted_docs_and_scores = sorted(docs_and_scores, key=lambda x: x[1])

        # Set a similarity threshold to determine relevance
        similarity_threshold = 0.6

        # Check if the most similar document is below the threshold
        if sorted_docs_and_scores[0][1] <= similarity_threshold:
            return jsonify({
                "exists": True, 
                "passage": sorted_docs_and_scores[0][0].page_content,
                "similarity_score": float(sorted_docs_and_scores[0][1]) 
            }), 200
        else:
            return jsonify({"exists": False}), 200
    else:
        # No documents found at all
        return jsonify({"exists": False}), 200
    

if __name__ == "__main__":
    # Choose one of these to run:
    # main_ui()  # Run Streamlit UI for preprocessing
    app.run(debug=True)  # Run the Flask API (after preprocessing)
