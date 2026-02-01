# RAG (Retrieval-Augmented Generation)

This repository explores building RAG systems that enhance Large Language Models by retrieving relevant context from external documents. RAG addresses LLM limitations like hallucinations and outdated knowledge by grounding responses in actual source documents.

## Background

RAG works by:
1. Converting documents into vector embeddings
2. Storing vectors in a database (ChromaDB/FAISS)
3. Retrieving relevant chunks via similarity search when queried
4. Providing retrieved context to the LLM for response generation

This approach enables dynamic knowledge updates without retraining models and provides source attribution for generated responses.

## Setup

**Prerequisites**: Python 3.8+

**Using Conda:**
```bash
conda env create -f environment.yml
conda activate RAG
```

**Using pip:**
```bash
pip install -r requirements.txt
```