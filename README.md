# 🔍 RAG (Retrieval-Augmented Generation)

A collection of reusable templates and implementations for building RAG systems. This repository provides ready-to-use code patterns for document processing, vector embeddings, and semantic search that can be easily adapted to your specific use case.

## 📚 Background

RAG enhances Large Language Models by retrieving relevant context from external documents before generating responses. This addresses LLM limitations like hallucinations and outdated knowledge.

**How RAG works:**
1. 📄 Convert documents into vector embeddings
2. 💾 Store vectors in a database (ChromaDB/FAISS)
3. 🔎 Retrieve relevant chunks via similarity search when queried
4. 🤖 Provide retrieved context to the LLM for response generation

## 📦 What's Inside

This repo contains modular, copy-paste ready templates for:

- **📑 Document Loading**: Load PDFs, text files, and other formats
- **✂️ Text Chunking**: Split documents with configurable chunk sizes and overlap strategies
- **🧬 Embedding Generation**: Create vector embeddings using Sentence Transformers
- **🗄️ Vector Storage**: Store and persist embeddings in ChromaDB or FAISS
- **🎯 Semantic Search**: Query and retrieve relevant document chunks

Each component is designed to be independent and reusable - take what you need and adapt it to your project.

## ⚙️ Setup

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

## 🚀 How to Use

1. **Explore the notebooks** in the `notebook/` directory to see working examples
2. **Copy the templates** that match your needs
3. **Customize parameters** like chunk size, embedding model, or vector store
4. **Integrate into your project** - each template is self-contained and modular

Common customization points:
- Chunk size and overlap for your document type
- Embedding model based on speed/quality trade-off
- Vector store choice (ChromaDB for persistence, FAISS for speed)
- Number of retrieved results (top-k)