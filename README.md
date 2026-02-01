# RAG Templates & Learning Hub 🚀

> **Learning and implementing Retrieval-Augmented Generation using LangChain: exploring document loaders, text processing, vector embeddings, and semantic search for enhanced AI responses.**

A comprehensive collection of copy-paste ready templates and implementations for Retrieval-Augmented Generation (RAG) workflows. Perfect for quickly bootstrapping your RAG projects or learning the fundamentals.

---

## 📋 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Usage Examples](#-usage-examples)
- [Templates](#-templates)
- [Contributing](#-contributing)

---

## ✨ Features

- **Ready-to-use templates** for common RAG workflows
- **Document loaders** for PDFs, text files, and more
- **Text processing** and chunking strategies
- **Vector embeddings** with multiple backend support
- **Semantic search** implementations
- **ChromaDB & FAISS** integration examples
- **Well-documented Jupyter notebooks**
- **Production-ready code patterns**

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/RAG.git
cd RAG

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate RAG

# Launch Jupyter
jupyter notebook
```

---

## 📦 Installation

### Prerequisites
- [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Python 3.11+

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/RAG.git
   cd RAG
   ```

2. **Create environment from file**
   ```bash
   conda env create -f environment.yml
   ```

3. **Activate the environment**
   ```bash
   conda activate RAG
   ```

4. **Verify installation**
   ```bash
   python -c "import langchain, chromadb, faiss; print('All packages imported successfully!')"
   ```

### Manual Installation (Alternative)

```bash
conda create -n RAG python=3.11 -y
conda activate RAG
conda install -c conda-forge langchain langchain-core langchain-community \
    pypdf tqdm sentence-transformers faiss-cpu chromadb -y
pip install pymupdf
```

---

## 📁 Project Structure

```
RAG/
├── data/
│   ├── pdf_files/          # PDF documents for processing
│   └── text_files/         # Text documents
│       └── sample_text.txt
├── notebook/
│   ├── document.ipynb      # Document loading examples
│   └── embedding.ipynb     # Embedding and vector search
├── environment.yml         # Conda environment specification
├── requirements.txt        # Package list
├── .gitignore
└── README.md
```

---

## 💡 Usage Examples

### 1. Loading Documents

```python
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# Load all PDFs from a directory
loader = DirectoryLoader(
    "../data/pdf_files",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader,
    show_progress=True
)
documents = loader.load()
```

### 2. Text Splitting

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)
```

### 3. Creating Embeddings

```python
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

### 4. Vector Store (ChromaDB)

```python
from langchain_community.vectorstores import Chroma

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
```

### 5. Semantic Search

```python
# Similarity search
results = vectorstore.similarity_search(
    "Your query here",
    k=3
)

for doc in results:
    print(doc.page_content)
```

---

## 📝 Templates

### Basic RAG Pipeline

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Load documents
loader = PyPDFLoader("your_document.pdf")
documents = loader.load()

# 2. Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)

# 3. Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Create vector store
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)

# 5. Query
query = "What is the main topic?"
results = vectorstore.similarity_search(query, k=3)
```

### FAISS Vector Store

```python
from langchain_community.vectorstores import FAISS

# Create FAISS index
vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

# Save index
vectorstore.save_local("faiss_index")

# Load index
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)
```

---

## 🛠️ Key Dependencies

- **LangChain** - Framework for LLM applications
- **ChromaDB** - Vector database for embeddings
- **FAISS** - Facebook AI Similarity Search
- **Sentence Transformers** - State-of-the-art sentence embeddings
- **PyPDF/PyMuPDF** - PDF processing
- **HuggingFace** - Pre-trained models

---

## 📚 Notebooks

| Notebook | Description |
|----------|-------------|
| `document.ipynb` | Document loading, parsing, and preprocessing |
| `embedding.ipynb` | Creating embeddings and vector stores |

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

- Add new templates
- Improve documentation
- Report bugs
- Suggest features

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🌟 Acknowledgments

- LangChain community
- HuggingFace team
- All contributors

---

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Happy RAG building! 🎉**