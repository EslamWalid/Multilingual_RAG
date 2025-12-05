# Multilingual RAG Pipeline 🚀

A modular **Retrieval-Augmented Generation (RAG)** system that supports multilingual queries.  
It uses **FAISS** for semantic search, **SentenceTransformers** for embeddings, and integrates with **Qwen LLM** via Ollama.  
The project is wrapped with a **FastAPI** service so you can query it over HTTP.

---

## 📂 Project Structure

'''
multilingual_rag/
├── main.py                # FastAPI entry point (ask function + endpoints)
├── config.py              # Configs, paths, thresholds
├── requirements.txt       # Dependencies
├── data/
│   └── dataset_loader.py  # Load dataset, extract QA pairs
├── preprocessing/
│   ├── chunking.py        # Chunking function
│   ├── normalization.py   # Query normalization
│   └── contextual_query.py # Build contextual query
├── retrieval/
│   ├── corpus_builder.py  # Build corpus texts & metadata
│   ├── faiss_index.py     # Build FAISS index
│   ├── hybrid.py          # Hybrid retrieval
│   └── ranking.py         # Scoring & ranking
├── llm/
│   ├── call_qwen.py       # LLM call wrapper
│   ├── prompt_builder.py  # Build prompt
│   ├── validation.py      # Answer validation
│   └── fallback.py        # Answer with fallback
└── artifacts/
    ├── save_artifacts.py  # Save FAISS, corpus, embeddings, config
    └── load_artifacts.py  # Load FAISS, corpus, embeddings, config




---

## ⚙️ Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/EslamWalid/multilingual_rag.git
   cd multilingual_rag

2. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate   # Linux/Mac
    venv\Scripts\activate      # Windows


3. Install dependencies:
    ```bash
    pip install -r requirements.txt

---
## ▶️ Running the FastAPI App


```bash
uvicorn main:app --reload







