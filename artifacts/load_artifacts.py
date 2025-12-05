import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# Paths (same as above)
ARTIFACT_DIR = r"multilingual_RAG\artifacts"
FAISS_PATH = ARTIFACT_DIR + r"\faiss_index.bin"
TEXTS_PATH = ARTIFACT_DIR + r"\corpus_texts.json"
META_PATH = ARTIFACT_DIR + r"\corpus_meta.json"
EMB_PATH = ARTIFACT_DIR + r"\corpus_embeddings.npy"
CONFIG_PATH = ARTIFACT_DIR + r"\config.json"

def load_artifacts():
    # 1) Load FAISS index
    index = faiss.read_index(FAISS_PATH)

    # 2) Load corpus texts and metadata
    with open(TEXTS_PATH, "r", encoding="utf-8") as f:
        corpus_texts = json.load(f)

    with open(META_PATH, "r", encoding="utf-8") as f:
        corpus_meta = json.load(f)

    # 3) Load embeddings (optional)
    if os.path.exists(EMB_PATH):
        corpus_embeddings = np.load(EMB_PATH)

    # 4) Load config and re-instantiate model
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    model = SentenceTransformer(config["model_name"])

    return index, corpus_texts, corpus_meta, model
