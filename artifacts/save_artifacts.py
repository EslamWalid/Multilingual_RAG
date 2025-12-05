import faiss
import json
import numpy as np

# Paths
ARTIFACT_DIR = r"multilingual_RAG\artifacts"
FAISS_PATH = ARTIFACT_DIR + r"\faiss_index.bin"
TEXTS_PATH = ARTIFACT_DIR + r"\corpus_texts.json"
META_PATH = ARTIFACT_DIR + r"\corpus_meta.json"
EMB_PATH = ARTIFACT_DIR + r"\corpus_embeddings.npy"
CONFIG_PATH = ARTIFACT_DIR + r"\config.json"

# Ensure directory exists
import os
os.makedirs(ARTIFACT_DIR, exist_ok=True)

config = {
        "model_name": "distiluse-base-multilingual-cased-v1",
        "w_sem": 0.7,
        "w_lex": 0.3,
        "retrieval_threshold": 0.2,
        "grounding_threshold": 0.2,
        "max_ctx_chars": 6000
    }
with open(CONFIG_PATH, "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False)


def save_artifacts(model, name):

    if name == "faiss_index":
        index = model
        faiss.write_index(index, FAISS_PATH)

    elif name == "corpus":
        corpus_texts  = model
        
        with open(TEXTS_PATH, "w", encoding="utf-8") as f:
            json.dump(corpus_texts, f, ensure_ascii=False)

    elif name == "corpus_meta":
        corpus_meta  = model
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(corpus_meta, f, ensure_ascii=False)

    elif name == "corpus_embeddings":
        corpus_embeddings = model
        np.save(EMB_PATH, corpus_embeddings)

    else:
        raise ValueError(f"Unknown artifact name: {name}")
    # 4) Save config (model name, weights, thresholds)
    
