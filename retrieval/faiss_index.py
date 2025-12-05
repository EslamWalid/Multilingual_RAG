import faiss
from artifacts import save_artifacts
def build_faiss_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    save_artifacts.save_artifacts(index, "faiss_index")
    return index
