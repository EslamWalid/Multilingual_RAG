from preprocessing.normalization import normalize_query
from preprocessing.contextual_query import build_contextual_query
from retrieval.hybrid import hybrid_retrieve
from retrieval.ranking import score_and_rank
from llm.fallback import answer_with_fallback
from artifacts.load_artifacts import load_artifacts
from fastapi import FastAPI
import uvicorn
app = FastAPI()


faiss_index, corpus_texts, corpus_meta, model = load_artifacts()

def ask(question, history, model, faiss_index, corpus_texts, corpus_meta):
    q_norm, lang = normalize_query(question)
    ctx_query = build_contextual_query(history, q_norm, model)
    merged = hybrid_retrieve(ctx_query, model, faiss_index, corpus_texts, corpus_meta)
    ranked = score_and_rank(merged, corpus_meta, q_norm, lang)
    thresholds = {"retrieval": 0.2, "grounding": 0.2}
    return answer_with_fallback(q_norm, ranked, corpus_texts, thresholds)


@app.post("/ask")
def ask_endpoint(request: dict):
    question = request.get("question", "")
    history = request.get("history", [])
    answer = ask(question, history, model, faiss_index, corpus_texts, corpus_meta)
    return {"answer": answer}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
