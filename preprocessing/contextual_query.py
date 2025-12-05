import numpy as np
def build_contextual_query(history, user_query, model, k_ctx=2):
    prev_texts = [h["assistant_answer"] for h in history[-5:] if "assistant_answer" in h]
    if not prev_texts:
        return user_query
    prev_embs = model.encode(prev_texts)
    q_emb = model.encode([user_query])
    sims = (prev_embs @ q_emb.T) / (np.linalg.norm(prev_embs, axis=1, keepdims=True) * np.linalg.norm(q_emb))
    top_idx = np.argsort(-sims.squeeze())[:k_ctx]
    ctx = "\n\n".join([prev_texts[i] for i in top_idx])
    return f"Context:\n{ctx}\n\nQuestion:\n{user_query}"