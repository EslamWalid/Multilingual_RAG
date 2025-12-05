def hybrid_retrieve(query, model, faiss_index, corpus_texts, corpus_meta, k_sem=20):
    q_vec = model.encode([query])
    D, I = faiss_index.search(q_vec, k_sem)
    merged = {}
    for i, d in zip(I[0], D[0]):
        if i != -1:
            # Add lexical overlap
            i = int(i)
            lex_score = sum(1 for w in query.split() if w in corpus_texts[i].lower())
            merged[int(i)] = {"sem": float(d), "lex": float(lex_score)}
    return merged