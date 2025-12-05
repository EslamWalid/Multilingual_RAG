def score_and_rank(merged, corpus_meta, query, lang, w_sem=0.7, w_lex=0.3):
    def qtype(q):
        if q.startswith(("who","من")): return "person"
        if q.startswith(("where","أين")): return "location"
        if q.startswith(("when","متى")): return "time"
        return "generic"

    qt = qtype(query)
    scored = []
    for doc_id, comp in merged.items():
        if doc_id not in corpus_meta:
            continue
        meta = corpus_meta[doc_id]
        base = w_sem * comp["sem"] + w_lex * comp["lex"]
        if meta.get("question_type") == qt: base *= 1.1
        if meta.get("lang") == lang: base *= 1.05
        scored.append((doc_id, base))
    scored.sort(key=lambda x: -x[1])
    return scored