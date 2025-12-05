import re
def validate_answer(answer, selected_texts, min_overlap=0.2):
    def ngrams(s, n=3):
        toks = re.findall(r"\w+", s.lower())
        return set(tuple(toks[i:i+n]) for i in range(len(toks)-n+1))
    ans_ngrams = ngrams(answer)
    ctx_ngrams = set()
    for t in selected_texts:
        ctx_ngrams |= ngrams(t)
    overlap = len(ans_ngrams & ctx_ngrams) / (len(ans_ngrams) + 1e-9)
    return overlap >= min_overlap