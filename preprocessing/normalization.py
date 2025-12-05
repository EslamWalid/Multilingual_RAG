from langdetect import detect
import re

STOPWORDS = {
    "en": {"the","a","an","of","to","and"},
    "ar": {"و","في","من"}
}

def normalize_query(q: str):
    q = q.strip().lower()
    q = re.sub(r"\s+", " ", q)
    lang = detect(q)
    toks = q.split()
    if lang in STOPWORDS:
        toks = [t for t in toks if t not in STOPWORDS[lang]]
    return " ".join(toks), lang
