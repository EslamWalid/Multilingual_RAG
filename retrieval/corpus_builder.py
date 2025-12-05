import pandas as pd
from artifacts import save_artifacts

def build_corpus():
        
    df = pd.read_csv(r"data/Natural-Questions-Filtered.csv")

    corpus_texts = {}
    corpus_meta = {}

    for idx, row in df.iterrows():
        text = str(row["long_answers"]) if pd.notna(row["long_answers"]) else str(row["short_answers"])
        corpus_texts[idx] = text
        
        q = str(row["question"]).lower()
        if q.startswith(("who","من")):
            qtype = "person"
        elif q.startswith(("where","أين")):
            qtype = "location"
        elif q.startswith(("when","متى")):
            qtype = "time"
        else:
            qtype = "generic"
        
        corpus_meta[idx] = {
            "question_type": qtype,
            "lang": "en"  # could detect dynamically
        }

    save_artifacts.save_artifacts(corpus_texts, "corpus")
    save_artifacts.save_artifacts(corpus_meta, "corpus_meta")