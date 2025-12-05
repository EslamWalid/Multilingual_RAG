from llm.call_qwen import call_qwen
from llm.prompt_builder import build_prompt
from llm.validation import validate_answer

def answer_with_fallback(query, ranked_docs, corpus_texts, thresholds):
    if not ranked_docs or ranked_docs[0][1] < thresholds.get("retrieval", 0.2):
        return "I need more detail. What exactly are you looking to know?"
    print(corpus_texts)
    prompt = build_prompt(query, ranked_docs, corpus_texts)
    ans = call_qwen(prompt)
    selected = [corpus_texts[doc_id] for doc_id, _ in ranked_docs[:3]]
    if not validate_answer(ans, selected, thresholds.get("grounding", 0.2)):
        return "Based on the available context, here’s the best summary:\n\n" + "\n\n".join(selected[:2])
    return ans