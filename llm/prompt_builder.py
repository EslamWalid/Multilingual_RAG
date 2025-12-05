def build_prompt(query, ranked_docs, corpus_texts, max_ctx_chars=6000, language="en"):
    context_parts, total = [], 0
    for doc_id, _ in ranked_docs[:10]:
        txt = corpus_texts[doc_id]
        if total + len(txt) > max_ctx_chars: break
        context_parts.append(f"[Doc {doc_id}]\n{txt}")
        total += len(txt)
    context_block = "\n\n".join(context_parts)
    return f"""You are a helpful assistant. Answer using only the provided context.

Language: {language}

Question:
{query}

Context:
{context_block}

Requirements:
- Be concise but complete.
- If insufficient context, say so briefly.
- Preserve multilingual terms as appropriate."""