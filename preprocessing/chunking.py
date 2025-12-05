from nltk.tokenize import sent_tokenize

def chunk_text(text, max_tokens=100):
    sentences = sent_tokenize(text)
    chunks, chunk, token_count = [], [], 0
    for sent in sentences:
        tokens = sent.split()
        if token_count + len(tokens) > max_tokens:
            chunks.append(" ".join(chunk))
            chunk, token_count = [], 0
        chunk.append(sent)
        token_count += len(tokens)
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks
