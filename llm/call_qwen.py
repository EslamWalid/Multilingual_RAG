import ollama

def call_qwen(prompt, temperature=0.2, max_tokens=512):
    response = ollama.chat(
        model="qwen2.5:3b-instruct",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature, "num_predict": max_tokens}
    )
    return response["message"]["content"].strip()
