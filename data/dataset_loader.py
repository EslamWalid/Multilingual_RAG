from datasets import load_dataset

def extract_qa(example):
    return {
        "question": example["question"],
        "short_answer": example.get("short_answer", ""),
        "long_answer": example.get("long_answer", "")
    }

def load_dataset_csv(path):
    dataset = load_dataset("csv", data_files=path, split="train")
    return dataset.map(extract_qa)
