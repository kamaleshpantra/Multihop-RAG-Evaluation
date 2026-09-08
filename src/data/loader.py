import json
from pathlib import Path


def load_hotpotqa(path: str):

    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def extract_context(data):

    documents = []

    seen = set()

    for example in data:

        for title, sentences in example["context"]:

            text = " ".join(sentences)

            document_key = (title, text)

            if document_key in seen:
                continue

            seen.add(document_key)

            documents.append(
                {
                    "title": title,
                    "text": text,
                    "sentences": sentences,
                    "source_id": example["_id"],
                }
            )

    return documents