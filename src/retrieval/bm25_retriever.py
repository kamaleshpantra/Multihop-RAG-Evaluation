import json
import re

import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


CHUNKS_PATH = "data/processed/chunks.json"


def tokenize(text: str):
    """
    Convert text into lowercase word tokens.

    Example:
        "Ed Wood was American."
    becomes:
        ["ed", "wood", "was", "american"]
    """
    return re.findall(r"\b\w+\b", text.lower())


def load_chunks():
    """
    Load the same processed chunks that were used
    to build the FAISS dense index.
    """
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return chunks


class BM25Retriever:

    def __init__(self):
        print("Loading chunks...")

        self.chunks = load_chunks()

        print(f"Loaded {len(self.chunks)} chunks")

        print("Tokenizing corpus...")

        self.tokenized_corpus = [
            tokenize(chunk["text"])
            for chunk in self.chunks
        ]

        print("Building BM25 index...")

        self.bm25 = BM25Okapi(
            self.tokenized_corpus
        )

        print("BM25 index ready.")

    def retrieve(self, query: str, k: int = 5):

        tokenized_query = tokenize(query)

        scores = self.bm25.get_scores(
            tokenized_query
        )

        top_indices = np.argsort(scores)[::-1][:k]

        documents = []

        for index in top_indices:

            chunk = self.chunks[index]

            document = Document(
                page_content=chunk["text"],
                metadata={
                    "title": chunk["title"],
                    "source_id": chunk["source_id"],
                    "sentence_ids": chunk["sentence_ids"],
                    "bm25_score": float(scores[index]),
                },
            )

            documents.append(document)

        return documents