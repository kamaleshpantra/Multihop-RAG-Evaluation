import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from src.indexing.embeddings import get_embedding_model


CHUNKS_PATH = "data/processed/chunks.json"
INDEX_PATH = "vectorstore/faiss"


def load_chunks():

    with open(
        CHUNKS_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        chunks = json.load(f)

    return chunks


def build_index():

    print("Loading chunks...")

    chunks = load_chunks()

    print(
        f"Loaded {len(chunks):,} chunks."
    )

    documents = []

    for chunk in chunks:

        document = Document(
            page_content=chunk["text"],

            metadata={
                "title": chunk["title"],
                "source_id": chunk["source_id"],
                "sentence_ids": chunk["sentence_ids"],
            }
        )

        documents.append(document)

    print(
        f"Created {len(documents):,} LangChain documents."
    )

    print("Loading embedding model...")

    embeddings = get_embedding_model()

    print("Building FAISS index...")

    vectorstore = FAISS.from_documents(
        documents,
        embeddings
    )

    Path(INDEX_PATH).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    vectorstore.save_local(
        INDEX_PATH
    )

    print(
        f"FAISS index saved to: {INDEX_PATH}"
    )


if __name__ == "__main__":

    build_index()