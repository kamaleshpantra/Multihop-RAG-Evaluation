from langchain_community.vectorstores import FAISS

from src.indexing.embeddings import get_embedding_model


INDEX_PATH = "vectorstore/faiss"


def load_vectorstore():

    embeddings = get_embedding_model()

    vectorstore = FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore


def retrieve(query: str, k: int = 5):

    vectorstore = load_vectorstore()

    results = vectorstore.similarity_search(
        query,
        k=k
    )

    return results