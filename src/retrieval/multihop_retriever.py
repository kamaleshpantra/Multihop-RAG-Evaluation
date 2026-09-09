from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker


class MultiHopRetriever:
    """
    Iterative multi-hop retriever.

    Stage 1:
        Retrieve documents using the original question.

    Stage 2:
        Use titles from the initial retrieval results as
        intermediate entities and perform expanded retrieval.

    Final:
        Merge all candidates and rerank them with the
        cross-encoder.
    """

    def __init__(
        self,
        candidate_k=20,
        initial_k=10,
        expansion_k=5,
        final_k=15,
        rrf_k=60,
    ):
        self.candidate_k = candidate_k
        self.initial_k = initial_k
        self.expansion_k = expansion_k
        self.final_k = final_k

        self.hybrid_retriever = HybridRetriever(
            rrf_k=rrf_k
        )

        self.reranker = Reranker()

    def _document_key(self, document):
        """
        Create a stable key for deduplicating documents.
        """

        title = document.metadata.get(
            "title",
            "",
        )

        source_id = document.metadata.get(
            "source_id",
            "",
        )

        text = document.page_content

        return (
            title,
            source_id,
            text,
        )

    def _get_expansion_titles(self, documents):
        """
        Extract unique article titles from the initial
        retrieval results.

        We only use the first few documents because using
        every retrieved document would introduce too many
        noisy expansion queries.
        """

        titles = []

        seen = set()

        for document in documents:

            title = document.metadata.get(
                "title",
                "",
            ).strip()

            if not title:
                continue

            normalized_title = title.lower()

            if normalized_title in seen:
                continue

            seen.add(normalized_title)

            titles.append(title)

            if len(titles) >= self.expansion_k:
                break

        return titles

    def retrieve(
        self,
        query,
        k=None,
    ):
        """
        Perform two-stage multi-hop retrieval.

        Parameters
        ----------
        query : str
            Original user question.

        k : int
            Number of final documents returned.
        """

        if k is None:
            k = self.final_k

        # ----------------------------------------------------
        # Stage 1: initial retrieval
        # ----------------------------------------------------

        initial_documents = (
            self.hybrid_retriever.retrieve(
                query,
                k=self.initial_k,
                candidate_k=self.candidate_k,
            )
        )

        # ----------------------------------------------------
        # Extract intermediate entities
        # ----------------------------------------------------

        expansion_titles = (
            self._get_expansion_titles(
                initial_documents
            )
        )

        # ----------------------------------------------------
        # Store all candidates
        # ----------------------------------------------------

        candidate_documents = {}

        # Add initial documents
        for document in initial_documents:

            key = self._document_key(
                document
            )

            candidate_documents[key] = document

        # ----------------------------------------------------
        # Stage 2: expanded retrieval
        # ----------------------------------------------------

        for title in expansion_titles:

            expanded_query = (
                f"{query} {title}"
            )

            expanded_documents = (
                self.hybrid_retriever.retrieve(
                    expanded_query,
                    k=self.candidate_k,
                    candidate_k=self.candidate_k,
                )
            )

            for document in expanded_documents:

                key = self._document_key(
                    document
                )

                candidate_documents[key] = document

        # ----------------------------------------------------
        # Convert candidates to list
        # ----------------------------------------------------

        candidates = list(
            candidate_documents.values()
        )

        if not candidates:
            return []

        # ----------------------------------------------------
        # Final cross-encoder reranking
        # ----------------------------------------------------

        reranked_documents = (
            self.reranker.rerank(
                query,
                candidates,
                k=k,
            )
        )

        return reranked_documents