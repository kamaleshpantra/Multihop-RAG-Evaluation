from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker


class RecallPreservingMultiHopRetriever:
    """
    Multi-hop retriever that preserves the baseline retrieval
    results while adding second-hop candidates.

    Pipeline:

        Original query
              |
              v
        Baseline hybrid retrieval
              |
              +--------------------+
              |                    |
              v                    v
        Baseline top-k       Title-based expansion
                                   |
                                   v
                            Second-hop retrieval
                                   |
                                   v
                              Merge candidates
                                   |
                                   v
                           Cross-encoder reranker
                                   |
                                   v
                              Final results

    Baseline documents are protected so that adding noisy
    second-hop candidates cannot completely eliminate the
    baseline evidence.
    """

    def __init__(
        self,
        candidate_k=20,
        initial_k=15,
        expansion_k=5,
        expansion_candidate_k=20,
        protected_k=5,
        final_k=15,
        rrf_k=60,
    ):
        self.candidate_k = candidate_k
        self.initial_k = initial_k
        self.expansion_k = expansion_k
        self.expansion_candidate_k = expansion_candidate_k
        self.protected_k = protected_k
        self.final_k = final_k

        self.hybrid_retriever = HybridRetriever(
            rrf_k=rrf_k
        )

        self.reranker = Reranker()

    def _document_key(self, document):
        """
        Create a stable key for document deduplication.
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
        Extract unique titles from the strongest initial
        retrieval results.
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

            normalized = title.lower()

            if normalized in seen:
                continue

            seen.add(normalized)
            titles.append(title)

            if len(titles) >= self.expansion_k:
                break

        return titles

    def retrieve(self, query, k=None):

        if k is None:
            k = self.final_k

        # =====================================================
        # STEP 1: Baseline retrieval
        # =====================================================

        baseline_documents = (
            self.hybrid_retriever.retrieve(
                query,
                k=self.initial_k,
                candidate_k=self.candidate_k,
            )
        )

        if not baseline_documents:
            return []

        # =====================================================
        # STEP 2: Identify intermediate entities
        # =====================================================

        expansion_titles = (
            self._get_expansion_titles(
                baseline_documents
            )
        )

        # =====================================================
        # STEP 3: Collect candidates
        # =====================================================

        candidates = {}

        # Add baseline documents first.
        for document in baseline_documents:

            key = self._document_key(
                document
            )

            candidates[key] = document

        # =====================================================
        # STEP 4: Second-hop retrieval
        # =====================================================

        for title in expansion_titles:

            expanded_query = (
                f"{query} {title}"
            )

            expanded_documents = (
                self.hybrid_retriever.retrieve(
                    expanded_query,
                    k=self.expansion_candidate_k,
                    candidate_k=self.candidate_k,
                )
            )

            for document in expanded_documents:

                key = self._document_key(
                    document
                )

                candidates[key] = document

        candidate_list = list(
            candidates.values()
        )

        if not candidate_list:
            return []

        # =====================================================
        # STEP 5: Cross-encoder reranking
        # =====================================================

        reranked = self.reranker.rerank(
            query,
            candidate_list,
            k=len(candidate_list),
        )

        # =====================================================
        # STEP 6: Protect strongest baseline documents
        # =====================================================

        protected_baseline = baseline_documents[
            :min(
                self.protected_k,
                len(baseline_documents),
            )
        ]

        protected_keys = {
            self._document_key(document)
            for document in protected_baseline
        }

        # Start final results with protected baseline docs.
        final_documents = []

        for document in protected_baseline:

            key = self._document_key(
                document
            )

            if key not in {
                self._document_key(doc)
                for doc in final_documents
            }:

                final_documents.append(
                    document
                )

        # =====================================================
        # STEP 7: Fill remaining slots using reranking
        # =====================================================

        existing_keys = {
            self._document_key(document)
            for document in final_documents
        }

        for document in reranked:

            key = self._document_key(
                document
            )

            if key in existing_keys:
                continue

            final_documents.append(
                document
            )

            existing_keys.add(key)

            if len(final_documents) >= k:
                break

        # =====================================================
        # STEP 8: Safety fallback
        # =====================================================

        if len(final_documents) < k:

            for document in baseline_documents:

                key = self._document_key(
                    document
                )

                if key in existing_keys:
                    continue

                final_documents.append(
                    document
                )

                existing_keys.add(key)

                if len(final_documents) >= k:
                    break

        return final_documents[:k]