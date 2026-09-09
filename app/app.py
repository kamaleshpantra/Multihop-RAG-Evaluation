import sys
from pathlib import Path

import streamlit as st


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="MultiHop RAG Evaluation",
    page_icon="🔎",
    layout="wide",
)


# ============================================================
# IMPORTS
# ============================================================

from src.generation.rag_pipeline import RAGPipeline


# ============================================================
# LOAD RAG PIPELINE
# ============================================================

@st.cache_resource(show_spinner="Loading RAG pipeline... Please wait.")
def load_pipeline():

    pipeline = RAGPipeline(
        candidate_k=20,
        retrieval_k=15,
        initial_k=15,
        expansion_k=5,
        expansion_candidate_k=20,
        protected_k=5,
    )

    return pipeline


# ============================================================
# HEADER
# ============================================================

st.title("🔎 MultiHop RAG Evaluation")

st.markdown(
    """
A **multi-hop Retrieval-Augmented Generation system** built on
HotpotQA.

The system combines dense retrieval, BM25, reciprocal rank fusion,
multi-hop query expansion, cross-encoder reranking, and LLM
generation.
"""
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Retrieval Pipeline")

    st.markdown(
        """
**Dense Retrieval**

FAISS + all-MiniLM-L6-v2

**Sparse Retrieval**

BM25

**Fusion**

Reciprocal Rank Fusion (RRF)

**Multi-Hop**

Top-5 retrieved source titles

**Reranker**

MS MARCO MiniLM-L6-v2
"""
    )

    st.divider()

    st.markdown("### Final Configuration")

    st.write("Candidate K: **20**")
    st.write("Initial K: **15**")
    st.write("Expansion K: **5**")
    st.write("Protected K: **5**")
    st.write("Final K: **15**")


# ============================================================
# LOAD PIPELINE
# ============================================================

try:

    pipeline = load_pipeline()

except Exception as e:

    st.error("Failed to load the RAG pipeline.")

    st.exception(e)

    st.stop()


# ============================================================
# QUESTION
# ============================================================

st.subheader("Ask a Question")

question = st.text_area(
    "Question",
    placeholder=(
        "Were Scott Derrickson and Ed Wood of the same nationality?"
    ),
    height=100,
)


# ============================================================
# ASK BUTTON
# ============================================================

if st.button(
    "🔍 Ask Question",
    type="primary",
    use_container_width=True,
):

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner(
            "Retrieving evidence and generating answer..."
        ):

            try:

                result = pipeline.answer(
                    question.strip()
                )

            except Exception as e:

                st.error(
                    "An error occurred while answering the question."
                )

                st.exception(e)

                st.stop()


        # ========================================================
        # ANSWER
        # ========================================================

        st.subheader("Answer")

        st.success(result["answer"])


        # ========================================================
        # RETRIEVED DOCUMENTS
        # ========================================================

        documents = result["documents"]

        st.subheader("Retrieved Evidence")

        st.write(
            f"Retrieved **{len(documents)}** evidence chunks."
        )


        # ========================================================
        # DISPLAY EVIDENCE
        # ========================================================

        for i, document in enumerate(
            documents,
            start=1,
        ):

            metadata = document.metadata

            title = metadata.get(
                "title",
                "Unknown Source",
            )

            reranker_score = metadata.get(
                "reranker_score",
                None,
            )

            sentence_ids = metadata.get(
                "sentence_ids",
                [],
            )

            with st.expander(
                f"Evidence {i} — {title}"
            ):

                st.markdown(
                    f"**Source:** {title}"
                )

                st.write(
                    document.page_content
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.caption(
                        "Sentence IDs"
                    )

                    st.write(
                        sentence_ids
                    )

                with col2:

                    if reranker_score is not None:

                        st.caption(
                            "Reranker Score"
                        )

                        st.write(
                            f"{reranker_score:.4f}"
                        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "MultiHop-RAG-Evaluation | HotpotQA | "
    "Recall-Preserving Multi-Hop Retrieval"
)