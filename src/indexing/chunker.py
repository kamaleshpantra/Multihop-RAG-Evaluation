from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []

    for document in documents:

        text = document["text"]

        split_texts = splitter.split_text(text)

        search_start = 0

        for text_chunk in split_texts:

            chunk_start = text.find(
                text_chunk,
                search_start
            )

            if chunk_start == -1:
                chunk_start = 0

            chunk_end = chunk_start + len(text_chunk)

            sentence_ids = []

            current_position = 0

            for sentence_id, sentence in enumerate(
                document["sentences"]
            ):

                sentence_start = current_position

                sentence_end = (
                    current_position + len(sentence)
                )

                # Account for the space added by " ".join(...)
                current_position = sentence_end + 1

                if (
                    sentence_end > chunk_start
                    and sentence_start < chunk_end
                ):
                    sentence_ids.append(sentence_id)

            chunks.append(
                {
                    "title": document["title"],
                    "text": text_chunk,
                    "source_id": document["source_id"],
                    "sentence_ids": sentence_ids,
                }
            )

            search_start = chunk_start + 1

    return chunks