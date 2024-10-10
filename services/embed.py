import uuid
import logging

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

logger = logging.getLogger("EMBED")


# https://sbert.net/
class Embed:
    def __init__(self):
        self.hf_embed = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": False}
        )

        self.vector_store = Chroma(
            collection_name="documents_collection",
            embedding_function=self.hf_embed,
            persist_directory="../database",
        )

    def save_in_vectorstore(self, documents):
        try:
            uuids = [str(uuid.uuid4()) for _ in range(len(documents))]
            self.vector_store.add_documents(documents=documents, ids=uuids)
            logger.info("Docs add into VectorStore")
        except Exception as e:
            logger.error(f"Embed error: {e}")

    def search_in_vectorstore(self, query, num_docs=3):
        try:
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=num_docs,
            )
            return results
        except Exception as e:
            logger.error(f"Search error: {e}")