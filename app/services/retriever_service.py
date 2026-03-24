import chromadb
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import os

class RetrieverService:
    def __init__(self, collection_name="asktemoc_collection"):
        self.client = chromadb.PersistentClient(path=os.getenv("CHROMA_PERSIST_DIRECTORY"))
        self.collection_name = collection_name
        self.embeddings = OllamaEmbeddings(model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"))

    def get_retriever(self):
        vector_store = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
        )
        return vector_store.as_retriever()
    
    def search(self, query: str, k: int = 5):
        vector_store = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
        )

        results = vector_store.similarity_search(
            query,
            k=k
        )

        return results

retriever_service = RetrieverService()
