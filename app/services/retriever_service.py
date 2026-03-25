import chromadb
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from app.core.config import get_settings
import os

class RetrieverService:
    def __init__(self, collection_name="asktemoc_collection"):
        settings = get_settings()

        self.client = chromadb.PersistentClient(
            path=str(settings.chroma_persist_path)
        )

        self.collection_name = (
            collection_name or settings.CHROMA_COLLECTION_NAME
        )

        self.embeddings = OllamaEmbeddings(
            model=settings.OLLAMA_EMBEDDING_MODEL
        )

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
    
    async def a_search(self, query: str, k: int = 5):
        vector_store = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
        )
        
        results = await vector_store.asimilarity_search(query, k = k)
        
        return results
        

retriever_service = RetrieverService()
