from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from app.models.requests import EmbedBatch, QueryRequest

class EmbeddingsService:
    def __init__(self, embedding_model_name: str, persist_directory: str, collection_name: str):
        """Initialize the embedding model and Chroma vector store."""
        self.embedding_model = OllamaEmbeddings(model=embedding_model_name)
        
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embedding_model,
            persist_directory=persist_directory,
        )

    def embed_batch(self, req: EmbedBatch):
        """Store text chunks as vectors in Chroma."""
        texts = [item.text for item in req.items]
        ids = [item.chunk_id for item in req.items]
        metadatas = [item.metadata for item in req.items]
        self.vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        return {"status": "ok", "count": len(texts)}
    
    def search(self, req: QueryRequest):
        """Search for the most similar chunks to a given query."""
        results = self.vector_store.similarity_search(req.query, req.top_k)
        return results
    
    async def a_embed_batch(self, req: EmbedBatch):
        """Store text chunks as vectors in Chroma."""
        texts = [item.text for item in req.items]
        ids = [item.chunk_id for item in req.items]
        metadatas = [item.metadata for item in req.items]
        await self.vector_store.aadd_texts(texts=texts, metadatas=metadatas, ids=ids)
        return {"status": "ok", "count": len(texts)}
    
    async def a_search(self, req: QueryRequest):
        """Search for the most similar chunks to a given query."""
        results = await self.vector_store.asimilarity_search(req.query, req.top_k)
        return results