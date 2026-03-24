from typing import List
from app.services.llm_service import LLMService
from app.services.embedding_service import EmbeddingsService
from app.models.requests import QueryRequest
from langchain_core.documents import Document


class RAGService:
    def __init__(
        self, llm_service: LLMService, embedding_service: EmbeddingsService, top_k: int = 5):
        self.llm_service = llm_service
        self.embedding_service = embedding_service
        self.top_k = top_k

    def build_augmented_prompt(self, user_query: str, context_docs: List[Document]) -> str:
        formatted_context = [
            f"Document {i}:\n{doc.page_content}"
            for i, doc in enumerate(context_docs, start=1)
        ]

        context_str = "\n\n".join(formatted_context)

        template = f"""
            You are an AI assistant that answers questions using the provided context.
            If the answer is not contained in the context, say you don't know.

            User Query:
            {user_query}

            Relevant Context:
            {context_str}

            Answer:
        """
        return template.strip()

    def answer(self, query: str) -> str:
        query_request = QueryRequest(query=query, top_k=self.top_k)
        context = self.embedding_service.search(query_request)

        prompt = self.build_augmented_prompt(query, context)
        return self.llm_service.call(prompt)

    async def a_answer(self, query: str) -> str:
        query_request = QueryRequest(query=query, top_k=self.top_k)
        context = self.embedding_service.search(query_request)

        prompt = self.build_augmented_prompt(query, context)
        return await self.llm_service.a_call(prompt)

    async def test_llm(self, query: str) -> str:
        return await self.llm_service.a_call(query)
