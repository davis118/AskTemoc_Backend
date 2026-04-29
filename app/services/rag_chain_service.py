"""
RAG chain using OpenAI LLM + pgvector retriever.
"""

import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from app.services.prompt_service import rag_prompt_template
from app.services.retriever_service import retriever_service
from app.core.config import get_settings


def _get_llm():
    settings = get_settings()
    if settings.use_openai:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=0.4,
            api_key=settings.OPENAI_API_KEY,
        )
    # Ollama fallback
    from langchain_community.llms import Ollama
    return Ollama(
        model=settings.OLLAMA_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
    )


class RagChainService:
    def __init__(self):
        self.retriever = retriever_service.get_retriever()
        self.llm = _get_llm()

    def get_chain(self):
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain_from_docs = (
            RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
            | rag_prompt_template
            | self.llm
            | StrOutputParser()
        )

        return RunnableParallel(
            {"context": self.retriever, "question": RunnablePassthrough()}
        ).assign(answer=rag_chain_from_docs)


rag_chain_service = RagChainService()
