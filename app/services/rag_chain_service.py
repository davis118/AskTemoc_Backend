from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

from app.services.prompt_service import rag_prompt_template
from app.services.retriever_service import retriever_service
import os

class RagChainService:
    def __init__(self):
        self.retriever = retriever_service.get_retriever()
        self.llm = Ollama(model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"), base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))

    def get_chain(self):
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # We keep raw documents unmodified and preserve all metadata
        # Format only a copy for the LLM prompt while keeping originals intact
        rag_chain_from_docs = (
            RunnablePassthrough.assign(formatted_context=(lambda x: format_docs(x["context"])))
            | RunnablePassthrough.assign(prompt_input=lambda x: {"context": x["formatted_context"], "question": x["question"]})
            | (lambda x: x["prompt_input"])
            | rag_prompt_template
            | self.llm
            | StrOutputParser()
        )

        # Return full original Document objects with complete metadata
        # Context field maintains raw documents exactly as retrieved
        rag_chain_with_source = RunnableParallel(
            {"context": self.retriever, "question": RunnablePassthrough()}
        ).assign(answer=rag_chain_from_docs)

        return rag_chain_with_source

rag_chain_service = RagChainService()
