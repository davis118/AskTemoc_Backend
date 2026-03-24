from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

from app.services.prompt_service import rag_prompt_template
from app.services.retriever_service import retriever_service
import os

class RagChainService:
    def __init__(self):
        self.retriever = retriever_service.get_retriever()
        self.llm = OllamaLLM(model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"), base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))

    def get_chain(self):
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain_from_docs = (
            RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
            | rag_prompt_template
            | self.llm
            | StrOutputParser()
        )

        rag_chain_with_source = RunnableParallel(
            {"context": self.retriever, "question": RunnablePassthrough()}
        ).assign(answer=rag_chain_from_docs)

        return rag_chain_with_source

rag_chain_service = RagChainService()
