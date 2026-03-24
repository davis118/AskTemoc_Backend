from fastapi import APIRouter, Depends
from app.services.rag_service import RAGService
from app.services.llm_service import LLMService
from app.services.embedding_service import EmbeddingsService
from app.models.requests import QueryRequest
from app.models.response import QueryResponse


router = APIRouter()

@router.post("/", response_model=QueryResponse)
async def query(request: QueryRequest):
    llm_service = LLMService()
    embedding_service = EmbeddingsService(
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
            persist_directory="/db/chroma/",
            collection_name="example_collection"
    )
    rag = RAGService(llm_service=llm_service, embedding_service=embedding_service)
    answer = await rag.a_answer(request.query)
    return QueryResponse(answer=answer)

