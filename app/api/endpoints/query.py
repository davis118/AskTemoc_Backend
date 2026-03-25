from fastapi import APIRouter, Depends
from app.services.rag_service import RAGService
from app.services.llm_service import LLMService
from app.services.search_service import SearchService
from app.models.requests import QueryRequest
from app.models.response import QueryResponse


router = APIRouter()

@router.post("/", response_model=QueryResponse)
async def query(request: QueryRequest):
    llm_service = LLMService()
    search_service = SearchService()
    rag = RAGService(llm_service=llm_service, search_service=search_service)
    answer = await rag.a_answer(request.query)
    return QueryResponse(answer=answer)

