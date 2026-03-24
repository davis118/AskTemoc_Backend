from fastapi import APIRouter, Query
from typing import List

from app.services.search_service import search_service

router = APIRouter(prefix="/search", tags=["search"])


@router.get("/")
def search_documents(query: str = Query(...), k: int = Query(5)):

    results = search_service.search(query, k)

    return {
        "query": query,
        "results": results
    }