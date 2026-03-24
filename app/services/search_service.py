from typing import List, Dict, Any
from app.services.retriever_service import retriever_service


class SearchService:

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:

        results = retriever_service.search(query, k)

        formatted = []

        for doc in results:

            formatted.append({
                "text": doc.page_content,
                "metadata": doc.metadata
            })

        return formatted


search_service = SearchService()