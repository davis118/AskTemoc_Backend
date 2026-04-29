from pydantic import BaseModel
from typing import List, Dict, Union, Optional

class Citation(BaseModel):
    url: str
    chunk_text: str
    document_title: Optional[str] = None
    chunk_index: Optional[int] = None

class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    # sources: List[Union[str, Dict]]