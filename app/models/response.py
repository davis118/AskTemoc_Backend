from pydantic import BaseModel, Field
from typing import List, Dict, Union, Optional

class QueryResponse(BaseModel):
    answer: str
    sources: Optional[List[Union[str, Dict]]] = Field(default=None)