from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str
    
class EmbedItem(BaseModel):
    chunk_id: str
    text: str
    metadata: dict = {}

class EmbedBatch(BaseModel):
    items: list[EmbedItem]

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5