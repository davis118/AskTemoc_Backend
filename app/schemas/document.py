from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime


class DocumentResponse(BaseModel):
    id: str
    title: str
    source: Optional[str] = None
    doc_metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime]

    model_config = {
        "from_attributes": True
    }