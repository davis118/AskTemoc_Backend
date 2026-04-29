import logging
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DATABASE_URL: str
    LOG_LEVEL: str = "INFO"
    ENVIRONMENT: str = "development"
    DB_ECHO: bool = False
    DEBUG: bool = False

    # OpenAI
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-large"

    # Ollama (optional fallback)
    OLLAMA_MODEL: str = "llama3.1:8b"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"
    OLLAMA_TEMPERATURE: float = 0.4

    # ChromaDB (kept for compatibility)
    CHROMA_PERSIST_DIRECTORY: Path = Path("./app/chroma_db")
    CHROMA_COLLECTION_NAME: str = "asktemoc_collection"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def chroma_persist_path(self) -> Path:
        project_root = Path(__file__).resolve().parent.parent
        return (project_root / self.CHROMA_PERSIST_DIRECTORY).resolve()

    @property
    def use_openai(self) -> bool:
        return bool(self.OPENAI_API_KEY)


def get_settings() -> Settings:
    return Settings()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)

logger = logging.getLogger("app")
