import logging
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    DATABASE_URL: str
    LOG_LEVEL: str
    ENVIRONMENT: str
    DB_ECHO: bool = False
    DEBUG: bool = False
    OLLAMA_MODEL: str
    OLLAMA_BASE_URL: str
    OLLAMA_EMBEDDING_MODEL: str
    OLLAMA_TEMPERATURE: float = 0.4
    CHROMA_PERSIST_DIRECTORY: Path
    CHROMA_COLLECTION_NAME: str
    model_config = SettingsConfigDict(env_file=".env")
    
    @property
    def chroma_persist_path(self) -> Path:
        project_root = Path(__file__).resolve().parent.parent
        return (project_root / self.CHROMA_PERSIST_DIRECTORY).resolve()


def get_settings() -> Settings:
    return Settings()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)

logger = logging.getLogger("app")