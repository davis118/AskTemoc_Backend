import logging
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
    CHROMA_PERSIST_DIRECTORY: str
    CHROMA_COLLECTION_NAME: str
    model_config = SettingsConfigDict(env_file=".env")


def get_settings() -> Settings:
    return Settings()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)

logger = logging.getLogger("app")