"""
Database session management and initialization.
"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from app.db.models import Base
from app.core.config import get_settings


def _make_engine():
    settings = get_settings()
    url = settings.DATABASE_URL
    return create_engine(
        url,
        connect_args={"check_same_thread": False} if "sqlite" in url else {},
        echo=settings.DB_ECHO,
    )


engine = _make_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Initialize database: enable pgvector extension (Postgres only), then create all tables.
    """
    settings = get_settings()
    if "postgresql" in settings.DATABASE_URL:
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()

    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully.")


def drop_db():
    Base.metadata.drop_all(bind=engine)
    print("All database tables dropped.")
