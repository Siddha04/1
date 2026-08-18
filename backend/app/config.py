"""
Centralized configuration for the backend. Every value is overridable via
environment variables or a .env file at the repo root.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str = "Personal RAG Assistant"
    ENV: str = "development"

    # --- LLM ---
    BASE_MODEL_ID: str = "mistralai/Mistral-7B-Instruct-v0.3"
    LLM_TEMPERATURE: float = 0.3
    LLM_MAX_TOKENS: int = 600

    # --- Embeddings ---
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"

    # --- Vector store ---
    CHROMA_PERSIST_DIR: str = "./data/vector_store"
    CHROMA_COLLECTION: str = "knowledge_base"

    # --- Live connectors ---
    TAVILY_API_KEY: str = ""
    SPORTSDB_API_KEY: str = "3"  # "3" = TheSportsDB's shared free test key

    # --- Retrieval tuning ---
    TOP_K_RESULTS: int = 5
    MAX_CONTEXT_CHARS: int = 6000

    # --- Training / CI-CD ---
    ADAPTER_DIR: str = "./models/adapters"
    ACTIVE_ADAPTER_FILE: str = "./models/adapters/ACTIVE"


@lru_cache
def get_settings() -> Settings:
    return Settings()
