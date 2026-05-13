from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    openai_api_key: SecretStr = SecretStr("sk-placeholder")
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "openai/gpt-4o-mini"

    # LightRAG embeddings (OpenAI-compatible via OpenRouter, 1536-dim)
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # HippoRAG embeddings (local sentence-transformers, 1024-dim)
    # Uses "Transformers/<model>" prefix to route to TransformersEmbeddingModel
    hipporag_embedding_model: str = "Transformers/BAAI/bge-m3"
    hipporag_embedding_dim: int = 1024

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "postgres"
    postgres_password: SecretStr = SecretStr("postgres")
    postgres_db: str = "lightrag"

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: SecretStr = SecretStr("password")

    # Qdrant
    qdrant_url: str = "http://localhost:6333"

    # Redis
    redis_url: str = "redis://localhost:6379"

    # Storage paths
    lightrag_working_dir: str = "./storage/lightrag"
    hipporag_working_dir: str = "./storage/hipporag"


settings = Settings()
