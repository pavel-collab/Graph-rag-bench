"""
LightRAG initialization with external storage backends.

Storage layout:
  - PostgreSQL  → KV storage (full_docs, text_chunks, entity/relation metadata)
  - Neo4j       → Graph storage (knowledge graph: nodes + edges)
  - Qdrant      → Vector storage (entity/chunk/relation embeddings)
  - Redis       → Document status storage (indexing pipeline state)
"""

import os
from functools import partial

from lightrag import LightRAG
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from config import settings


def _set_env_vars() -> None:
    """Expose connection settings as env vars required by LightRAG storage backends."""
    os.environ["POSTGRES_HOST"] = settings.postgres_host
    os.environ["POSTGRES_PORT"] = str(settings.postgres_port)
    os.environ["POSTGRES_USER"] = settings.postgres_user
    os.environ["POSTGRES_PASSWORD"] = settings.postgres_password.get_secret_value()
    os.environ["POSTGRES_DATABASE"] = settings.postgres_db

    os.environ["NEO4J_URI"] = settings.neo4j_uri
    os.environ["NEO4J_USERNAME"] = settings.neo4j_username
    os.environ["NEO4J_PASSWORD"] = settings.neo4j_password.get_secret_value()

    os.environ["QDRANT_URL"] = settings.qdrant_url

    os.environ["REDIS_URI"] = settings.redis_url

    os.environ["OPENAI_API_KEY"] = settings.openai_api_key.get_secret_value()


async def _llm_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        settings.llm_model,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=settings.openai_api_key.get_secret_value(),
        base_url=settings.llm_base_url,
        **kwargs,
    )


async def create_lightrag(working_dir: str | None = None) -> LightRAG:
    """
    Create and initialize a LightRAG instance backed by four databases:
      PostgreSQL (KV) · Neo4j (Graph) · Qdrant (Vector) · Redis (DocStatus)
    """
    _set_env_vars()

    wd = working_dir or settings.lightrag_working_dir
    os.makedirs(wd, exist_ok=True)

    rag = LightRAG(
        working_dir=wd,
        # ── LLM ──────────────────────────────────────────────────────────────
        llm_model_func=_llm_func,
        # ── Embedding ────────────────────────────────────────────────────────
        embedding_func=EmbeddingFunc(
            embedding_dim=settings.embedding_dim,
            max_token_size=8192,
            model_name=settings.embedding_model,
            func=partial(
                openai_embed,
                model=settings.embedding_model,
                base_url=settings.llm_base_url,
                api_key=settings.openai_api_key.get_secret_value(),
            ),
        ),
        # ── Storage backends (one DB per concern) ────────────────────────────
        kv_storage="RedisKVStorage",
        graph_storage="Neo4JStorage",        # Neo4j:      knowledge graph
        vector_storage="QdrantVectorDBStorage",  # Qdrant: entity/chunk/rel embeddings
        doc_status_storage="PGDocStatusStorage", # PostgreSQL: pipeline status per doc
        # ── Chunking ─────────────────────────────────────────────────────────
        chunk_token_size=1200,
        chunk_overlap_token_size=200,
        # ── Retrieval defaults ───────────────────────────────────────────────
        top_k=60,
        chunk_top_k=30,
        cosine_threshold=0.2,
        cosine_better_than_threshold=0.2,
        # ── Entity extraction ────────────────────────────────────────────────
        entity_extract_max_gleaning=1,
        max_graph_nodes=2000,
        # ── Token budgets ────────────────────────────────────────────────────
        max_entity_tokens=8000,
        max_relation_tokens=10000,
        max_total_tokens=32000,
    )

    await rag.initialize_storages()
    return rag
