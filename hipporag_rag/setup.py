"""
HippoRAG 2 initialization.

Storage: HippoRAG 2 uses local file-based storage only.
  - embeddings/ — chunk + node embeddings (.parquet / FAISS)
  - graph/       — serialized knowledge graph
  - openie_cache/ — cached triplet extraction results (keyed by content hash)

External DB support: not available in the current HippoRAG 2 release.
All state lives under `save_dir`.
"""

import os
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings


def get_hipporag_config() -> dict:
    """Return constructor kwargs for HippoRAG."""
    os.environ["OPENAI_API_KEY"] = settings.openai_api_key.get_secret_value()

    wd = settings.hipporag_working_dir
    Path(wd).mkdir(parents=True, exist_ok=True)

    return dict(
        save_dir=wd,
        llm_model=settings.llm_model,
        embedding_model=settings.embedding_model,
        llm_base_url=settings.llm_base_url,
        # Personalized PageRank damping factor (higher = longer hops)
        damping=0.1,
    )
