"""Ingest documents into LightRAG (entry point)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lightrag_rag.ingest import main

if __name__ == "__main__":
    main()
