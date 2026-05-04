"""
Ingest documents into HippoRAG 2.

Usage:
    uv run scripts/ingest_hipporag.py --docs-dir ./data/corpus
    uv run scripts/ingest_hipporag.py --docs-dir ./data/corpus --recursive
"""

import argparse
import logging
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hipporag_rag.setup import get_hipporag_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md"}


def _load_docs(docs_dir: Path, recursive: bool) -> list[str]:
    pattern = "**/*" if recursive else "*"
    docs = []
    for path in sorted(docs_dir.glob(pattern)):
        if path.suffix.lower() in SUPPORTED_EXTENSIONS and path.is_file():
            try:
                content = path.read_text(encoding="utf-8").strip()
                if content:
                    docs.append(content)
            except Exception as e:
                logger.warning("Cannot read %s: %s", path, e)
    return docs


def ingest(docs_dir: Path, recursive: bool) -> int:
    # Lazy import — hipporag pulls in torch/transformers which are heavy
    from hipporag import HippoRAG

    docs = _load_docs(docs_dir, recursive)
    logger.info("Loaded %d documents from %s", len(docs), docs_dir)

    cfg = get_hipporag_config()
    logger.info("Initializing HippoRAG (save_dir=%s)", cfg["save_dir"])
    hipporag = HippoRAG(**cfg)

    logger.info("Indexing...")
    hipporag.index(docs=docs)
    logger.info("Done. %d documents indexed.", len(docs))
    return len(docs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into HippoRAG 2")
    parser.add_argument("--docs-dir", required=True, type=Path)
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()

    if not args.docs_dir.exists():
        logger.error("Directory not found: %s", args.docs_dir)
        raise SystemExit(1)

    ingest(args.docs_dir, args.recursive)


if __name__ == "__main__":
    main()
