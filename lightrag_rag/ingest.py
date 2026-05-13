"""
Ingest documents into LightRAG.

Usage:
    uv run scripts/ingest_lightrag.py --docs-dir ./data/corpus
    uv run scripts/ingest_lightrag.py --docs-dir ./data/corpus --batch-size 10
"""

import argparse
import asyncio
import hashlib
import json
import logging
from pathlib import Path

from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lightrag_rag.setup import create_lightrag

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md"}
PROGRESS_FILE = ".ingest_progress.json"


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _load_docs(docs_dir: Path, recursive: bool) -> list[tuple[Path, str]]:
    pattern = "**/*" if recursive else "*"
    docs = []
    for path in sorted(docs_dir.glob(pattern)):
        if path.suffix.lower() in SUPPORTED_EXTENSIONS and path.is_file():
            try:
                content = path.read_text(encoding="utf-8").strip()
                if content:
                    docs.append((path, content))
            except Exception as e:
                logger.warning("Cannot read %s: %s", path, e)
    return docs


async def ingest(docs_dir: Path, recursive: bool, batch_size: int, resume: bool, max_docs: int | None = None) -> int:
    rag = await create_lightrag()

    docs = _load_docs(docs_dir, recursive)
    logger.info("Found %d documents in %s", len(docs), docs_dir)
    if max_docs is not None:
        docs = docs[:max_docs]
        logger.info("Limiting to %d documents (--max-docs)", max_docs)

    progress_path = Path(rag.working_dir) / PROGRESS_FILE
    uploaded: dict[str, str] = {}
    if resume and progress_path.exists():
        uploaded = json.loads(progress_path.read_text())
        logger.info("Resuming: %d already ingested", len(uploaded))

    pending = [(p, c) for p, c in docs if str(p) not in uploaded or uploaded[str(p)] != _hash(c)]
    logger.info("Pending: %d documents", len(pending))

    count = 0
    for i in range(0, len(pending), batch_size):
        batch = pending[i : i + batch_size]
        texts = [c for _, c in batch]
        try:
            await rag.ainsert(texts)
            for path, content in batch:
                uploaded[str(path)] = _hash(content)
            progress_path.write_text(json.dumps(uploaded, ensure_ascii=False, indent=2))
            count += len(batch)
        except Exception as e:
            logger.error("Batch %d failed: %s", i // batch_size, e)

        logger.info("Progress: %d / %d", count, len(pending))

    logger.info("Done. Ingested %d documents total.", count)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into LightRAG")
    parser.add_argument("--docs-dir", required=True, type=Path)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-docs", type=int, default=None, help="Limit total number of documents to ingest")
    args = parser.parse_args()

    if not args.docs_dir.exists():
        logger.error("Directory not found: %s", args.docs_dir)
        raise SystemExit(1)

    asyncio.run(ingest(args.docs_dir, args.recursive, args.batch_size, args.resume, args.max_docs))


if __name__ == "__main__":
    main()
