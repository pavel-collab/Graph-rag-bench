"""
Download and prepare a benchmark dataset for ingestion.

Usage:
    uv run scripts/prepare_dataset.py --dataset hotpotqa --limit 200
    uv run scripts/prepare_dataset.py --dataset musique --limit 100
    uv run scripts/prepare_dataset.py --obsidian /path/to/vault
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmark_datasets.loader import DatasetName, load_dataset, save_corpus, save_questions
from benchmark_datasets.obsidian_loader import load_obsidian_vault, obsidian_to_questions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare benchmark dataset")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", choices=["hotpotqa", "musique", "2wikimhqa", "squad", "rgb"])
    group.add_argument("--obsidian", type=Path, help="Path to Obsidian vault")

    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=200, help="Max number of QA pairs")
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/corpus"))
    parser.add_argument("--questions-out", type=Path, default=Path("data/questions.json"))
    args = parser.parse_args()

    if args.obsidian:
        texts, paths = load_obsidian_vault(args.obsidian)
        questions = obsidian_to_questions(paths, texts)
        if args.limit:
            texts = texts[: args.limit]
            questions = questions[: args.limit]
        save_corpus(texts, args.corpus_dir)
        save_questions(questions, args.questions_out)
    else:
        corpus, questions = load_dataset(
            args.dataset, split=args.split, limit=args.limit
        )
        if args.limit:
            questions = questions[: args.limit]
        save_corpus(corpus, args.corpus_dir)
        save_questions(questions, args.questions_out)

    logger.info("Done. Corpus: %s | Questions: %s", args.corpus_dir, args.questions_out)


if __name__ == "__main__":
    main()
