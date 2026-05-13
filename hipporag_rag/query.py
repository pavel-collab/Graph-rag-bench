"""
Query HippoRAG 2 and save answers to JSON.

Questions JSON format (input):
    {"questions": [{"id": "q1", "question": "...", "expected_answer": "..."}]}

Answers JSON format (output):
    {"answers": [{"id": "q1", "question": "...", "answer": "...",
                  "retrieval_context": [...], "expected_answer": "..."}]}

Usage:
    uv run scripts/query_hipporag.py --questions data/questions.json
    uv run scripts/query_hipporag.py --questions data/questions.json --num-retrieve 5
"""

import argparse
import json
import logging
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hipporag_rag.setup import get_hipporag_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def query_all(questions: list[dict], num_retrieve: int) -> list[dict]:
    from hipporag import HippoRAG

    cfg = get_hipporag_config()
    hipporag = HippoRAG(**cfg)

    texts = [item["question"] for item in questions]
    logger.info("Running rag_qa for %d questions...", len(texts))

    try:
        queries_solutions, _, _ = hipporag.rag_qa(queries=texts)
    except Exception as e:
        logger.error("rag_qa failed: %s", e)
        queries_solutions = [None] * len(texts)

    if len(queries_solutions) != len(questions):
        logger.warning(
            "rag_qa returned %d solutions for %d questions",
            len(queries_solutions), len(questions),
        )

    answers = []
    for item, sol in zip(questions, queries_solutions):
        answer = getattr(sol, "answer", "") or ""
        docs = getattr(sol, "docs", None) or []
        retrieval_context = [str(d) for d in docs[:num_retrieve]]

        answers.append({
            "id": item.get("id", ""),
            "question": item["question"],
            "answer": answer,
            "retrieval_context": retrieval_context,
            "expected_answer": item.get("expected_answer", ""),
        })

    return answers


def main() -> None:
    parser = argparse.ArgumentParser(description="Query HippoRAG 2")
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/answers/hipporag.json"))
    parser.add_argument("--num-retrieve", type=int, default=5)
    args = parser.parse_args()

    data = json.loads(args.questions.read_text(encoding="utf-8"))
    questions = data if isinstance(data, list) else data.get("questions", data.get("test_cases", []))

    answers = query_all(questions, args.num_retrieve)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"framework": "hipporag", "answers": answers}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Saved %d answers to %s", len(answers), args.output)


if __name__ == "__main__":
    main()
