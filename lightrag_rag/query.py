"""
Query LightRAG and save answers to JSON.

Questions JSON format (input):
    {"questions": [{"id": "q1", "question": "...", "expected_answer": "..."}]}

Answers JSON format (output):
    {"answers": [{"id": "q1", "question": "...", "answer": "...",
                  "mode": "mix", "retrieval_context": [...], "expected_answer": "..."}]}

Usage:
    uv run scripts/query_lightrag.py --questions data/questions.json --output data/answers/lightrag.json
    uv run scripts/query_lightrag.py --questions data/questions.json --mode hybrid
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Literal

from lightrag import QueryParam

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lightrag_rag.setup import create_lightrag

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

QueryMode = Literal["naive", "local", "global", "hybrid", "mix"]


async def query_all(
    questions: list[dict],
    mode: QueryMode,
    top_k: int,
) -> list[dict]:
    rag = await create_lightrag()
    answers = []

    for item in questions:
        qid = item.get("id", "")
        question = item["question"]
        logger.info("[%s] Querying: %s", qid, question[:80])

        try:
            param = QueryParam(
                mode=mode,
                stream=False,
                top_k=top_k,
                chunk_top_k=top_k // 2,
                enable_rerank=False,
            )
            response = await rag.aquery(question, param=param)
            answer = response if isinstance(response, str) else str(response)

            # Collect retrieval context via low-level naive retrieval for DeepEval
            ctx_param = QueryParam(mode="naive", stream=False, top_k=top_k, enable_rerank=False)
            ctx_response = await rag.aquery(question, param=ctx_param)
            retrieval_context = [ctx_response] if isinstance(ctx_response, str) else []

        except Exception as e:
            logger.error("[%s] Query failed: %s", qid, e)
            answer = ""
            retrieval_context = []

        answers.append({
            "id": qid,
            "question": question,
            "answer": answer,
            "mode": mode,
            "retrieval_context": retrieval_context,
            "expected_answer": item.get("expected_answer", ""),
        })

    return answers


def main() -> None:
    parser = argparse.ArgumentParser(description="Query LightRAG")
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/answers/lightrag.json"))
    parser.add_argument(
        "--mode",
        default="mix",
        choices=["naive", "local", "global", "hybrid", "mix"],
    )
    parser.add_argument("--top-k", type=int, default=60)
    args = parser.parse_args()

    data = json.loads(args.questions.read_text(encoding="utf-8"))
    questions = data if isinstance(data, list) else data.get("questions", data.get("test_cases", []))

    answers = asyncio.run(query_all(questions, args.mode, args.top_k))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"framework": "lightrag", "mode": args.mode, "answers": answers},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Saved %d answers to %s", len(answers), args.output)


if __name__ == "__main__":
    main()
