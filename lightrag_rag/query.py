"""
Query LightRAG and save answers to JSON.

Questions JSON format (input):
    {"questions": [{"id": "q1", "question": "...", "expected_answer": "..."}]}

Answers JSON format (output):
    {"answers": [{"id": "q1", "question": "...", "answer": "...",
                  "mode": "hybrid", "retrieval_context": [<raw chunk>, ...],
                  "expected_answer": "..."}]}

retrieval_context contains the RAW text chunks LightRAG retrieved for the query,
extracted via aquery_data (same retrieval pipeline as aquery, stopped before LLM
generation). This is the apples-to-apples analog of HippoRAG's sol.docs and is
what DeepEval's Faithfulness / Hallucination / Contextual Precision metrics need
to be comparable between the two frameworks.

Usage:
    uv run scripts/query_lightrag.py --questions data/questions.json
    uv run scripts/query_lightrag.py --questions data/questions.json --mode hybrid
    uv run scripts/query_lightrag.py --questions data/questions.json --num-retrieve 5
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


def _extract_chunks(data_response: dict, num_retrieve: int) -> list[str]:
    """Pull the raw chunk strings out of aquery_data's structured response."""
    if not isinstance(data_response, dict) or data_response.get("status") != "success":
        return []
    chunks = data_response.get("data", {}).get("chunks") or []
    texts: list[str] = []
    for ch in chunks:
        content = ch.get("content") if isinstance(ch, dict) else None
        if content:
            texts.append(str(content))
        if len(texts) >= num_retrieve:
            break
    return texts


async def query_all(
    questions: list[dict],
    mode: QueryMode,
    top_k: int,
    num_retrieve: int,
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
                chunk_top_k=max(top_k // 2, num_retrieve),
                enable_rerank=False,
            )
            response = await rag.aquery(question, param=param)
            answer = response if isinstance(response, str) else str(response)

            # Same retrieval pipeline as aquery, but stops before LLM generation
            # and returns the structured list of chunks. This is what DeepEval
            # should see — raw retrieved text, not an LLM-synthesized paragraph.
            data_param = QueryParam(
                mode=mode,
                stream=False,
                top_k=top_k,
                chunk_top_k=max(top_k // 2, num_retrieve),
                enable_rerank=False,
            )
            data_response = await rag.aquery_data(question, param=data_param)
            retrieval_context = _extract_chunks(data_response, num_retrieve)

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
        default="hybrid",
        choices=["naive", "local", "global", "hybrid", "mix"],
        help="Retrieval mode for both answer generation and exported context.",
    )
    parser.add_argument("--top-k", type=int, default=60)
    parser.add_argument(
        "--num-retrieve",
        type=int,
        default=5,
        help="Number of chunks to export as retrieval_context. "
             "Matches HippoRAG's --num-retrieve so DeepEval sees comparable inputs.",
    )
    args = parser.parse_args()

    data = json.loads(args.questions.read_text(encoding="utf-8"))
    questions = data if isinstance(data, list) else data.get("questions", data.get("test_cases", []))

    answers = asyncio.run(query_all(questions, args.mode, args.top_k, args.num_retrieve))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"framework": "lightrag", "mode": args.mode, "answers": answers},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Saved %d answers to %s", len(answers), args.output)


if __name__ == "__main__":
    main()
