"""
Dataset loaders for RAG benchmarking.

Supported datasets:
  - hotpotqa   — multi-hop QA (100k+ Wikipedia questions, full + distractor variants)
  - musique     — multi-hop QA requiring 2–4 reasoning hops (Trivedi et al., 2022)
  - 2wikimhqa  — 2WikiMultiHopQA (cross-doc reasoning)
  - squad       — single-hop extraction (baseline / sanity check)
  - rgb         — RAG-specific benchmark (noise robustness, negative rejection)

Each loader returns:
  - corpus: list[str]  — documents to ingest
  - questions: list[dict]  — [{id, question, expected_answer, supporting_docs}]

Usage:
    from benchmark_datasets.loader import load_dataset
    corpus, questions = load_dataset("hotpotqa", split="validation", limit=200)
"""

import json
import logging
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

DatasetName = Literal["hotpotqa", "musique", "2wikimhqa", "squad", "rgb"]


def load_dataset(
    name: DatasetName,
    split: str = "validation",
    limit: int | None = None,
    cache_dir: str = "./data/datasets",
) -> tuple[list[str], list[dict]]:
    """Load a benchmark dataset and return (corpus, questions)."""
    loaders = {
        "hotpotqa": _load_hotpotqa,
        "musique": _load_musique,
        "2wikimhqa": _load_2wikimhqa,
        "squad": _load_squad,
        "rgb": _load_rgb,
    }
    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(loaders)}")

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    return loaders[name](split=split, limit=limit, cache_dir=cache_dir)


# ─────────────────────────────────────────────────────────────────────────────
# HotpotQA
# ─────────────────────────────────────────────────────────────────────────────

def _load_hotpotqa(split: str, limit: int | None, cache_dir: str):
    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        raise ImportError("pip install datasets  # HuggingFace datasets required")

    logger.info("Loading HotpotQA (%s)...", split)
    ds = hf_load("hotpot_qa", "distractor", split=split, trust_remote_code=True)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    corpus_texts: set[str] = set()
    questions = []

    for i, item in enumerate(ds):
        context_docs = [
            " ".join(sents)
            for title, sents in zip(item["context"]["title"], item["context"]["sentences"])
        ]
        for doc in context_docs:
            corpus_texts.add(doc)

        questions.append({
            "id": item.get("id", f"hotpot_{i}"),
            "question": item["question"],
            "expected_answer": item["answer"],
            "supporting_docs": [
                " ".join(item["context"]["sentences"][
                    item["context"]["title"].index(title)
                ])
                for title in item["supporting_facts"]["title"]
                if title in item["context"]["title"]
            ],
            "type": item.get("type", ""),
        })

    logger.info("HotpotQA: %d docs, %d questions", len(corpus_texts), len(questions))
    return list(corpus_texts), questions


# ─────────────────────────────────────────────────────────────────────────────
# MuSiQue
# ─────────────────────────────────────────────────────────────────────────────

def _load_musique(split: str, limit: int | None, cache_dir: str):
    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        raise ImportError("pip install datasets")

    logger.info("Loading MuSiQue (%s)...", split)
    ds = hf_load("dgslibisey/MuSiQue", split=split, trust_remote_code=True)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    corpus_texts: set[str] = set()
    questions = []

    for i, item in enumerate(ds):
        for para in item.get("paragraphs", []):
            text = para.get("paragraph_text", "").strip()
            if text:
                corpus_texts.add(text)

        questions.append({
            "id": item.get("id", f"musique_{i}"),
            "question": item["question"],
            "expected_answer": item.get("answer", ""),
            "supporting_docs": [
                p["paragraph_text"]
                for p in item.get("paragraphs", [])
                if p.get("is_supporting")
            ],
        })

    logger.info("MuSiQue: %d docs, %d questions", len(corpus_texts), len(questions))
    return list(corpus_texts), questions


# ─────────────────────────────────────────────────────────────────────────────
# 2WikiMultiHopQA
# ─────────────────────────────────────────────────────────────────────────────

def _load_2wikimhqa(split: str, limit: int | None, cache_dir: str):
    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        raise ImportError("pip install datasets")

    logger.info("Loading 2WikiMultiHopQA (%s)...", split)
    ds = hf_load("xanhho/2WikiMultiHopQA", split=split, trust_remote_code=True)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    corpus_texts: set[str] = set()
    questions = []

    for i, item in enumerate(ds):
        for title, sents in zip(
            item["context"]["title"], item["context"]["sentences"]
        ):
            corpus_texts.add(" ".join(sents))

        questions.append({
            "id": item.get("_id", f"2wiki_{i}"),
            "question": item["question"],
            "expected_answer": item.get("answer", ""),
            "supporting_docs": [],
        })

    logger.info("2WikiMultiHopQA: %d docs, %d questions", len(corpus_texts), len(questions))
    return list(corpus_texts), questions


# ─────────────────────────────────────────────────────────────────────────────
# SQuAD (single-hop baseline)
# ─────────────────────────────────────────────────────────────────────────────

def _load_squad(split: str, limit: int | None, cache_dir: str):
    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        raise ImportError("pip install datasets")

    logger.info("Loading SQuAD v2 (%s)...", split)
    ds = hf_load("squad_v2", split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    corpus_texts: set[str] = set()
    questions = []

    for i, item in enumerate(ds):
        corpus_texts.add(item["context"])
        answers = item["answers"]["text"]
        questions.append({
            "id": item.get("id", f"squad_{i}"),
            "question": item["question"],
            "expected_answer": answers[0] if answers else "",
            "supporting_docs": [item["context"]],
        })

    logger.info("SQuAD: %d docs, %d questions", len(corpus_texts), len(questions))
    return list(corpus_texts), questions


# ─────────────────────────────────────────────────────────────────────────────
# RGB — RAG-specific benchmark
# ─────────────────────────────────────────────────────────────────────────────

def _load_rgb(split: str, limit: int | None, cache_dir: str):
    """
    RGB (Chen et al., 2023): tests noise robustness, negative rejection,
    information integration, and counterfactual robustness.
    Source: https://huggingface.co/datasets/explodinggradients/ragas-wikiqa
    Fallback: locally cached JSON if HuggingFace is unavailable.
    """
    try:
        from datasets import load_dataset as hf_load
        logger.info("Loading RGB (ragas-wikiqa) (%s)...", split)
        ds = hf_load("explodinggradients/ragas-wikiqa", split="train")
        if limit:
            ds = ds.select(range(min(limit, len(ds))))

        corpus_texts: set[str] = set()
        questions = []
        for i, item in enumerate(ds):
            for ctx in item.get("context", []):
                if ctx:
                    corpus_texts.add(ctx)
            questions.append({
                "id": f"rgb_{i}",
                "question": item["question"],
                "expected_answer": item.get("correct_answer", ""),
                "supporting_docs": item.get("context", []),
            })

        logger.info("RGB: %d docs, %d questions", len(corpus_texts), len(questions))
        return list(corpus_texts), questions

    except Exception as e:
        logger.warning("RGB HuggingFace load failed (%s). Returning empty dataset.", e)
        return [], []


# ─────────────────────────────────────────────────────────────────────────────
# Save helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_questions(questions: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"questions": questions}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Saved %d questions to %s", len(questions), path)


def save_corpus(corpus: list[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, text in enumerate(corpus):
        (out_dir / f"doc_{i:05d}.txt").write_text(text, encoding="utf-8")
    logger.info("Saved %d corpus documents to %s", len(corpus), out_dir)
