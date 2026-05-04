"""
RAG evaluation with DeepEval metrics.

Reads answers JSON produced by query_lightrag.py / query_hipporag.py and computes:
  - Answer Relevancy   — does the answer address the question?
  - Faithfulness       — is the answer grounded in the retrieved context?
  - Contextual Recall  — does the context contain the expected answer facts?
  - Contextual Precision — is the retrieved context relevant (no noise)?
  - Hallucination      — does the answer contain facts absent from the context?

Usage:
    uv run scripts/evaluate.py --answers data/answers/lightrag.json
    uv run scripts/evaluate.py --answers data/answers/lightrag.json data/answers/hipporag.json --compare
    uv run scripts/evaluate.py --answers data/answers/lightrag.json --charts
"""

import argparse
import json
import logging
import os
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _setup_deepeval() -> None:
    """Configure DeepEval to use the project LLM. DeepEval requires OPENAI_API_KEY."""
    os.environ["OPENAI_API_KEY"] = settings.openai_api_key.get_secret_value()
    os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS", "120")
    os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "600")


def _build_test_cases(answers: list[dict]):
    from deepeval.test_case import LLMTestCase

    cases = []
    for item in answers:
        question = item.get("question", "")
        answer = item.get("answer", "")
        expected = item.get("expected_answer", "")
        context = item.get("retrieval_context", [])

        if not answer:
            logger.warning("Skipping %s: empty answer", item.get("id", "?"))
            continue

        cases.append(
            LLMTestCase(
                input=question,
                actual_output=answer,
                expected_output=expected or None,
                retrieval_context=context if context else None,
                context=context if context else None,
            )
        )
    return cases


def _build_metrics(use_expected: bool):
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        FaithfulnessMetric,
        ContextualRecallMetric,
        ContextualPrecisionMetric,
        HallucinationMetric,
    )

    metrics = [
        AnswerRelevancyMetric(threshold=0.5, verbose_mode=False),
        FaithfulnessMetric(threshold=0.5, verbose_mode=False),
        HallucinationMetric(threshold=0.5, verbose_mode=False),
        ContextualPrecisionMetric(threshold=0.5, verbose_mode=False),
    ]
    if use_expected:
        metrics.append(ContextualRecallMetric(threshold=0.5, verbose_mode=False))
    return metrics


def evaluate_file(answers_path: Path, charts: bool) -> dict:
    from deepeval import evaluate as dv_evaluate

    _setup_deepeval()

    data = json.loads(answers_path.read_text(encoding="utf-8"))
    raw_answers = data if isinstance(data, list) else data.get("answers", [])
    framework = data.get("framework", answers_path.stem) if isinstance(data, dict) else answers_path.stem

    logger.info("Evaluating %s (%d answers)", framework, len(raw_answers))

    test_cases = _build_test_cases(raw_answers)
    if not test_cases:
        logger.warning("No valid test cases found in %s", answers_path)
        return {}

    has_expected = any(tc.expected_output for tc in test_cases)
    metrics = _build_metrics(has_expected)

    results = dv_evaluate(test_cases=test_cases, metrics=metrics, run_async=True)

    # Aggregate scores
    metric_scores: dict[str, list[float]] = {}
    for tc_result in results.test_results:
        for m in tc_result.metrics_data:
            metric_scores.setdefault(m.name, []).append(m.score or 0.0)

    summary = {
        name: round(sum(scores) / len(scores), 4)
        for name, scores in metric_scores.items()
    }

    output = {
        "framework": framework,
        "n_cases": len(test_cases),
        "metrics": summary,
        "per_question": [
            {
                "id": raw_answers[i].get("id", str(i)),
                "question": raw_answers[i].get("question", ""),
                "scores": {
                    m.name: m.score
                    for m in results.test_results[i].metrics_data
                    if i < len(results.test_results)
                },
            }
            for i in range(min(len(test_cases), len(results.test_results)))
        ],
    }

    out_path = answers_path.parent / f"{answers_path.stem}_eval.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Results saved to %s", out_path)

    if charts:
        _save_charts(output, answers_path.parent)

    return output


def _save_charts(result: dict, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt

        framework = result["framework"]
        metrics = result["metrics"]

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.barh(list(metrics.keys()), list(metrics.values()))
        ax.set_xlim(0, 1)
        ax.set_xlabel("Score")
        ax.set_title(f"RAG Metrics — {framework}")
        for bar, val in zip(bars, metrics.values()):
            ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center")
        plt.tight_layout()
        chart_path = out_dir / f"{framework}_metrics.png"
        fig.savefig(chart_path, dpi=150)
        plt.close(fig)
        logger.info("Chart saved: %s", chart_path)
    except Exception as e:
        logger.warning("Chart generation failed: %s", e)


def compare(results: list[dict], out_dir: Path) -> None:
    """Print and save side-by-side comparison table."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        all_metrics = sorted({m for r in results for m in r.get("metrics", {})})
        frameworks = [r["framework"] for r in results]

        scores = np.array([
            [r["metrics"].get(m, 0.0) for m in all_metrics]
            for r in results
        ])

        x = np.arange(len(all_metrics))
        width = 0.8 / len(frameworks)

        fig, ax = plt.subplots(figsize=(max(10, len(all_metrics) * 2), 5))
        for i, (fw, row) in enumerate(zip(frameworks, scores)):
            offset = (i - len(frameworks) / 2 + 0.5) * width
            ax.bar(x + offset, row, width, label=fw)

        ax.set_xticks(x)
        ax.set_xticklabels(all_metrics, rotation=15, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.set_title("RAG Framework Comparison")
        ax.legend()
        plt.tight_layout()
        chart_path = out_dir / "comparison.png"
        fig.savefig(chart_path, dpi=150)
        plt.close(fig)
        logger.info("Comparison chart saved: %s", chart_path)
    except Exception as e:
        logger.warning("Comparison chart failed: %s", e)

    # Print table to console
    print("\n=== COMPARISON ===")
    all_metrics_sorted = sorted({m for r in results for m in r.get("metrics", {})})
    header = f"{'Metric':<35}" + "".join(f"{fw:<15}" for fw in [r['framework'] for r in results])
    print(header)
    print("-" * len(header))
    for metric in all_metrics_sorted:
        row = f"{metric:<35}" + "".join(f"{r['metrics'].get(metric, 0.0):<15.4f}" for r in results)
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG answers with DeepEval")
    parser.add_argument("--answers", nargs="+", required=True, type=Path,
                        help="One or more answer JSON files")
    parser.add_argument("--compare", action="store_true",
                        help="Generate comparison chart when multiple files given")
    parser.add_argument("--charts", action="store_true")
    args = parser.parse_args()

    all_results = []
    for path in args.answers:
        if not path.exists():
            logger.error("File not found: %s", path)
            continue
        result = evaluate_file(path, args.charts)
        if result:
            all_results.append(result)

    if args.compare and len(all_results) > 1:
        out_dir = args.answers[0].parent
        compare(all_results, out_dir)


if __name__ == "__main__":
    main()
