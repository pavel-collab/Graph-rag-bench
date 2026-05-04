"""
Obsidian vault loader for RAG ingestion.

Loads .md files from an Obsidian vault, strips frontmatter and wikilinks,
and returns plain-text documents suitable for indexing.
"""

import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_FRONTMATTER_RE = re.compile(r"^---\n.*?\n---\n", re.DOTALL)
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_HEADING_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)


def _clean(text: str) -> str:
    text = _FRONTMATTER_RE.sub("", text)
    text = _WIKILINK_RE.sub(r"\1", text)
    text = _MD_LINK_RE.sub(r"\1", text)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    return text.strip()


def load_obsidian_vault(
    vault_path: str | Path,
    recursive: bool = True,
    min_chars: int = 100,
    exclude_dirs: set[str] | None = None,
) -> tuple[list[str], list[Path]]:
    """
    Load text documents from an Obsidian vault.

    Args:
        vault_path: Path to the vault root.
        recursive: Whether to traverse subdirectories.
        min_chars: Minimum character count to include a document.
        exclude_dirs: Directory names to skip (e.g. {'.obsidian', 'templates'}).

    Returns:
        (texts, paths) — parallel lists of document text and their source paths.
    """
    vault = Path(vault_path)
    if not vault.exists():
        raise FileNotFoundError(f"Vault not found: {vault}")

    excluded = exclude_dirs or {".obsidian", ".trash", "templates", "attachments"}
    pattern = "**/*.md" if recursive else "*.md"

    texts, paths = [], []
    for md_path in sorted(vault.glob(pattern)):
        # Skip excluded directories
        if any(part in excluded for part in md_path.parts):
            continue
        try:
            raw = md_path.read_text(encoding="utf-8")
            cleaned = _clean(raw)
            if len(cleaned) >= min_chars:
                texts.append(cleaned)
                paths.append(md_path)
        except Exception as e:
            logger.warning("Cannot read %s: %s", md_path, e)

    logger.info("Obsidian vault %s: loaded %d documents", vault, len(texts))
    return texts, paths


def obsidian_to_questions(paths: list[Path], texts: list[str]) -> list[dict]:
    """
    Generate simple factoid questions from Obsidian note titles.
    Used for quick smoke-testing when no gold QA set is available.
    """
    questions = []
    for i, (path, text) in enumerate(zip(paths, texts)):
        title = path.stem.replace("-", " ").replace("_", " ")
        questions.append({
            "id": f"obs_{i:04d}",
            "question": f"What is described in the note about '{title}'?",
            "expected_answer": "",
            "source_path": str(path),
        })
    return questions
