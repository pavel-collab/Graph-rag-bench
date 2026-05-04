"""Query HippoRAG 2 (entry point)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hipporag_rag.query import main

if __name__ == "__main__":
    main()
