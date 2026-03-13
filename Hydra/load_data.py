from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ospec_common.chunking import save_chunks_to_jsonl

__all__ = ["save_chunks_to_jsonl"]
