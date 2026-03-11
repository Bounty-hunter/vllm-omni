import logging
import subprocess
import sys
import tempfile
from pathlib import Path

OVI_REPO = "https://github.com/character-ai/Ovi.git"
CACHE_DIR = Path(tempfile.gettempdir()) / "vllm-omni-dependency"

logger = logging.getLogger(__name__)


def ensure_dependencies():
    if not (CACHE_DIR / "ovi").exists():
        logger.info(f"[ensure_dependencies] downloading dependency to {CACHE_DIR} ...")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        subprocess.run(["git", "clone", "--depth", "1", OVI_REPO, str(CACHE_DIR)], check=True)
        logger.info(f"[ensure_dependencies] finish download dependency to {CACHE_DIR} ...")
    if str(CACHE_DIR) not in sys.path:
        sys.path.insert(0, str(CACHE_DIR))
