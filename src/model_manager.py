import logging
import sys
import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
MODEL_FILENAME = "face_landmarker_v2_with_blendshapes.task"
EXPECTED_SIZE_MIN = 3_500_000  # ~3.8MB

logger = logging.getLogger(__name__)


def ensure_model(model_path: Path) -> Path:
    """Check if model exists; download if missing. Returns model path."""
    if model_path.exists() and model_path.stat().st_size >= EXPECTED_SIZE_MIN:
        logger.info(f"Model found: {model_path}")
        return model_path

    logger.info(f"Model not found at {model_path}, downloading...")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    _download(model_path)
    return model_path


def _download(dest: Path) -> None:
    """Download the face landmarker model with progress."""

    def _progress(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb = downloaded / 1_000_000
            total_mb = total_size / 1_000_000
            sys.stdout.write(f"\rDownloading model: {mb:.1f}/{total_mb:.1f} MB ({pct}%)")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(MODEL_URL, str(dest), reporthook=_progress)
        print()  # newline after progress
        size = dest.stat().st_size
        if size < EXPECTED_SIZE_MIN:
            dest.unlink()
            raise RuntimeError(
                f"Downloaded file too small ({size} bytes), expected >= {EXPECTED_SIZE_MIN}"
            )
        logger.info(f"Model downloaded: {dest} ({size / 1_000_000:.1f} MB)")
    except Exception as e:
        if dest.exists():
            dest.unlink()
        raise RuntimeError(f"Failed to download model: {e}") from e
