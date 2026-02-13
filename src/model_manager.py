import logging
import sys
import urllib.request
from pathlib import Path

MODELS = {
    "face": {
        "url": (
            "https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        ),
        "filename": "face_landmarker_v2_with_blendshapes.task",
        "min_size": 3_500_000,
    },
    "pose_lite": {
        "url": (
            "https://storage.googleapis.com/mediapipe-models/"
            "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
        ),
        "filename": "pose_landmarker_lite.task",
        "min_size": 5_000_000,
    },
    "pose_full": {
        "url": (
            "https://storage.googleapis.com/mediapipe-models/"
            "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
        ),
        "filename": "pose_landmarker_full.task",
        "min_size": 8_000_000,
    },
    "pose_heavy": {
        "url": (
            "https://storage.googleapis.com/mediapipe-models/"
            "pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
        ),
        "filename": "pose_landmarker_heavy.task",
        "min_size": 20_000_000,
    },
    "hand": {
        "url": (
            "https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
        ),
        "filename": "hand_landmarker.task",
        "min_size": 3_000_000,
    },
}

# Backwards-compatible aliases
MODEL_URL = MODELS["face"]["url"]
MODEL_FILENAME = MODELS["face"]["filename"]
EXPECTED_SIZE_MIN = MODELS["face"]["min_size"]

logger = logging.getLogger(__name__)


def ensure_model(model_path: Path, model_key: str = "face") -> Path:
    """Check if model exists; download if missing. Returns model path."""
    info = MODELS[model_key]
    min_size = info["min_size"]

    if model_path.exists() and model_path.stat().st_size >= min_size:
        logger.info(f"Model found: {model_path}")
        return model_path

    logger.info(f"Model not found at {model_path}, downloading...")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    _download(info["url"], model_path, min_size)
    return model_path


def get_model_path(model_dir: Path, model_key: str) -> Path:
    """Get the expected path for a model file."""
    return model_dir / MODELS[model_key]["filename"]


def _download(url: str, dest: Path, min_size: int) -> None:
    """Download a model with progress."""

    def _progress(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb = downloaded / 1_000_000
            total_mb = total_size / 1_000_000
            sys.stdout.write(f"\rDownloading {dest.name}: {mb:.1f}/{total_mb:.1f} MB ({pct}%)")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, str(dest), reporthook=_progress)
        print()  # newline after progress
        size = dest.stat().st_size
        if size < min_size:
            dest.unlink()
            raise RuntimeError(
                f"Downloaded file too small ({size} bytes), expected >= {min_size}"
            )
        logger.info(f"Model downloaded: {dest} ({size / 1_000_000:.1f} MB)")
    except Exception as e:
        if dest.exists():
            dest.unlink()
        raise RuntimeError(f"Failed to download model: {e}") from e
