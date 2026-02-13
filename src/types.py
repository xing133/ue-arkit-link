from dataclasses import dataclass, field
import numpy as np


@dataclass
class FrameResult:
    """Result of processing a single video frame."""

    frame_index: int
    timestamp_ms: int
    # Per-face list of 468 landmarks, each with x, y, z (normalized 0-1)
    landmarks: list[list[dict[str, float]]]
    # Per-face dict of ARKit blendshape name -> score (0.0-1.0)
    blendshapes: list[dict[str, float]]
    # Per-face 4x4 facial transformation matrix
    transformation_matrix: list[np.ndarray | None]
    # Body pose (33 landmarks)
    pose_landmarks: list[dict[str, float]] | None = None
    pose_world_landmarks: list[dict[str, float]] | None = None
    # Hands (up to 2 hands, 21 landmarks each)
    hand_landmarks: list[list[dict[str, float]]] | None = None
    hand_world_landmarks: list[list[dict[str, float]]] | None = None
    handedness: list[str] | None = None


@dataclass
class VideoMetadata:
    """Metadata about the input video."""

    width: int
    height: int
    fps: float
    total_frames: int
