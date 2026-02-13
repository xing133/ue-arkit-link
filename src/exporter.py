import csv
import json
from pathlib import Path

import numpy as np

from src.types import FrameResult, VideoMetadata

# 52 ARKit blendshape names in canonical order
ARKIT_BLENDSHAPE_NAMES = [
    "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight",
    "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
    "tongueOut",
]


class DataExporter:
    """Export detection results as JSON or CSV for Blender import."""

    def __init__(self, export_format: str = "json"):
        self._format = export_format
        self._frames: list[FrameResult] = []
        self._metadata: VideoMetadata | None = None

    def set_metadata(self, metadata: VideoMetadata) -> None:
        self._metadata = metadata

    def add_frame(self, result: FrameResult) -> None:
        self._frames.append(result)

    def save(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if self._format == "json":
            self._save_json(output_path)
        elif self._format == "csv":
            self._save_csv(output_path)

    def _save_json(self, path: Path) -> None:
        data = {
            "metadata": {
                "fps": self._metadata.fps if self._metadata else 0,
                "total_frames": len(self._frames),
                "width": self._metadata.width if self._metadata else 0,
                "height": self._metadata.height if self._metadata else 0,
            },
            "blendshape_names": ARKIT_BLENDSHAPE_NAMES,
            "frames": [],
        }

        for fr in self._frames:
            frame_data: dict = {
                "frame_index": fr.frame_index,
                "timestamp_ms": fr.timestamp_ms,
                "faces": [],
            }
            for face_idx in range(len(fr.landmarks)):
                face: dict = {}
                # Blendshapes
                if face_idx < len(fr.blendshapes):
                    face["blendshapes"] = {
                        k: round(v, 6)
                        for k, v in fr.blendshapes[face_idx].items()
                        if k != "_neutral"
                    }
                # 3D landmarks
                face["landmarks_3d"] = [
                    {"x": round(lm["x"], 6), "y": round(lm["y"], 6), "z": round(lm["z"], 6)}
                    for lm in fr.landmarks[face_idx]
                ]
                # Transformation matrix
                if face_idx < len(fr.transformation_matrix) and fr.transformation_matrix[face_idx] is not None:
                    face["transformation_matrix"] = fr.transformation_matrix[face_idx].tolist()

                frame_data["faces"].append(face)

            data["frames"].append(frame_data)

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _save_csv(self, path: Path) -> None:
        """One row per frame. Columns: frame, timestamp_ms, then 52 blendshape scores."""
        header = ["frame", "timestamp_ms"] + ARKIT_BLENDSHAPE_NAMES

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for fr in self._frames:
                if fr.blendshapes:
                    bs = fr.blendshapes[0]  # First face only
                    row = [fr.frame_index, fr.timestamp_ms] + [
                        round(bs.get(name, 0.0), 6) for name in ARKIT_BLENDSHAPE_NAMES
                    ]
                else:
                    row = [fr.frame_index, fr.timestamp_ms] + [0.0] * len(ARKIT_BLENDSHAPE_NAMES)
                writer.writerow(row)
