import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections

from src.types import FrameResult


class FrameVisualizer:
    """Draws detection results onto video frames."""

    def __init__(
        self,
        draw_tesselation: bool = True,
        draw_contours: bool = True,
        draw_irises: bool = True,
        show_blendshapes: bool = True,
        show_depth_map: bool = False,
        top_n_blendshapes: int = 10,
    ):
        self._draw_tess = draw_tesselation
        self._draw_contours = draw_contours
        self._draw_irises = draw_irises
        self._show_bs = show_blendshapes
        self._show_depth = show_depth_map
        self._top_n = top_n_blendshapes

    def annotate_frame(
        self, bgr_frame: np.ndarray, result: FrameResult
    ) -> np.ndarray:
        """Draw all visualizations onto a copy of the frame."""
        annotated = bgr_frame.copy()

        for face_landmarks in result.landmarks:
            lm_list = self._to_normalized_landmarks(face_landmarks)

            if self._draw_tess:
                drawing_utils.draw_landmarks(
                    image=annotated,
                    landmark_list=lm_list,
                    connections=FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style(),
                    is_drawing_landmarks=False,
                )

            if self._draw_contours:
                drawing_utils.draw_landmarks(
                    image=annotated,
                    landmark_list=lm_list,
                    connections=FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style(),
                    is_drawing_landmarks=False,
                )

            if self._draw_irises:
                drawing_utils.draw_landmarks(
                    image=annotated,
                    landmark_list=lm_list,
                    connections=(
                        FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS
                        + FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS
                    ),
                    landmark_drawing_spec=None,
                    connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style(),
                    is_drawing_landmarks=False,
                )

        # Blendshape text overlay (first face only)
        if self._show_bs and result.blendshapes:
            annotated = self._draw_blendshape_overlay(annotated, result.blendshapes[0])

        # Depth sidebar
        if self._show_depth and result.landmarks:
            annotated = self._draw_depth_sidebar(annotated, result.landmarks[0])

        return annotated

    @staticmethod
    def _to_normalized_landmarks(
        landmarks: list[dict[str, float]],
    ) -> list[NormalizedLandmark]:
        """Convert landmark dicts to NormalizedLandmark objects for drawing."""
        return [
            NormalizedLandmark(x=lm["x"], y=lm["y"], z=lm["z"])
            for lm in landmarks
        ]

    def _draw_blendshape_overlay(
        self, frame: np.ndarray, blendshapes: dict[str, float]
    ) -> np.ndarray:
        """Draw top-N blendshape scores in the top-left corner."""
        sorted_bs = sorted(
            [(name, score) for name, score in blendshapes.items() if name != "_neutral"],
            key=lambda x: x[1],
            reverse=True,
        )[: self._top_n]

        y_offset = 30
        for name, score in sorted_bs:
            text = f"{name}: {score:.3f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                frame,
                (8, y_offset - th - 4),
                (12 + tw, y_offset + 4),
                (0, 0, 0),
                cv2.FILLED,
            )
            cv2.putText(
                frame,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            y_offset += 22

        return frame

    def _draw_depth_sidebar(
        self, frame: np.ndarray, landmarks: list[dict[str, float]]
    ) -> np.ndarray:
        """Render a depth map sidebar from landmark Z coordinates."""
        h, w = frame.shape[:2]
        sidebar_w = w // 4
        depth_img = np.zeros((h, sidebar_w), dtype=np.uint8)

        z_values = [lm["z"] for lm in landmarks]
        z_min, z_max = min(z_values), max(z_values)
        z_range = z_max - z_min if z_max != z_min else 1.0

        for lm in landmarks:
            px = int(lm["x"] * sidebar_w)
            py = int(lm["y"] * h)
            # Closer = brighter
            z_norm = int(255 * (1.0 - (lm["z"] - z_min) / z_range))
            if 0 <= px < sidebar_w and 0 <= py < h:
                cv2.circle(depth_img, (px, py), 2, z_norm, -1)

        depth_bgr = cv2.applyColorMap(depth_img, cv2.COLORMAP_INFERNO)
        # Black out background (where depth_img == 0)
        mask = depth_img == 0
        depth_bgr[mask] = [0, 0, 0]

        return np.hstack([frame, depth_bgr])
