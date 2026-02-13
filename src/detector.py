import cv2
import numpy as np
import mediapipe as mp

from src.types import FrameResult
from src.config import PipelineConfig

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


class FaceDetector:
    """Wraps MediaPipe FaceLandmarker for video-mode detection."""

    def __init__(self, config: PipelineConfig):
        self._config = config
        self._landmarker: FaceLandmarker | None = None

    def __enter__(self) -> "FaceDetector":
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(self._config.model_path)
            ),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=self._config.num_faces,
            min_face_detection_confidence=self._config.min_detection_confidence,
            min_face_presence_confidence=self._config.min_presence_confidence,
            min_tracking_confidence=self._config.min_tracking_confidence,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)
        return self

    def __exit__(self, *args) -> None:
        if self._landmarker:
            self._landmarker.close()
            self._landmarker = None

    def detect_frame(
        self,
        bgr_frame: np.ndarray,
        frame_index: int,
        timestamp_ms: int,
    ) -> FrameResult:
        """Run face detection on a single BGR frame. Timestamp must be monotonically increasing."""
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        return self._convert_result(result, frame_index, timestamp_ms)

    def _convert_result(
        self, result, frame_index: int, timestamp_ms: int
    ) -> FrameResult:
        landmarks_all: list[list[dict[str, float]]] = []
        blendshapes_all: list[dict[str, float]] = []
        matrices_all: list[np.ndarray | None] = []

        for face_idx in range(len(result.face_landmarks)):
            # 468 landmarks with x, y, z
            face_lms = [
                {"x": lm.x, "y": lm.y, "z": lm.z}
                for lm in result.face_landmarks[face_idx]
            ]
            landmarks_all.append(face_lms)

            # 52 ARKit blendshapes + _neutral
            if result.face_blendshapes and face_idx < len(result.face_blendshapes):
                bs_dict = {
                    cat.category_name: cat.score
                    for cat in result.face_blendshapes[face_idx]
                }
                blendshapes_all.append(bs_dict)

            # 4x4 transformation matrix
            if (
                result.facial_transformation_matrixes
                and face_idx < len(result.facial_transformation_matrixes)
            ):
                mat = np.array(result.facial_transformation_matrixes[face_idx])
                matrices_all.append(mat)
            else:
                matrices_all.append(None)

        return FrameResult(
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            landmarks=landmarks_all,
            blendshapes=blendshapes_all,
            transformation_matrix=matrices_all,
        )
