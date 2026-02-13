from dataclasses import dataclass

import cv2
import numpy as np
import mediapipe as mp

from src.types import FrameResult
from src.config import PipelineConfig

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


@dataclass
class PoseResult:
    landmarks: list[dict[str, float]] | None = None
    world_landmarks: list[dict[str, float]] | None = None


@dataclass
class HandResult:
    landmarks: list[list[dict[str, float]]] | None = None
    world_landmarks: list[list[dict[str, float]]] | None = None
    handedness: list[str] | None = None


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


class PoseDetector:
    """Wraps MediaPipe PoseLandmarker for video-mode detection."""

    def __init__(self, config: PipelineConfig):
        self._config = config
        self._landmarker: PoseLandmarker | None = None

    def __enter__(self) -> "PoseDetector":
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(self._config.pose_model_path)
            ),
            running_mode=VisionRunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=self._config.min_detection_confidence,
            min_tracking_confidence=self._config.min_tracking_confidence,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)
        return self

    def __exit__(self, *args) -> None:
        if self._landmarker:
            self._landmarker.close()
            self._landmarker = None

    def detect_frame(
        self, bgr_frame: np.ndarray, timestamp_ms: int
    ) -> PoseResult:
        """Run pose detection on a single BGR frame."""
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.pose_landmarks:
            return PoseResult()

        # First person only
        lms = [
            {"x": lm.x, "y": lm.y, "z": lm.z}
            for lm in result.pose_landmarks[0]
        ]
        world_lms = None
        if result.pose_world_landmarks:
            world_lms = [
                {"x": lm.x, "y": lm.y, "z": lm.z}
                for lm in result.pose_world_landmarks[0]
            ]
        return PoseResult(landmarks=lms, world_landmarks=world_lms)


class HandDetector:
    """Wraps MediaPipe HandLandmarker for video-mode detection."""

    def __init__(self, config: PipelineConfig):
        self._config = config
        self._landmarker: HandLandmarker | None = None

    def __enter__(self) -> "HandDetector":
        options = HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(self._config.hand_model_path)
            ),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=self._config.num_hands,
            min_hand_detection_confidence=self._config.min_detection_confidence,
            min_tracking_confidence=self._config.min_tracking_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        return self

    def __exit__(self, *args) -> None:
        if self._landmarker:
            self._landmarker.close()
            self._landmarker = None

    def detect_frame(
        self, bgr_frame: np.ndarray, timestamp_ms: int
    ) -> HandResult:
        """Run hand detection on a single BGR frame."""
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.hand_landmarks:
            return HandResult()

        all_lms: list[list[dict[str, float]]] = []
        all_world: list[list[dict[str, float]]] = []
        all_handedness: list[str] = []

        for i in range(len(result.hand_landmarks)):
            lms = [
                {"x": lm.x, "y": lm.y, "z": lm.z}
                for lm in result.hand_landmarks[i]
            ]
            all_lms.append(lms)

            if result.hand_world_landmarks and i < len(result.hand_world_landmarks):
                world = [
                    {"x": lm.x, "y": lm.y, "z": lm.z}
                    for lm in result.hand_world_landmarks[i]
                ]
                all_world.append(world)

            if result.handedness and i < len(result.handedness):
                all_handedness.append(result.handedness[i][0].category_name)

        return HandResult(
            landmarks=all_lms,
            world_landmarks=all_world if all_world else None,
            handedness=all_handedness if all_handedness else None,
        )
