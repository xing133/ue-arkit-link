import logging
import time
from contextlib import ExitStack

import cv2

from src.config import PipelineConfig
from src.detector import FaceDetector, PoseDetector, HandDetector
from src.livelink import LiveLinkSender
from src.model_manager import ensure_model
from src.visualizer import FrameVisualizer

logger = logging.getLogger(__name__)

PREVIEW_WINDOW = "Expression Detect - Stream"


class StreamPipeline:
    """Real-time streaming pipeline: camera/video -> detect -> Live Link -> UE5."""

    def __init__(self, config: PipelineConfig):
        self._config = config

    def run(self) -> None:
        ensure_model(self._config.model_path)
        if self._config.enable_body:
            ensure_model(self._config.pose_model_path, f"pose_{self._config.pose_model}")
        if self._config.enable_hands:
            ensure_model(self._config.hand_model_path, "hand")

        # Open video source
        if self._config.camera_id is not None:
            cap = cv2.VideoCapture(self._config.camera_id)
            source_name = f"camera {self._config.camera_id}"
        else:
            cap = cv2.VideoCapture(str(self._config.input_video))
            source_name = str(self._config.input_video)

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source_name}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Streaming from {source_name} ({width}x{height} @ {fps:.0f} fps)")

        # Live Link sender
        sender = LiveLinkSender(
            host=self._config.livelink_host,
            port=self._config.livelink_port,
            subject_name=self._config.livelink_subject,
            fps=int(fps),
        )

        # Visualizer for local preview
        visualizer = FrameVisualizer(
            draw_tesselation=self._config.draw_tesselation,
            draw_contours=self._config.draw_contours,
            draw_irises=self._config.draw_irises,
            show_blendshapes=self._config.show_blendshapes,
            show_depth_map=self._config.show_depth_map,
            top_n_blendshapes=self._config.top_n_blendshapes,
        )

        if self._config.show_preview:
            cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_NORMAL)

        # Use video file frame delay for playback pacing
        is_camera = self._config.camera_id is not None
        frame_delay_ms = 1 if is_camera else max(1, int(1000 / fps))

        t0 = time.monotonic_ns()
        frame_index = 0

        with ExitStack() as stack:
            face_det = stack.enter_context(FaceDetector(self._config))
            pose_det = stack.enter_context(PoseDetector(self._config)) if self._config.enable_body else None
            hand_det = stack.enter_context(HandDetector(self._config)) if self._config.enable_hands else None

            while True:
                ret, bgr_frame = cap.read()
                if not ret:
                    if is_camera:
                        logger.warning("Camera read failed, retrying...")
                        continue
                    break  # End of video file

                # Monotonically increasing timestamp
                timestamp_ms = (time.monotonic_ns() - t0) // 1_000_000

                result = face_det.detect_frame(bgr_frame, frame_index, timestamp_ms)

                # Body pose detection
                if pose_det:
                    pose_result = pose_det.detect_frame(bgr_frame, timestamp_ms)
                    result.pose_landmarks = pose_result.landmarks
                    result.pose_world_landmarks = pose_result.world_landmarks

                # Hand detection
                if hand_det:
                    hand_result = hand_det.detect_frame(bgr_frame, timestamp_ms)
                    result.hand_landmarks = hand_result.landmarks
                    result.hand_world_landmarks = hand_result.world_landmarks
                    result.handedness = hand_result.handedness

                sender.send_frame(result)

                if self._config.show_preview:
                    annotated = visualizer.annotate_frame(bgr_frame, result)
                    cv2.imshow(PREVIEW_WINDOW, annotated)

                key = cv2.waitKey(frame_delay_ms) & 0xFF
                if key == ord("q"):
                    logger.info("Stream stopped by user (q pressed)")
                    break

                frame_index += 1
                if frame_index % 300 == 0:
                    elapsed = (time.monotonic_ns() - t0) / 1e9
                    actual_fps = frame_index / elapsed if elapsed > 0 else 0
                    logger.info(f"  Streamed {frame_index} frames ({actual_fps:.1f} fps)")

        cap.release()
        sender.close()
        if self._config.show_preview:
            cv2.destroyAllWindows()

        logger.info(f"Stream ended after {frame_index} frames")
