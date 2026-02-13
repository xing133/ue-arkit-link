import logging
from pathlib import Path

import cv2

from src.config import PipelineConfig
from src.detector import FaceDetector
from src.exporter import DataExporter
from src.model_manager import ensure_model
from src.types import VideoMetadata
from src.visualizer import FrameVisualizer

logger = logging.getLogger(__name__)

PREVIEW_WINDOW = "Expression Detect"


class Pipeline:
    """Main processing pipeline: video in -> detection -> visualization + export."""

    def __init__(self, config: PipelineConfig):
        self._config = config

    def run(self) -> None:
        # Ensure model is downloaded
        ensure_model(self._config.model_path)

        # Open input video
        cap = cv2.VideoCapture(str(self._config.input_video))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self._config.input_video}")

        metadata = self._extract_metadata(cap)
        logger.info(
            f"Processing {metadata.total_frames} frames "
            f"at {metadata.fps:.1f} fps ({metadata.width}x{metadata.height})"
        )

        # Output video writer
        output_video_path = self._resolve_output_path("video")
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        out_width = metadata.width
        if self._config.show_depth_map:
            out_width += metadata.width // 4

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_video_path), fourcc, metadata.fps, (out_width, metadata.height)
        )

        # Exporter
        exporter = DataExporter(self._config.export_format)
        exporter.set_metadata(metadata)

        # Visualizer
        visualizer = FrameVisualizer(
            draw_tesselation=self._config.draw_tesselation,
            draw_contours=self._config.draw_contours,
            draw_irises=self._config.draw_irises,
            show_blendshapes=self._config.show_blendshapes,
            show_depth_map=self._config.show_depth_map,
            top_n_blendshapes=self._config.top_n_blendshapes,
        )

        # Preview window
        if self._config.show_preview:
            cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_NORMAL)

        interrupted = False

        with FaceDetector(self._config) as detector:
            frame_index = 0
            while True:
                ret, bgr_frame = cap.read()
                if not ret:
                    break

                timestamp_ms = int(frame_index * 1000 / metadata.fps)

                # Detect
                result = detector.detect_frame(bgr_frame, frame_index, timestamp_ms)

                # Export
                exporter.add_frame(result)

                # Visualize
                annotated = visualizer.annotate_frame(bgr_frame, result)
                writer.write(annotated)

                # Real-time preview
                if self._config.show_preview:
                    cv2.imshow(PREVIEW_WINDOW, annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        logger.info("Preview interrupted by user (q pressed)")
                        interrupted = True
                        break

                frame_index += 1
                if frame_index % 100 == 0:
                    logger.info(f"  Processed {frame_index}/{metadata.total_frames} frames")

        # Cleanup
        cap.release()
        writer.release()
        if self._config.show_preview:
            cv2.destroyAllWindows()

        # Save export data
        data_path = self._resolve_output_path("data")
        exporter.save(data_path)

        if interrupted:
            logger.info(f"Stopped early at frame {frame_index}/{metadata.total_frames}")
        else:
            logger.info(f"Processed all {metadata.total_frames} frames")

        logger.info(f"Output video: {output_video_path}")
        logger.info(f"Output data:  {data_path}")

    def _extract_metadata(self, cap: cv2.VideoCapture) -> VideoMetadata:
        return VideoMetadata(
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=cap.get(cv2.CAP_PROP_FPS),
            total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )

    def _resolve_output_path(self, kind: str) -> Path:
        stem = self._config.input_video.stem
        if kind == "video":
            if self._config.output_video:
                return self._config.output_video
            return Path("output") / f"{stem}_annotated.mp4"
        else:
            if self._config.output_data:
                return self._config.output_data
            ext = "json" if self._config.export_format == "json" else "csv"
            return Path("output") / f"{stem}_blendshapes.{ext}"
