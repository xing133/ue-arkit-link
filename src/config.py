from dataclasses import dataclass
from pathlib import Path
import argparse


@dataclass
class PipelineConfig:
    input_video: Path
    output_video: Path | None = None
    output_data: Path | None = None
    model_path: Path = Path("models/face_landmarker_v2_with_blendshapes.task")

    # Detection settings
    num_faces: int = 1
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    # Visualization settings
    draw_tesselation: bool = True
    draw_contours: bool = True
    draw_irises: bool = True
    show_blendshapes: bool = True
    show_depth_map: bool = False
    top_n_blendshapes: int = 10

    # Export settings
    export_format: str = "json"  # "json" or "csv"

    # Preview
    show_preview: bool = True


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(
        description="Facial expression detection with ARKit blendshape output"
    )
    parser.add_argument("input_video", type=Path, help="Path to input video file")
    parser.add_argument("-o", "--output-video", type=Path, default=None)
    parser.add_argument("-d", "--output-data", type=Path, default=None)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/face_landmarker_v2_with_blendshapes.task"),
    )
    parser.add_argument("--num-faces", type=int, default=1)
    parser.add_argument("--no-tesselation", action="store_true")
    parser.add_argument("--no-contours", action="store_true")
    parser.add_argument("--no-irises", action="store_true")
    parser.add_argument("--no-blendshapes", action="store_true")
    parser.add_argument("--show-depth-map", action="store_true")
    parser.add_argument("--top-n-blendshapes", type=int, default=10)
    parser.add_argument(
        "--export-format", choices=["json", "csv"], default="json"
    )
    parser.add_argument(
        "--no-preview", action="store_true", help="Disable real-time preview window"
    )
    parser.add_argument(
        "--min-detection-confidence", type=float, default=0.5
    )
    parser.add_argument(
        "--min-tracking-confidence", type=float, default=0.5
    )

    args = parser.parse_args()

    return PipelineConfig(
        input_video=args.input_video,
        output_video=args.output_video,
        output_data=args.output_data,
        model_path=args.model_path,
        num_faces=args.num_faces,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        draw_tesselation=not args.no_tesselation,
        draw_contours=not args.no_contours,
        draw_irises=not args.no_irises,
        show_blendshapes=not args.no_blendshapes,
        show_depth_map=args.show_depth_map,
        top_n_blendshapes=args.top_n_blendshapes,
        export_format=args.export_format,
        show_preview=not args.no_preview,
    )
