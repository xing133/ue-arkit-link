from dataclasses import dataclass
from pathlib import Path
import argparse


@dataclass
class PipelineConfig:
    input_video: Path = Path("")
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

    # Body tracking
    enable_body: bool = False
    pose_model: str = "heavy"  # lite / full / heavy
    pose_model_path: Path = Path("models/pose_landmarker_heavy.task")

    # Hand tracking
    enable_hands: bool = False
    num_hands: int = 2
    hand_model_path: Path = Path("models/hand_landmarker.task")

    # Streaming mode
    stream_mode: bool = False
    camera_id: int | None = None
    livelink_host: str = "127.0.0.1"
    livelink_port: int = 11111
    livelink_subject: str = "PythonFace"


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(
        description="Facial expression detection with ARKit blendshape output"
    )
    parser.add_argument(
        "input_video", type=Path, nargs="?", default=None,
        help="Path to input video file (optional in --stream --camera mode)",
    )
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

    # Streaming mode
    parser.add_argument(
        "--stream", action="store_true",
        help="Enable streaming mode: send blendshapes to UE5 via Live Link",
    )
    parser.add_argument(
        "--camera", type=int, default=None, metavar="ID",
        help="Camera device ID for live capture (e.g. 0)",
    )
    parser.add_argument("--livelink-host", default="127.0.0.1")
    parser.add_argument("--livelink-port", type=int, default=11111)
    parser.add_argument("--livelink-subject", default="PythonFace")

    # Body & hand tracking
    parser.add_argument(
        "--enable-body", action="store_true",
        help="Enable body pose tracking (33 landmarks)",
    )
    parser.add_argument(
        "--enable-hands", action="store_true",
        help="Enable hand tracking (21 landmarks per hand)",
    )
    parser.add_argument(
        "--pose-model", choices=["lite", "full", "heavy"], default="heavy",
        help="Pose model variant: lite (fast), full (balanced), heavy (accurate, default)",
    )
    parser.add_argument("--num-hands", type=int, default=2, choices=[1, 2])

    args = parser.parse_args()

    # Validate: need either input_video or --camera in stream mode
    if args.stream and args.input_video is None and args.camera is None:
        parser.error("--stream requires either a video file or --camera ID")
    if not args.stream and args.input_video is None:
        parser.error("input_video is required (or use --stream --camera)")

    # Resolve pose model path based on variant
    pose_model_key = f"pose_{args.pose_model}"
    from src.model_manager import MODELS
    pose_filename = MODELS[pose_model_key]["filename"]

    return PipelineConfig(
        input_video=args.input_video or Path(""),
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
        enable_body=args.enable_body,
        pose_model=args.pose_model,
        pose_model_path=Path(f"models/{pose_filename}"),
        enable_hands=args.enable_hands,
        num_hands=args.num_hands,
        hand_model_path=Path("models/hand_landmarker.task"),
        stream_mode=args.stream,
        camera_id=args.camera,
        livelink_host=args.livelink_host,
        livelink_port=args.livelink_port,
        livelink_subject=args.livelink_subject,
    )
