import logging
import sys

from src.config import parse_args
from src.pipeline import Pipeline


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    config = parse_args()

    if config.stream_mode:
        from src.stream import StreamPipeline

        if config.camera_id is None and not config.input_video.exists():
            logging.error(f"Input video not found: {config.input_video}")
            sys.exit(1)
        StreamPipeline(config).run()
    else:
        if not config.input_video.exists():
            logging.error(f"Input video not found: {config.input_video}")
            sys.exit(1)
        Pipeline(config).run()


if __name__ == "__main__":
    main()
