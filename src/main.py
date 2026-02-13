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

    if not config.input_video.exists():
        logging.error(f"Input video not found: {config.input_video}")
        sys.exit(1)

    pipeline = Pipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
