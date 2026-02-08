import argparse
from pathlib import Path

from focusguard.config import load_yaml
from focusguard.realtime.loop import run_camera_loop


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=["camera"],
        help="Run camera preview"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to config YAML"
    )
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))

    if args.command == "camera":
        run_camera_loop(cfg)


if __name__ == "__main__":
    main()
