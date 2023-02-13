import argparse
from typing import Any

import yaml

from src.audio.utils.constants import CONFIG_DIR
from src.audio.ASD.utils.asd_pipeline_tools import write_to_terminal


def parse_arguments() -> dict:
    """Parse command line arguments.
    Returns:
        argparse.ArgumentParser: Parsed arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        dest="config",
        default=CONFIG_DIR / "config.yaml",
        help="Path to config file (default: config.yaml).",
    )

    args = parser.parse_args()

    try:
        configs: dict[str, Any] = yaml.safe_load(args.config.read_text())
        write_to_terminal("Loaded config file into python dict!")
    except yaml.YAMLError as exc:
        write_to_terminal("Error in config file!", str(exc))

    return configs