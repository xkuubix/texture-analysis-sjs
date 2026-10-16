import argparse
from pathlib import Path

import yaml


def _load_config(config_path: Path):
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {config_path}")
    return config


def parse_args():
    try:
        default_config = Path(__file__).with_name("config.yml")
    except NameError:
        default_config = Path("config.yml")

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=default_config)
    config_args, _ = config_parser.parse_known_args()
    config = _load_config(config_args.config)

    parser = argparse.ArgumentParser(
        description="Extract PyRadiomics features for a patient dataset."
    )
    parser.add_argument(
        "--config", type=Path, default=config_args.config,
        help="YAML configuration file (default: config.yml)"
    )
    parser.add_argument(
        "--data-root", type=Path, default=config.get("data_root"),
        help="Directory containing patient folders"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=config.get("output_dir"),
        help="Directory for the extracted features"
    )
    parser.add_argument(
        "--mode", choices=("2D", "3D"), default=config.get("mode", None)
    )
    parser.add_argument(
        "--param-file-3d", type=Path,
        default=config.get("param_file_3d", "3d_params.yml")
    )
    parser.add_argument(
        "--param-file-2d", type=Path,
        default=config.get("param_file_2d", "2d_params.yml")
    )
    parser.add_argument(
        "--param-file-adc", type=Path,
        default=config.get("param_file_adc", "2d_params_adc.yml")
    )
    parser.add_argument(
        "--regions", nargs="+", default=config.get("regions"),
        help="Regions to process; defaults to all available regions"
    )
    parser.add_argument(
        "--label", type=int, default=config.get("label", 1),
        help="Mask label to extract"
    )

    parser.add_argument(
        "--correct", action="store_true",
        help="Correct mismatched masks after backing up the originals"
    )
    parser.add_argument(
        "--slice-selection", choices=("largest",),
        default=config.get("slice_selection"),
        help="For 2D mode, select the slice with the largest ROI"
    )

    args, unknown = parser.parse_known_args()

    if args.data_root is None or args.output_dir is None:
        parser.error("data_root and output_dir must be set in config.yml or on the command line")
        
    config_dir = config_args.config.resolve().parent
    for name in ("param_file_3d", "param_file_2d", "param_file_adc"):
        path = getattr(args, name)
        if not path.is_absolute():
            setattr(args, name, config_dir / path)
    return args
