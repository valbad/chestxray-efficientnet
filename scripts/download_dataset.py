# scripts/download_dataset.py

import yaml
import argparse
from src.data.download import ensure_data_dirs

def main():
    parser = argparse.ArgumentParser(description="Download and prepare dataset.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config YAML"
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    print("📥 Using config:", args.config)
    ensure_data_dirs(cfg["data"])

    print("✨ Dataset ready.")

if __name__ == "__main__":
    main()
