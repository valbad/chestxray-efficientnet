import argparse
import yaml
import os
import pandas as pd
from src.training.train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="YAML config file")
    parser.add_argument("--train_dir", required=True,
                        help="Directory containing the training data (images + train.csv)")
    parser.add_argument("--history_file", type=str,
                        default="outputs/history/training_history.csv",
                        help="Where to save training metrics")

    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Inject override
    cfg["data"]["train_dir"] = args.train_dir

    # Check validity
    if cfg["data"]["train_dir"] is None:
        raise ValueError("train_dir is required but missing")

    # Run training
    history = train(cfg)

    # Save history as CSV
    os.makedirs(os.path.dirname(args.history_file), exist_ok=True)
    df_hist = pd.DataFrame(history)
    df_hist["epoch"] = range(1, len(df_hist) + 1)
    df_hist.to_csv(args.history_file, index=False)

    print(f"📊 Training history saved to: {args.history_file}")
