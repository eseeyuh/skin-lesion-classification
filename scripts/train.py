"""Train one or all architectures on HAM10000 from the command line.

Each trained model writes a checkpoint to <output_dir>/models/ and a result file
<output_dir>/<model>_result.json (metrics + test predictions) that evaluate.py reads.

Examples:
    python -m scripts.train --model resnet50
    python -m scripts.train --model all --epochs 20
    python scripts/train.py --config configs/default.yaml --model vit_base_patch16_224 --lr 5e-5
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import (Config, set_seed, get_device, describe_device, setup_logging,
                 load_metadata, group_split, train_model)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=Path("configs/default.yaml"),
                   help="Path to a YAML config (default: configs/default.yaml)")
    p.add_argument("--model", default="all",
                   help="Model name or 'all' (default: all)")
    p.add_argument("--lr", type=float, default=None, help="Override the learning rate")
    p.add_argument("--epochs", type=int, default=None, help="Override the number of epochs")
    p.add_argument("--seed", type=int, default=None, help="Override the random seed")
    p.add_argument("--data-dir", type=Path, default=None, help="Override the data directory")
    p.add_argument("--output-dir", type=Path, default=None, help="Override the output directory")
    p.add_argument("--no-pretrained", action="store_true",
                   help="Train from random init instead of ImageNet weights")
    return p.parse_args()


def main():
    args = parse_args()
    logger = setup_logging()

    cfg = Config.from_yaml(args.config) if args.config.exists() else Config()
    if args.seed is not None:
        cfg.seed = args.seed
    if args.epochs is not None:
        cfg.num_epochs = args.epochs
    if args.data_dir is not None:
        cfg.data_dir = args.data_dir
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    cfg.make_dirs()

    if args.model == "all":
        models = list(cfg.model_names)
    elif args.model in cfg.model_names:
        models = [args.model]
    else:
        raise SystemExit(
            f"Unknown model '{args.model}'. Choose from {', '.join(cfg.model_names)} or 'all'.")

    set_seed(cfg.seed)
    device = get_device()
    logger.info("Device: %s", describe_device(device))

    df = load_metadata(cfg)
    df_train, df_val, df_test = group_split(df, cfg)

    for name in models:
        lr = args.lr if args.lr is not None else cfg.lr_for(name)
        result = train_model(cfg, name, df_train, df_val, df_test, device,
                             lr=lr, pretrained=not args.no_pretrained)
        out = cfg.output_dir / f"{name}_result.json"
        with open(out, "w") as f:
            json.dump(result, f)
        logger.info("Saved %s | balanced acc = %.4f",
                    out, result["metrics"]["balanced_accuracy"])


if __name__ == "__main__":
    main()
