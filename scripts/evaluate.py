"""Evaluate trained models: comparison, confusion, calibration, McNemar, ensemble, Grad-CAM.

Reads the per-model result files written by train.py (<output_dir>/<model>_result.json)
and regenerates every figure and summary table into <output_dir>. Misclassified examples
and Grad-CAM additionally need the dataset and checkpoints; they are skipped if absent.

Examples:
    python -m scripts.evaluate
    python scripts/evaluate.py --output-dir results --no-gradcam
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import (Config, set_seed, get_device, setup_logging, load_metadata, group_split,
                 compute_metrics, mcnemar_test, significance_stars, soft_vote)
from src import viz
from src.interpretability import gradcam_for_model


def load_results(cfg):
    results = {}
    for name in cfg.model_names:
        path = cfg.output_dir / f"{name}_result.json"
        if path.exists():
            with open(path) as f:
                results[name] = json.load(f)
    return results


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    p.add_argument("--output-dir", type=Path, default=None, help="Override the output directory")
    p.add_argument("--data-dir", type=Path, default=None, help="Override the data directory")
    p.add_argument("--no-gradcam", action="store_true", help="Skip Grad-CAM generation")
    return p.parse_args()


def main():
    args = parse_args()
    logger = setup_logging()

    cfg = Config.from_yaml(args.config) if args.config.exists() else Config()
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.data_dir is not None:
        cfg.data_dir = args.data_dir
    cfg.make_dirs()

    results = load_results(cfg)
    if not results:
        raise SystemExit(f"No result files found in {cfg.output_dir}. Run train.py first.")
    names = [n for n in cfg.model_names if n in results]
    logger.info("Loaded results for: %s", ", ".join(cfg.model_display[n] for n in names))

    # comparison table, per-class table and figures
    comp_df = viz.build_comparison_table(results, cfg)
    comp_df.to_csv(cfg.output_dir / "comparison_table.csv", index=False)
    viz.plot_model_comparison(comp_df, cfg)
    viz.plot_confusion_matrices(results, cfg)
    pc_df = viz.per_class_table(results, cfg)
    pc_df.to_csv(cfg.output_dir / "per_class.csv", index=False)
    viz.plot_per_class_f1(pc_df, cfg)
    viz.plot_calibration(results, cfg)

    # statistical significance and ensemble (need at least two models)
    ensemble_metrics = None
    if len(names) >= 2:
        labels = np.array(results[names[0]]["test_labels"])
        rows = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                r = mcnemar_test(np.array(results[names[i]]["test_preds"]),
                                 np.array(results[names[j]]["test_preds"]), labels)
                rows.append({"model_a": cfg.model_display[names[i]],
                             "model_b": cfg.model_display[names[j]], **r,
                             "significance": significance_stars(r["pvalue"])})
        pd.DataFrame(rows).to_csv(cfg.output_dir / "mcnemar.csv", index=False)

        probs, preds = soft_vote(results, names)
        ensemble_metrics = compute_metrics(labels, preds, probs, cfg.class_names)
        logger.info("Ensemble balanced accuracy = %.4f", ensemble_metrics["balanced_accuracy"])

    # image-dependent outputs: rebuild the identical test split for paths
    df_test = None
    if (cfg.data_dir / cfg.metadata_csv).exists():
        set_seed(cfg.seed)
        _, _, df_test = group_split(load_metadata(cfg), cfg)
        viz.plot_misclassified(results, df_test, cfg)
        if not args.no_gradcam:
            device = get_device()
            for name in names:
                gradcam_for_model(name, cfg, df_test, results, device)
    else:
        logger.warning("Data not found at %s; skipping misclassified examples and Grad-CAM.",
                       cfg.data_dir)

    best = viz.best_model_name(results)
    summary = {
        "individual_models": {n: results[n]["metrics"] for n in names},
        "ensemble": ensemble_metrics,
        "best_model": cfg.model_display[best],
    }
    with open(cfg.output_dir / "final_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Best model: %s | outputs written to %s", cfg.model_display[best], cfg.output_dir)


if __name__ == "__main__":
    main()
