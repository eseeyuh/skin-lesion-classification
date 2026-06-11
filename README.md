# Skin Lesion Classification on HAM10000

Comparative study of three deep-learning architectures — ResNet-50, EfficientNet-B3,
and Vision Transformer (ViT-B/16) — for multi-class skin lesion classification on the
[HAM10000](https://doi.org/10.7910/DVN/DBW86T) dataset, with a soft-voting ensemble.

> **Status: work in progress.** The pipeline lives in the `src/` package and is
> demonstrated end to end in `notebooks/01_skin_lesion_classification.ipynb`.
> CLI entry points and a Streamlit demo are on the roadmap below.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data

The dataset is downloaded from Kaggle and is **not** committed to the repository.
Place your Kaggle API token at `~/.kaggle/kaggle.json`
([how to get one](https://www.kaggle.com/docs/api#authentication)), then run:

```bash
python scripts/download_data.py --data-dir data/ham10000
```

## Library usage

```python
from src import Config, set_seed, get_device, load_metadata, group_split, train_model

cfg = Config.from_yaml("configs/default.yaml")
cfg.make_dirs()
set_seed(cfg.seed)
device = get_device()

df = load_metadata(cfg)
df_train, df_val, df_test = group_split(df, cfg)
result = train_model(cfg, "resnet50", df_train, df_val, df_test, device,
                     lr=cfg.lr_for("resnet50"))
print(result["metrics"]["balanced_accuracy"])
```

Plotting and Grad-CAM helpers live in `src.viz` and `src.interpretability`.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite covers the leakage-free `lesion_id` split, metric correctness and
bounds, and the frozen-backbone model heads (16 tests).

## Layout

```
configs/    experiment configuration (YAML)
data/        dataset (gitignored, created by download_data.py)
notebooks/   exploratory + training notebook
results/     metrics, plots, checkpoints (gitignored)
scripts/     data download and CLI entry points
src/         library code (config, data, models, engine, metrics, ensemble, viz, ...)
tests/       unit tests
```

## Roadmap

- [x] Stabilise the baseline notebook (fp32 evaluation, gradient clipping)
- [x] Repository scaffold, pinned dependencies, Colab-free data download
- [x] Refactor notebook logic into the `src/` package
- [ ] CLI entry points for training and evaluation
- [x] Unit tests (leakage-free split, metrics, model heads)
- [ ] Streamlit demo with Grad-CAM
