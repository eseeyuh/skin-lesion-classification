"""Experiment configuration."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

import yaml


@dataclass
class Config:
    # paths
    data_dir: Path = Path("data/ham10000")
    output_dir: Path = Path("results")
    metadata_csv: str = "HAM10000_metadata.csv"

    # data / training
    image_size: int = 224
    batch_size: int = 32
    num_epochs: int = 15
    early_stop_patience: int = 5
    num_workers: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    val_size: float = 0.15
    test_size: float = 0.15
    seed: int = 42

    # models
    model_names: Tuple[str, ...] = ("resnet50", "efficientnet_b3", "vit_base_patch16_224")
    model_lrs: Dict[str, float] = field(default_factory=lambda: {
        "resnet50": 1e-4,
        "efficientnet_b3": 1e-3,
        "vit_base_patch16_224": 1e-4,
    })
    model_display: Dict[str, str] = field(default_factory=lambda: {
        "resnet50": "ResNet-50",
        "efficientnet_b3": "EfficientNet-B3",
        "vit_base_patch16_224": "ViT-B/16",
    })

    # classes
    class_names: Tuple[str, ...] = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
    class_full: Dict[str, str] = field(default_factory=lambda: {
        "akiec": "Actinic keratoses",
        "bcc": "Basal cell carcinoma",
        "bkl": "Benign keratosis",
        "df": "Dermatofibroma",
        "mel": "Melanoma",
        "nv": "Melanocytic nevi",
        "vasc": "Vascular lesions",
    })

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def label2idx(self) -> Dict[str, int]:
        return {c: i for i, c in enumerate(self.class_names)}

    @property
    def idx2label(self) -> Dict[int, str]:
        return {i: c for i, c in enumerate(self.class_names)}

    @property
    def models_dir(self) -> Path:
        return self.output_dir / "models"

    @property
    def plots_dir(self) -> Path:
        return self.output_dir / "plots"

    def lr_for(self, model_name: str) -> float:
        """Learning rate for a given model, falling back to the global default."""
        return self.model_lrs.get(model_name, self.learning_rate)

    def make_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)

    @classmethod
    def from_yaml(cls, path) -> "Config":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        for key in ("data_dir", "output_dir"):
            if key in data:
                data[key] = Path(data[key])
        for key in ("model_names", "class_names"):
            if key in data:
                data[key] = tuple(data[key])
        return cls(**data)
