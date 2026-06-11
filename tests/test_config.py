from pathlib import Path

from src.config import Config


def test_from_yaml_loads_defaults():
    cfg = Config.from_yaml("configs/default.yaml")
    assert cfg.num_classes == 7
    assert cfg.batch_size == 32
    assert isinstance(cfg.data_dir, Path)       # coerced from str
    assert isinstance(cfg.model_names, tuple)   # coerced from list


def test_derived_properties():
    cfg = Config()
    assert cfg.num_classes == len(cfg.class_names)
    assert cfg.label2idx["akiec"] == 0
    assert cfg.label2idx["vasc"] == cfg.num_classes - 1
    assert cfg.idx2label[0] == "akiec"
    assert set(cfg.label2idx) == set(cfg.class_names)


def test_lr_for_falls_back_to_default():
    cfg = Config()
    assert cfg.lr_for("efficientnet_b3") == cfg.model_lrs["efficientnet_b3"]
    assert cfg.lr_for("does_not_exist") == cfg.learning_rate


def test_make_dirs_creates_subdirs(tmp_path):
    cfg = Config(output_dir=tmp_path / "out")
    cfg.make_dirs()
    assert cfg.output_dir.is_dir()
    assert cfg.plots_dir.is_dir()
    assert cfg.models_dir.is_dir()
