import torch

from src.models import build_model, count_trainable_params

NUM_CLASSES = 7
DEVICE = torch.device("cpu")


def test_only_classifier_head_is_trainable():
    model = build_model("resnet50", NUM_CLASSES, DEVICE, pretrained=False)
    head = set(model.get_classifier().parameters())
    for p in model.parameters():
        assert p.requires_grad == (p in head)


def test_output_shape_matches_num_classes():
    model = build_model("resnet50", NUM_CLASSES, DEVICE, pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, NUM_CLASSES)


def test_trainable_params_fewer_than_total():
    model = build_model("resnet50", NUM_CLASSES, DEVICE, pretrained=False)
    trainable, total = count_trainable_params(model)
    assert 0 < trainable < total
