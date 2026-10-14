"""Backbones (all torchvision, all ImageNet-1K pre-trained) and optimiser parameter groups."""
import re

import torch.nn as nn
from torchvision import models as tvm

# name -> (constructor, weights, attribute path of the classification layer)
SPECS = {
    "resnet18":   (tvm.resnet18, tvm.ResNet18_Weights.IMAGENET1K_V1, "fc"),
    "effb0":      (tvm.efficientnet_b0, tvm.EfficientNet_B0_Weights.IMAGENET1K_V1, "classifier.1"),
    "effv2s":     (tvm.efficientnet_v2_s, tvm.EfficientNet_V2_S_Weights.IMAGENET1K_V1, "classifier.1"),
    "convnext_t": (tvm.convnext_tiny, tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1, "classifier.2"),
    "swin_t":     (tvm.swin_t, tvm.Swin_T_Weights.IMAGENET1K_V1, "head"),
    "vit_b16":    (tvm.vit_b_16, tvm.ViT_B_16_Weights.IMAGENET1K_V1, "heads.head"),
    "maxvit_t":   (tvm.maxvit_t, tvm.MaxVit_T_Weights.IMAGENET1K_V1, "classifier.5"),  # hybrid CNN-Transformer
}
DISPLAY = {
    "resnet18": "ResNet-18", "effb0": "EfficientNet-B0", "effv2s": "EfficientNetV2-S",
    "convnext_t": "ConvNeXt-T", "swin_t": "Swin-T", "vit_b16": "ViT-B/16",
    "maxvit_t": "MaxViT-T",
}
TRANSFORMERS = {"swin_t", "vit_b16"}


def uses_channels_last(name):
    """channels_last memory format only for the pure CNNs."""
    return name not in TRANSFORMERS and name != "maxvit_t"


def _get(module, dotted):
    for part in dotted.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def _set(module, dotted, value):
    *parents, last = dotted.split(".")
    parent = _get(module, ".".join(parents)) if parents else module
    if last.isdigit():
        parent[int(last)] = value
    else:
        setattr(parent, last, value)


def build_model(name, num_classes, pretrained=True):
    ctor, weights, head = SPECS[name]
    model = ctor(weights=weights if pretrained else None)
    old = _get(model, head)
    _set(model, head, nn.Linear(old.in_features, num_classes))
    return model


# ------------------------------------------------------------- layer-wise LR decay
def _layer_id(name, param_name):
    """Depth index used for layer-wise learning-rate decay (transformers only)."""
    if name == "vit_b16":
        if param_name.startswith(("conv_proj", "class_token", "encoder.pos_embedding")):
            return 0
        m = re.match(r"encoder\.layers\.encoder_layer_(\d+)\.", param_name)
        if m:
            return int(m.group(1)) + 1
        return 13  # final norm + head
    if name == "swin_t":
        # features: 0 patch-embed, 1/3/5/7 stages (2,2,6,2 blocks), 2/4/6 patch merging
        if param_name.startswith("features.0."):
            return 0
        m = re.match(r"features\.(\d+)\.(\d+)\.", param_name)
        if m:
            stage_idx, block = int(m.group(1)), int(m.group(2))
            offset = {1: 0, 3: 2, 5: 4, 7: 10}
            if stage_idx in offset:
                return offset[stage_idx] + block + 1
        m = re.match(r"features\.(\d+)\.", param_name)
        if m:  # patch merging layers sit between stages
            return {2: 2, 4: 4, 6: 10}[int(m.group(1))]
        return 13
    return None


def num_layers(name):
    return 12 if name in TRANSFORMERS else 0


def param_groups(model, name, lr, weight_decay, llrd=None):
    """AdamW groups: no decay on biases/norms/embeddings; optional layer-wise LR decay."""
    groups = {}
    for pname, p in model.named_parameters():
        if not p.requires_grad:
            continue
        no_decay = (p.ndim <= 1 or pname.endswith((".bias", "class_token", "pos_embedding"))
                    or "relative_position_bias_table" in pname)
        scale = 1.0
        lid = _layer_id(name, pname) if llrd else None
        if lid is not None:
            scale = llrd ** (num_layers(name) + 1 - lid)
        key = (no_decay, round(scale, 8))
        if key not in groups:
            groups[key] = {"params": [], "weight_decay": 0.0 if no_decay else weight_decay,
                           "lr": lr * scale, "lr_scale": scale}
        groups[key]["params"].append(p)
    return list(groups.values())
