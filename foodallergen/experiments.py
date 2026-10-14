"""Experiment registry. Every run is fully described here; nothing is set by hand.

Common recipe (identical for every backbone; only the learning rate is tuned, with the
same 4-value grid and the same budget for each model, on the validation set):
  AdamW (wd 0.05, no decay on norms/biases), linear warm-up 2 epochs + cosine decay,
  20 epochs, effective batch 64, bf16 autocast, label smoothing 0.1, class-weighted loss,
  RandomResizedCrop + flip + RandAugment(2, 9) + RandomErasing(0.25), 224 px input,
  ImageNet-1K pre-trained torchvision weights, model selection by validation accuracy.
  Transformers additionally use layer-wise LR decay 0.75 (standard ViT/Swin fine-tuning).
"""
import json

from . import paths

MODELS = ["resnet18", "effb0", "effv2s", "convnext_t", "swin_t", "vit_b16"]
LR_GRID = [3e-5, 1e-4, 3e-4, 1e-3]
SEEDS = [1, 2, 3]
SWEEP_SEED = 0
SWEEP_EPOCHS = 5
LLRD = {"swin_t": 0.75, "vit_b16": 0.75}
SELECTED_LR_FILE = paths.REPO_DATA / "selected_lr.json"
ABLATION_MODEL = "effb0"

# Each ablation variant removes one component of the common recipe.
ABLATIONS = {
    "no_class_weights": dict(class_weights=False),
    "no_label_smoothing": dict(label_smoothing=0.0),
    "adam_constant_lr": dict(optimizer="adam", schedule="constant", warmup_epochs=0),
    "basic_augmentation": dict(aug="basic"),
    "v1_recipe": dict(class_weights=False, label_smoothing=0.0, optimizer="adam",
                      schedule="constant", warmup_epochs=0, aug="basic"),
}


def base_cfg(model):
    return dict(model=model, llrd=LLRD.get(model))


def selected_lr():
    return json.loads(SELECTED_LR_FILE.read_text())


def sweep_runs():
    for m in MODELS:
        for lr in LR_GRID:
            cfg = dict(base_cfg(m), lr=lr, seed=SWEEP_SEED, epochs=SWEEP_EPOCHS,
                       warmup_epochs=1, eval_test=False, keep_best_weights=False)
            yield paths.RUNS / "sweep" / m / f"lr{lr:g}", cfg


def main_runs():
    lrs = selected_lr()
    for seed in SEEDS:  # seed-major order: one full comparison is available early
        for m in MODELS:
            yield paths.RUNS / "main" / m / f"seed{seed}", dict(base_cfg(m), lr=lrs[m], seed=seed)


def ablation_runs():
    lr = selected_lr()[ABLATION_MODEL]
    for seed in SEEDS:
        for name, change in ABLATIONS.items():
            cfg = dict(base_cfg(ABLATION_MODEL), lr=lr, seed=seed, **change)
            yield paths.RUNS / "ablation" / name / f"seed{seed}", cfg


FINAL_CHOICE_FILE = paths.REPO_DATA / "final_choice.json"
FINAL = dict(img_size=320, epochs=30, warmup_epochs=2, image_dir="images_448")


def final_runs():
    """Deployed model: the backbone with the best mean VALIDATION accuracy in `main`
    (scripts/select_final.py; the test set plays no part in the choice), retrained at
    320 px for 30 epochs with the same recipe and learning rate. Reported separately
    from the controlled comparison."""
    choice = json.loads(FINAL_CHOICE_FILE.read_text())
    m = choice["model"]
    extra = dict(FINAL)
    if m == "vit_b16":  # torchvision ViT-B/16 weights are tied to 224 px position embeddings
        extra["img_size"] = 224
    for seed in SEEDS:
        yield paths.RUNS / "final" / m / f"seed{seed}", dict(base_cfg(m), lr=choice["lr"], seed=seed, **extra)


def pilot_runs():
    yield paths.RUNS / "pilot" / "effb0", dict(base_cfg("effb0"), lr=1e-3, seed=0, epochs=1,
                                               warmup_epochs=1, eval_test=False,
                                               keep_best_weights=False)


EXPERIMENTS = {"pilot": pilot_runs, "sweep": sweep_runs, "main": main_runs,
               "ablation": ablation_runs, "final": final_runs}
