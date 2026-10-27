"""
ResNet-18 transfer-learning pipeline for writer identification.

Grayscale handling
------------------
ResNet-18 expects 3-channel (RGB) input.  We handle grayscale in two ways:
  * ``mode='repeat'``  – replicate the single channel 3 times (cheap, works well)
  * ``mode='adapt'``   – replace the first Conv2d with a 1-channel version and
                         initialise it with the average of the pretrained weights
                         (preferred when images are genuinely grayscale).

Pretrained weights
------------------
Uses ``torchvision.models.ResNet18_Weights.IMAGENET1K_V1`` when available
(torchvision ≥ 0.13).  Falls back to ``pretrained=True`` for older versions.
No internet access is needed if the weights are already cached
(``~/.cache/torch/hub/checkpoints/``).  Pass ``pretrained=False`` to skip.

Usage
-----
    model = WriterResNet18(num_classes=657, mode='adapt', pretrained=True)
    history = train(model, train_loader, val_loader, num_epochs=15)
    metrics = evaluate(model, test_loader)
"""

import os
import time
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    import torchvision.models as tvm
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

from utils.constants import (
    BATCH_SIZE,
    CHECKPOINTS_DIR,
    LEARNING_RATE,
    NUM_EPOCHS,
    RANDOM_SEED,
)


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------


class WriterResNet18(nn.Module):
    """
    ResNet-18 adapted for closed-set writer identification on grayscale images.

    Parameters
    ----------
    num_classes : int   – number of writer classes
    mode        : str   – 'repeat' or 'adapt' (see module docstring)
    pretrained  : bool  – load ImageNet-pretrained weights
    """

    def __init__(
        self,
        num_classes: int,
        mode:        str  = "adapt",
        pretrained:  bool = True,
    ):
        if not _TORCH_OK:
            raise ImportError("PyTorch + torchvision are required for WriterResNet18.")

        super().__init__()
        self.num_classes = num_classes
        self.mode        = mode

        # ── Load backbone ──────────────────────────────────────────────
        backbone = self._load_backbone(pretrained)

        # ── Adapt first conv layer for grayscale ───────────────────────
        if mode == "adapt":
            old_conv = backbone.conv1             # (64, 3, 7, 7)
            new_conv = nn.Conv2d(
                1, 64,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
            if pretrained:
                # Average the 3 RGB channels into 1
                with torch.no_grad():
                    new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
            backbone.conv1 = new_conv
        # elif mode == 'repeat': keep original conv1; repeat channel in forward()

        # ── Replace final FC ───────────────────────────────────────────
        in_features         = backbone.fc.in_features
        backbone.fc         = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes),
        )

        self.backbone = backbone

    # ------------------------------------------------------------------

    @staticmethod
    def _load_backbone(pretrained: bool):
        """Load ResNet-18, compatible with old and new torchvision APIs."""
        try:
            # torchvision ≥ 0.13
            weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            return tvm.resnet18(weights=weights)
        except AttributeError:
            # older torchvision
            return tvm.resnet18(pretrained=pretrained)

    # ------------------------------------------------------------------

    def forward(self, x):
        """x : (B, 1, H, W)"""
        if self.mode == "repeat":
            x = x.repeat(1, 3, 1, 1)           # (B, 3, H, W)
        return self.backbone(x)

    # ------------------------------------------------------------------

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Training helpers  (same interface as cnn_model.py)
# ---------------------------------------------------------------------------


def get_device() -> "torch.device":
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _accuracy(output, target) -> float:
    return (output.argmax(dim=1) == target).float().mean().item()


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None) -> dict:
    model.train()
    running_loss = running_acc = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            import torch
            with torch.cuda.amp.autocast():
                out  = model(images)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out  = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        running_acc  += _accuracy(out, labels)

    n = max(len(loader), 1)
    return {"loss": running_loss / n, "accuracy": running_acc / n}


@torch.no_grad()
def validate(model, loader, criterion, device) -> dict:
    import torch
    model.eval()
    running_loss = running_acc = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        out    = model(images)
        loss   = criterion(out, labels)
        running_loss += loss.item()
        running_acc  += _accuracy(out, labels)

    n = max(len(loader), 1)
    return {"loss": running_loss / n, "accuracy": running_acc / n}


def train(
    model,
    train_loader,
    val_loader,
    num_epochs:        int   = NUM_EPOCHS,
    lr:                float = LEARNING_RATE,
    device:            Optional["torch.device"] = None,
    checkpoint_prefix: str   = "resnet18",
    use_amp:           bool  = False,
    freeze_backbone:   bool  = False,
) -> dict:
    """
    Train WriterResNet18.

    Parameters
    ----------
    freeze_backbone : bool
        If True, freeze all layers except the final FC during the first half
        of training (linear probing), then unfreeze for fine-tuning.
        Useful when the dataset is small.
    """
    import torch

    if device is None:
        device = get_device()

    model = model.to(device)

    if freeze_backbone:
        for name, param in model.backbone.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
        print("[ResNet18] Backbone frozen — linear probing phase.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler    = torch.cuda.amp.GradScaler() if (use_amp and device.type == "cuda") else None

    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    best_ckpt = os.path.join(CHECKPOINTS_DIR, f"{checkpoint_prefix}_best.pth")
    last_ckpt = os.path.join(CHECKPOINTS_DIR, f"{checkpoint_prefix}_last.pth")

    history   = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc  = 0.0
    unfreeze_done = False
    total_t0  = time.time()

    print(
        f"[ResNet18] Training on {device} | "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable params"
    )

    for epoch in range(1, num_epochs + 1):
        # ── Unfreeze backbone at halfway point ─────────────────────────
        if freeze_backbone and not unfreeze_done and epoch > num_epochs // 2:
            for param in model.backbone.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=lr * 0.1, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_epochs - epoch
            )
            unfreeze_done = True
            print(f"[ResNet18] Epoch {epoch}: backbone unfrozen (fine-tuning at lr={lr * 0.1}).")

        t0 = time.time()
        train_m = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_m   = validate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_m["loss"])
        history["train_acc"] .append(train_m["accuracy"])
        history["val_loss"]  .append(val_m  ["loss"])
        history["val_acc"]   .append(val_m  ["accuracy"])

        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch:3d}/{num_epochs} | "
            f"train loss {train_m['loss']:.4f} acc {train_m['accuracy']:.4f} | "
            f"val loss {val_m['loss']:.4f} acc {val_m['accuracy']:.4f} | "
            f"{elapsed:.1f}s"
        )

        if val_m["accuracy"] > best_acc:
            best_acc = val_m["accuracy"]
            torch.save(model.state_dict(), best_ckpt)
            print(f"    [OK] New best val acc: {best_acc:.4f} — checkpoint saved")

    torch.save(model.state_dict(), last_ckpt)
    history["best_val_acc"] = best_acc
    history["total_time_s"] = time.time() - total_t0
    print(f"[ResNet18] Training complete. Best val acc: {best_acc:.4f}")
    return history


@torch.no_grad()
def evaluate(
    model,
    loader,
    device:       Optional["torch.device"] = None,
    label_names:  Optional[list] = None,
) -> dict:
    """Evaluate model. Returns accuracy, macro_f1, confusion_matrix, inference time."""
    import torch
    from sklearn.metrics import (
        accuracy_score, f1_score, confusion_matrix, classification_report
    )

    if device is None:
        device = get_device()

    model.eval()
    model.to(device)

    all_preds, all_labels = [], []
    t0 = time.time()

    for images, labels in loader:
        images = images.to(device)
        out    = model(images)
        preds  = out.argmax(dim=1).cpu().numpy()
        all_preds .extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    elapsed = time.time() - t0
    y_pred  = np.array(all_preds)
    y_true  = np.array(all_labels)
    n       = max(len(y_true), 1)

    return {
        "accuracy":                   accuracy_score(y_true, y_pred),
        "macro_f1":                   f1_score(y_true, y_pred, average="macro", zero_division=0),
        "confusion_matrix":           confusion_matrix(y_true, y_pred),
        "report":                     classification_report(y_true, y_pred, target_names=label_names, zero_division=0),
        "inference_time_per_image_s": elapsed / n,
    }
