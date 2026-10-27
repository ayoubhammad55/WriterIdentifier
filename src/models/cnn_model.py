"""
Compact CNN writer-identification model.

Architecture
------------
  Input: (B, 1, H, W) grayscale
  3 × [Conv2d -> BN -> ReLU -> MaxPool2d]
  AdaptiveAvgPool2d -> Flatten
  FC -> ReLU -> Dropout
  FC -> N classes (softmax via cross-entropy loss)

Training utilities
------------------
  train_one_epoch(), validate(), train(), evaluate()
  All accept a device argument (cpu / cuda / mps).
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
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

from utils.constants import (
    BATCH_SIZE,
    CHECKPOINTS_DIR,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    LEARNING_RATE,
    NUM_EPOCHS,
    RANDOM_SEED,
)


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------


def _make_cnn(num_classes: int) -> "nn.Module":
    """Build and return the compact CNN as a Sequential module."""
    import torch.nn as nn

    def _block(in_ch, out_ch, pool=True):
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        return layers

    return nn.Sequential(
        *_block(1,   32),   # -> (32, H/2, W/2)
        *_block(32,  64),   # -> (64, H/4, W/4)
        *_block(64, 128),   # -> (128, H/8, W/8)
        nn.AdaptiveAvgPool2d((4, 4)),          # -> (128, 4, 4)
        nn.Flatten(),                          # -> 2048
        nn.Linear(128 * 4 * 4, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes),
    )


class WriterCNN(nn.Module):
    """Compact CNN for closed-set writer identification."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.net         = _make_cnn(num_classes)

    def forward(self, x):
        return self.net(x)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def get_device() -> "torch.device":
    """Return the best available device (CUDA > MPS > CPU)."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _accuracy(output, target) -> float:
    preds = output.argmax(dim=1)
    return (preds == target).float().mean().item()


def train_one_epoch(
    model,
    loader: "DataLoader",
    criterion,
    optimizer,
    device: "torch.device",
    scaler=None,
) -> dict:
    """Run one training epoch. Returns dict with loss and accuracy."""
    model.train()
    running_loss, running_acc = 0.0, 0.0

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
def validate(
    model,
    loader: "DataLoader",
    criterion,
    device: "torch.device",
) -> dict:
    """Validate the model. Returns dict with loss and accuracy."""
    import torch
    model.eval()
    running_loss, running_acc = 0.0, 0.0

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
    train_loader: "DataLoader",
    val_loader:   "DataLoader",
    num_epochs:   int   = NUM_EPOCHS,
    lr:           float = LEARNING_RATE,
    device:       Optional["torch.device"] = None,
    checkpoint_prefix: str = "cnn",
    use_amp:      bool = False,
) -> dict:
    """
    Full training loop with:
    - CosineAnnealingLR scheduler
    - Best-val-accuracy checkpointing
    - Mixed-precision support (use_amp=True)

    Returns history dict with lists: train_loss, train_acc, val_loss, val_acc.
    """
    import torch

    if device is None:
        device = get_device()

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler    = torch.cuda.amp.GradScaler() if (use_amp and device.type == "cuda") else None

    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    best_ckpt = os.path.join(CHECKPOINTS_DIR, f"{checkpoint_prefix}_best.pth")
    last_ckpt = os.path.join(CHECKPOINTS_DIR, f"{checkpoint_prefix}_last.pth")

    history   = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc  = 0.0
    total_t0  = time.time()

    print(f"[CNN] Training on {device} | {model.parameter_count:,} params")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_metrics   = validate(model, val_loader, criterion, device)

        scheduler.step()

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"] .append(train_metrics["accuracy"])
        history["val_loss"]  .append(val_metrics  ["loss"])
        history["val_acc"]   .append(val_metrics  ["accuracy"])

        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch:3d}/{num_epochs} | "
            f"train loss {train_metrics['loss']:.4f} acc {train_metrics['accuracy']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} acc {val_metrics['accuracy']:.4f} | "
            f"{elapsed:.1f}s"
        )

        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            torch.save(model.state_dict(), best_ckpt)
            print(f"    [OK] New best val acc: {best_acc:.4f} — checkpoint saved")

    # Save final model
    torch.save(model.state_dict(), last_ckpt)
    history["best_val_acc"] = best_acc
    history["total_time_s"] = time.time() - total_t0
    print(f"[CNN] Training complete. Best val acc: {best_acc:.4f}")
    return history


@torch.no_grad()
def evaluate(
    model,
    loader: "DataLoader",
    device: Optional["torch.device"] = None,
    label_names: Optional[list] = None,
) -> dict:
    """
    Evaluate model on a DataLoader.

    Returns accuracy, macro_f1, confusion_matrix, inference_time_per_image.
    """
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

    y_pred = np.array(all_preds)
    y_true = np.array(all_labels)

    acc      = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm       = confusion_matrix(y_true, y_pred)
    report   = classification_report(
        y_true, y_pred,
        target_names=label_names,
        zero_division=0,
    )

    n_images = max(len(y_true), 1)
    return {
        "accuracy":               acc,
        "macro_f1":               macro_f1,
        "confusion_matrix":       cm,
        "report":                 report,
        "inference_time_per_image_s": elapsed / n_images,
    }
