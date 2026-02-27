"""
Training script for Hybrid Quantum-Driven TB Detection.
Target: 98%+ accuracy on test set.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    DATA_ROOT,
    MODEL_SAVE_DIR,
    RESULTS_DIR,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    LABEL_SMOOTHING,
    EARLY_STOPPING_PATIENCE,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    NUM_WORKERS,
    FREEZE_BACKBONE,
    SEED,
    CLASSES,
    ensure_dirs,
)
from dataset import (
    TBChestXRayDataset,
    get_train_val_test_splits,
    get_transforms,
)
from model import build_model


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """Accuracy, sensitivity, recall (same as sensitivity for positive class), specificity, precision."""
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return {"accuracy": acc, "sensitivity": sens, "recall": sens, "specificity": spec, "precision": prec}


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct, total = 0, 0
    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total if total else 0.0


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / len(loader)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=str(DATA_ROOT), help="Path to dataset root")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--save_dir", type=str, default=str(MODEL_SAVE_DIR))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--freeze_backbone", type=int, default=int(FREEZE_BACKBONE), help="1=freeze backbone, 0=finetune")
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Device: {device} (CUDA available: {torch.cuda.is_available()})")
    if device.type == "cpu":
        # Use most CPU threads (keep 1 core free)
        try:
            torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
        except Exception:
            pass

    data_root = Path(args.data_root)
    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = get_train_val_test_splits(
        data_root, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO, seed=args.seed
    )

    train_ds = TBChestXRayDataset(train_paths, train_labels, transform=get_transforms(True), use_clahe=True)
    val_ds = TBChestXRayDataset(val_paths, val_labels, transform=get_transforms(False), use_clahe=True)
    test_ds = TBChestXRayDataset(test_paths, test_labels, transform=get_transforms(False), use_clahe=True)

    pin = device.type == "cuda"
    nw = max(0, int(args.num_workers))
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=nw, pin_memory=pin, persistent_workers=(nw > 0)
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=nw, pin_memory=pin, persistent_workers=(nw > 0))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=nw, pin_memory=pin, persistent_workers=(nw > 0))

    model = build_model(pretrained=True).to(device)
    # Freeze backbone for massive speedup on CPU (train only the head)
    if hasattr(model, "freeze_backbone") and int(args.freeze_backbone) == 1:
        model.freeze_backbone(True)
        print("Backbone frozen: training only the quantum-inspired head (fast).")
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    save_dir = Path(args.save_dir)
    best_val_acc = 0.0
    patience_left = EARLY_STOPPING_PATIENCE
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_sensitivity": [], "val_specificity": []}

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_sensitivity"].append(val_metrics["sensitivity"])
        history["val_specificity"].append(val_metrics["specificity"])

        print(
            f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} Val Acc: {val_metrics['accuracy']:.4f} "
            f"Sens: {val_metrics['sensitivity']:.4f} Spec: {val_metrics['specificity']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            patience_left = EARLY_STOPPING_PATIENCE
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch, "val_accuracy": best_val_acc},
                save_dir / "best_tb_model.pt",
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best and run on test set (weights_only=False because we saved a full dict)
    ckpt = torch.load(save_dir / "best_tb_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device)
    print("\n--- Test set (best model) ---")
    print(f"Accuracy:    {test_metrics['accuracy']*100:.2f}%")
    print(f"Sensitivity: {test_metrics['sensitivity']*100:.2f}%")
    print(f"Recall:      {test_metrics['recall']*100:.2f}%")
    print(f"Specificity: {test_metrics['specificity']*100:.2f}%")
    print(f"Precision:   {test_metrics['precision']*100:.2f}%")

    with open(RESULTS_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(RESULTS_DIR / "test_metrics.json", "w") as f:
        json.dump({k: float(v) for k, v in test_metrics.items()}, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR}")
    return 0 if test_metrics["accuracy"] >= 0.98 else 1


if __name__ == "__main__":
    sys.exit(main())
