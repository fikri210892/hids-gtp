#!/usr/bin/env python3
"""
step2_cnn_training.py

This avoids evaluating on the same attack types used for training and is more
appropriate for a generalization / unseen-attack experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

LEAKAGE_COLUMNS = [
    "attack_type",
    "source_pcap",
    "direction",
    "pcap_packet_index",
    "gtp_packet_index",
    "packet_time",
    "relative_time",
]
TARGET_COLUMN = "label"
ORDER_COLUMNS = ["pcap_packet_index", "packet_time", "gtp_packet_index"]
DEFAULT_TRAIN_ATTACKS = ["flood", "malformed"]
DEFAULT_TEST_ATTACKS = ["invalid_teid", "spoofing"]


@dataclass
class SplitData:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame


class PacketCNN(nn.Module):
    def __init__(self, num_features: int) -> None:
        super().__init__()
        if num_features < 4:
            raise ValueError(f"Need at least 4 model features, got {num_features}")

        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.15),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x)).squeeze(1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    except Exception:
        pass


def parse_csv_list(value: str) -> List[str]:
    items = [x.strip() for x in value.split(",") if x.strip()]
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN with holdout attack-type split")
    parser.add_argument("--input", default="training_dataset_gtp_fixed.csv", help="Input CSV dataset")
    parser.add_argument("--output-dir", default="training_outputs_holdout", help="Output directory")
    parser.add_argument(
        "--normal-train-ratio",
        type=float,
        default=0.50,
        help="Fraction of normal traffic reserved for train/val; remainder goes to test",
    )
    parser.add_argument(
        "--val-ratio-within-train",
        type=float,
        default=0.15,
        help="Validation ratio taken from the tail of each train portion",
    )
    parser.add_argument(
        "--train-attack-types",
        type=str,
        default=",".join(DEFAULT_TRAIN_ATTACKS),
        help="Comma-separated attack_type names used for train/val",
    )
    parser.add_argument(
        "--test-attack-types",
        type=str,
        default=",".join(DEFAULT_TEST_ATTACKS),
        help="Comma-separated attack_type names used only for test",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {TARGET_COLUMN, "attack_type", "source_pcap"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def sort_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ORDER_COLUMNS if c in df.columns]
    if cols:
        return df.sort_values(cols).reset_index(drop=True)
    return df.reset_index(drop=True)


def split_train_val_from_source(source_df: pd.DataFrame, val_ratio_within_train: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    source_df = sort_df(source_df)
    if not (0.0 <= val_ratio_within_train < 0.5):
        raise ValueError("val_ratio_within_train must be in [0.0, 0.5)")
    if len(source_df) < 2 or val_ratio_within_train == 0:
        return source_df.copy(), source_df.iloc[0:0].copy()

    val_size = max(1, int(math.floor(len(source_df) * val_ratio_within_train)))
    if len(source_df) - val_size < 1:
        val_size = len(source_df) - 1
    train_part = source_df.iloc[:-val_size].copy()
    val_part = source_df.iloc[-val_size:].copy()
    return train_part, val_part


def build_holdout_split(
    df: pd.DataFrame,
    normal_train_ratio: float,
    val_ratio_within_train: float,
    train_attack_types: List[str],
    test_attack_types: List[str],
) -> SplitData:
    if not (0.1 <= normal_train_ratio < 0.9):
        raise ValueError("normal_train_ratio must be in [0.1, 0.9)")

    train_set = set(train_attack_types)
    test_set = set(test_attack_types)
    overlap = sorted(train_set.intersection(test_set))
    if overlap:
        raise ValueError(f"Attack types overlap between train and test: {overlap}")

    known_types = sorted(df["attack_type"].dropna().unique().tolist())
    unknown_train = sorted(train_set.difference(known_types))
    unknown_test = sorted(test_set.difference(known_types))
    if unknown_train:
        raise ValueError(f"Unknown train attack_type values: {unknown_train}")
    if unknown_test:
        raise ValueError(f"Unknown test attack_type values: {unknown_test}")

    normal_df = sort_df(df[df["attack_type"] == "normal"].copy())
    if len(normal_df) < 2:
        raise ValueError("Need at least 2 normal rows for train/test split")

    normal_train_end = max(1, int(math.floor(len(normal_df) * normal_train_ratio)))
    if normal_train_end >= len(normal_df):
        normal_train_end = len(normal_df) - 1

    normal_train_all = normal_df.iloc[:normal_train_end].copy()
    normal_test = normal_df.iloc[normal_train_end:].copy()
    normal_train, normal_val = split_train_val_from_source(normal_train_all, val_ratio_within_train)

    train_parts = [normal_train]
    val_parts = [normal_val]
    test_parts = [normal_test]

    for attack_name in train_attack_types:
        attack_df = df[df["attack_type"] == attack_name].copy()
        attack_train, attack_val = split_train_val_from_source(attack_df, val_ratio_within_train)
        train_parts.append(attack_train)
        val_parts.append(attack_val)

    for attack_name in test_attack_types:
        attack_df = sort_df(df[df["attack_type"] == attack_name].copy())
        test_parts.append(attack_df)

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    return SplitData(train_df=train_df, val_df=val_df, test_df=test_df)


def choose_model_features(train_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    removed: List[str] = []
    feature_candidates = train_df.drop(columns=[TARGET_COLUMN], errors="ignore")

    for col in LEAKAGE_COLUMNS:
        if col in feature_candidates.columns:
            feature_candidates = feature_candidates.drop(columns=[col])
            removed.append(col)

    non_numeric = [c for c in feature_candidates.columns if not pd.api.types.is_numeric_dtype(feature_candidates[c])]
    if non_numeric:
        feature_candidates = feature_candidates.drop(columns=non_numeric)
        removed.extend(non_numeric)

    constant_cols = [c for c in feature_candidates.columns if feature_candidates[c].nunique(dropna=False) <= 1]
    if constant_cols:
        feature_candidates = feature_candidates.drop(columns=constant_cols)
        removed.extend(constant_cols)

    selected = feature_candidates.columns.tolist()
    if not selected:
        raise ValueError("No usable numeric features left after cleanup")
    return selected, removed


def make_tensors(
    df: pd.DataFrame,
    feature_cols: List[str],
    scaler: StandardScaler | None = None,
    fit_scaler: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, StandardScaler | None]:
    x = df[feature_cols].astype(np.float32).to_numpy()
    y = df[TARGET_COLUMN].astype(np.float32).to_numpy()

    if fit_scaler:
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
    elif scaler is not None:
        x = scaler.transform(x)

    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    return x_tensor, y_tensor, scaler


def make_loader(x: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle, num_workers=0)


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return metrics


def run_inference(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(yb.numpy())
    return np.concatenate(labels), np.concatenate(probs)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    patience: int,
    pos_weight: float,
) -> Tuple[nn.Module, List[Dict[str, float]]]:
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history: List[Dict[str, float]] = []
    best_state = None
    best_val_loss = float("inf")
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        epoch_info: Dict[str, float] = {"epoch": float(epoch), "train_loss": float(train_loss)}

        if val_loader is not None and len(val_loader.dataset) > 0:
            model.eval()
            val_total = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    val_total += loss.item() * xb.size(0)
            val_loss = val_total / len(val_loader.dataset)
            epoch_info["val_loss"] = float(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    history.append(epoch_info)
                    print(f"[!] Early stopping at epoch {epoch}")
                    break
        else:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        history.append(epoch_info)
        if "val_loss" in epoch_info:
            print(f"Epoch {epoch:02d} | train_loss={epoch_info['train_loss']:.5f} | val_loss={epoch_info['val_loss']:.5f}")
        else:
            print(f"Epoch {epoch:02d} | train_loss={epoch_info['train_loss']:.5f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def save_confusion_matrix(cm: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal (0)", "Attack (1)"])
    ax.set_yticklabels(["Normal (0)", "Attack (1)"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def per_attack_metrics(test_df: pd.DataFrame, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    test_copy = test_df.copy()
    test_copy["pred_prob"] = y_prob
    test_copy["pred_label"] = (test_copy["pred_prob"] >= threshold).astype(int)

    normal_subset = test_copy[test_copy["attack_type"] == "normal"]
    attack_types = [x for x in test_copy["attack_type"].dropna().unique().tolist() if x != "normal"]
    for attack_name in sorted(attack_types):
        attack_subset = test_copy[test_copy["attack_type"] == attack_name]
        comparison = pd.concat([normal_subset, attack_subset], ignore_index=True)
        metrics = evaluate_predictions(
            comparison[TARGET_COLUMN].to_numpy(),
            comparison["pred_prob"].to_numpy(),
            threshold=threshold,
        )
        metrics["attack_only_recall"] = float((attack_subset["pred_label"] == 1).mean()) if len(attack_subset) else 0.0
        metrics["attack_samples"] = int(len(attack_subset))
        metrics["normal_samples"] = int(len(normal_subset))
        result[attack_name] = metrics
    return result


def print_split_summary(split_data: SplitData) -> None:
    print("\n[*] Split summary:")
    for name, part in [("train", split_data.train_df), ("val", split_data.val_df), ("test", split_data.test_df)]:
        print(f"  - {name:5s}: {len(part):6d} rows | label_counts={part[TARGET_COLUMN].value_counts().to_dict()}")
        print(f"            attack_counts={part['attack_type'].value_counts().to_dict()}")
        print(f"            source_counts={part['source_pcap'].value_counts().to_dict()}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = Path.cwd() / input_path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_attack_types = parse_csv_list(args.train_attack_types)
    test_attack_types = parse_csv_list(args.test_attack_types)

    print("=" * 72)
    print("  Holdout Attack-Type CNN Training")
    print("=" * 72)
    print(f"[*] Input dataset       : {input_path}")
    print(f"[*] Output dir         : {output_dir}")
    print(f"[*] Normal train ratio : {args.normal_train_ratio}")
    print(f"[*] Train attack types : {train_attack_types}")
    print(f"[*] Test attack types  : {test_attack_types}")

    df = load_dataset(input_path)
    print(f"[*] Loaded rows        : {len(df)}")
    print(f"[*] Columns            : {len(df.columns)}")

    split_data = build_holdout_split(
        df=df,
        normal_train_ratio=args.normal_train_ratio,
        val_ratio_within_train=args.val_ratio_within_train,
        train_attack_types=train_attack_types,
        test_attack_types=test_attack_types,
    )
    print_split_summary(split_data)

    feature_cols, removed_cols = choose_model_features(split_data.train_df)
    print("\n[*] Selected model features:")
    print("  ", feature_cols)
    print("\n[*] Removed columns (leakage/non-numeric/constant):")
    print("  ", removed_cols)

    x_train, y_train, scaler = make_tensors(split_data.train_df, feature_cols, fit_scaler=True)
    x_val, y_val, _ = make_tensors(split_data.val_df, feature_cols, scaler=scaler)
    x_test, y_test, _ = make_tensors(split_data.test_df, feature_cols, scaler=scaler)

    train_loader = make_loader(x_train, y_train, args.batch_size, shuffle=True)
    val_loader = make_loader(x_val, y_val, args.batch_size, shuffle=False) if len(y_val) else None
    test_loader = make_loader(x_test, y_test, args.batch_size, shuffle=False)

    train_pos = float(y_train.sum().item())
    train_neg = float(len(y_train) - train_pos)
    pos_weight = train_neg / max(train_pos, 1.0)
    print(f"\n[*] Training class balance: neg={train_neg:.0f}, pos={train_pos:.0f}, pos_weight={pos_weight:.4f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")
    model = PacketCNN(num_features=len(feature_cols)).to(device)

    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
        pos_weight=pos_weight,
    )

    y_true, y_prob = run_inference(model, test_loader, device)
    y_pred = (y_prob >= 0.5).astype(int)
    overall_metrics = evaluate_predictions(y_true, y_prob, threshold=0.5)
    report = classification_report(y_true, y_pred, digits=4, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    attack_breakdown = per_attack_metrics(split_data.test_df, y_prob, threshold=0.5)

    print("\n" + "=" * 72)
    print("[✓] Overall test metrics")
    print("=" * 72)
    for k, v in overall_metrics.items():
        print(f"  {k:10s}: {v:.6f}")

    print("\n[✓] Per-attack metrics (against normal test subset):")
    for attack_name, metrics in attack_breakdown.items():
        print(
            f"  - {attack_name}: " + ", ".join(
                f"{mk}={mv:.4f}" if isinstance(mv, float) else f"{mk}={mv}"
                for mk, mv in metrics.items()
            )
        )

    torch.save(model.state_dict(), output_dir / "cnn_model.pt")
    joblib.dump(scaler, output_dir / "feature_scaler.joblib")

    with open(output_dir / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)
    with open(output_dir / "removed_columns.json", "w", encoding="utf-8") as f:
        json.dump(removed_cols, f, indent=2)
    with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    metrics_payload = {
        "split_scheme": {
            "normal_train_ratio": args.normal_train_ratio,
            "train_attack_types": train_attack_types,
            "test_attack_types": test_attack_types,
            "val_ratio_within_train": args.val_ratio_within_train,
        },
        "overall_metrics": overall_metrics,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "per_attack_metrics": attack_breakdown,
        "split_sizes": {
            "train": int(len(split_data.train_df)),
            "val": int(len(split_data.val_df)),
            "test": int(len(split_data.test_df)),
        },
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    pred_df = split_data.test_df.copy().reset_index(drop=True)
    pred_df["pred_prob"] = y_prob
    pred_df["pred_label"] = y_pred
    pred_df.to_csv(output_dir / "test_predictions.csv", index=False)

    save_confusion_matrix(cm, output_dir / "confusion_matrix.png")

    print("\n[✓] Saved artifacts:")
    for item in [
        "cnn_model.pt",
        "feature_scaler.joblib",
        "feature_columns.json",
        "removed_columns.json",
        "training_history.json",
        "metrics.json",
        "test_predictions.csv",
        "confusion_matrix.png",
    ]:
        print(f"  - {output_dir / item}")


if __name__ == "__main__":
    main()
