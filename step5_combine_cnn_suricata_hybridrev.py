#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def safe_int_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def safe_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def evaluate_binary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    total = len(y_true)
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "support": {
            "normal": int((y_true == 0).sum()),
            "attack": int((y_true == 1).sum()),
            "total": int(total),
        },
    }


def get_label_col(df: pd.DataFrame, preferred: List[str]) -> Optional[str]:
    for c in preferred:
        if c in df.columns:
            return c
    return None


def prepare_cnn(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Standardize core columns
    if "pred_label" in df.columns:
        df["cnn_label"] = safe_int_series(df["pred_label"])
    elif "cnn_label" in df.columns:
        df["cnn_label"] = safe_int_series(df["cnn_label"])
    else:
        raise ValueError("CNN CSV must contain 'pred_label' or 'cnn_label'.")

    prob_col = get_label_col(df, ["pred_prob", "cnn_prob", "probability", "score"])
    if prob_col:
        df["cnn_prob"] = safe_float_series(df[prob_col])
    else:
        df["cnn_prob"] = np.nan

    true_col = get_label_col(df, ["true_label", "label", "y_true"])
    if true_col:
        df["true_label"] = safe_int_series(df[true_col])
    else:
        df["true_label"] = pd.Series([pd.NA] * len(df), dtype="Int64")

    if "attack_type" not in df.columns:
        df["attack_type"] = np.nan
    if "source_pcap" not in df.columns:
        df["source_pcap"] = np.nan

    # Normalize indexing candidates
    if "pcap_cnt" in df.columns:
        df["pcap_cnt"] = safe_int_series(df["pcap_cnt"])

    for c in ["pcap_packet_index", "packet_index", "mixed_packet_index"]:
        if c in df.columns:
            df[c] = safe_int_series(df[c])

    return df


def prepare_suricata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "suricata_label" not in df.columns:
        raise ValueError("Suricata CSV must contain 'suricata_label'.")
    df["suricata_label"] = safe_int_series(df["suricata_label"])

    if "pcap_cnt" in df.columns:
        df["pcap_cnt"] = safe_int_series(df["pcap_cnt"])
    else:
        raise ValueError("Suricata CSV must contain 'pcap_cnt'.")

    if "alert_count" in df.columns:
        df["alert_count"] = safe_int_series(df["alert_count"])
    else:
        df["alert_count"] = pd.Series([0] * len(df), dtype="Int64")

    true_col = get_label_col(df, ["true_label", "label", "y_true"])
    if true_col:
        df["true_label"] = safe_int_series(df[true_col])
    else:
        df["true_label"] = pd.Series([pd.NA] * len(df), dtype="Int64")

    if "attack_type" not in df.columns:
        df["attack_type"] = np.nan
    if "source_pcap" not in df.columns:
        df["source_pcap"] = np.nan

    return df


def direct_join(cnn: pd.DataFrame, suri: pd.DataFrame) -> Optional[pd.DataFrame]:
    if "pcap_cnt" in cnn.columns and cnn["pcap_cnt"].notna().all():
        merged = cnn.merge(
            suri,
            on="pcap_cnt",
            how="inner",
            suffixes=("_cnn", "_suri"),
        )
        if len(merged) == len(cnn) == len(suri):
            merged["join_method"] = "pcap_cnt"
            return merged
    return None


def align_by_source_and_order(cnn: pd.DataFrame, suri: pd.DataFrame) -> pd.DataFrame:
    cnn = cnn.copy()
    suri = suri.copy()

    if cnn["source_pcap"].isna().any():
        raise ValueError("CNN CSV must contain source_pcap for fallback alignment.")
    if suri["source_pcap"].isna().any():
        raise ValueError("Suricata CSV must contain source_pcap for fallback alignment.")

    # Prefer original packet order in CNN if present; otherwise keep file order.
    order_col = None
    for c in ["pcap_packet_index", "packet_index", "mixed_packet_index"]:
        if c in cnn.columns and cnn[c].notna().any():
            order_col = c
            break

    if order_col:
        cnn = cnn.sort_values(["source_pcap", order_col], kind="stable").copy()
    else:
        cnn = cnn.copy()

    suri = suri.sort_values(["source_pcap", "pcap_cnt"], kind="stable").copy()

    cnn["source_seq"] = cnn.groupby("source_pcap").cumcount() + 1
    suri["source_seq"] = suri.groupby("source_pcap").cumcount() + 1

    cnn_counts = cnn.groupby("source_pcap").size().to_dict()
    suri_counts = suri.groupby("source_pcap").size().to_dict()
    if cnn_counts != suri_counts:
        raise ValueError(
            "Per-source row counts do not match between CNN and Suricata CSVs: "
            f"CNN={cnn_counts}, Suricata={suri_counts}"
        )

    merged = cnn.merge(
        suri,
        on=["source_pcap", "source_seq"],
        how="inner",
        suffixes=("_cnn", "_suri"),
    )
    merged["join_method"] = f"source_pcap+source_seq"
    return merged


def build_hybrid_label(
    row: pd.Series,
    mode: str,
    low: float,
    high: float,
    gate_tau: float = 0.90,
) -> int:
    cnn_label = int(row["cnn_label"])
    suri_label = int(row["suricata_label"])
    cnn_prob = row.get("cnn_prob", np.nan)

    if mode == "or":
        # y_hybrid = 1 jika (y_CNN = 1 ∨ y_Suricata = 1)
        return 1 if (cnn_label == 1 or suri_label == 1) else 0

    if mode == "and":
        # y_hybrid = 1 jika (y_CNN = 1 ∧ y_Suricata = 1)
        return 1 if (cnn_label == 1 and suri_label == 1) else 0

    if mode == "confidence_gate":
        # y_Gate = 1 jika (y_Suricata = 1 ∨ p_CNN ≥ τ)
        # Jika p_CNN tidak tersedia, fallback ke label CNN
        if pd.notna(cnn_prob):
            return 1 if (suri_label == 1 or cnn_prob >= gate_tau) else 0
        return 1 if (suri_label == 1 or cnn_label == 1) else 0

    if mode == "cnn_priority":
        if pd.notna(cnn_prob):
            if cnn_prob >= high:
                return 1
            if cnn_prob <= low:
                return 0
            return suri_label
        return cnn_label if cnn_label == 1 else suri_label

    if mode == "suricata_priority":
        return 1 if suri_label == 1 else cnn_label

    raise ValueError(f"Unsupported mode: {mode}")


def per_attack_metrics(df: pd.DataFrame, pred_col: str, label_col: str = "true_label") -> Dict[str, object]:
    out: Dict[str, object] = {}
    if "attack_type" not in df.columns or df["attack_type"].isna().all():
        return out

    normal_mask = df["attack_type"].astype(str).str.lower().eq("normal")
    for atk in sorted(set(df["attack_type"].dropna().astype(str)) - {"normal"}):
        sub = df[normal_mask | df["attack_type"].astype(str).eq(atk)].copy()
        if sub.empty:
            continue
        y_true = safe_int_series(sub[label_col]).fillna(0).astype(int).to_numpy()
        y_pred = safe_int_series(sub[pred_col]).fillna(0).astype(int).to_numpy()
        metrics = evaluate_binary(y_true, y_pred)
        metrics["attack_only_recall"] = (
            ((sub["attack_type"].astype(str).eq(atk)) & (safe_int_series(sub[pred_col]).fillna(0).astype(int).eq(1))).sum()
            / max((sub["attack_type"].astype(str).eq(atk)).sum(), 1)
        )
        metrics["attack_samples"] = int((sub["attack_type"].astype(str).eq(atk)).sum())
        metrics["normal_samples"] = int(normal_mask.loc[sub.index].sum())
        out[atk] = metrics
    return out


def find_ground_truth(merged: pd.DataFrame) -> str:
    candidates = ["true_label_cnn", "true_label_suri", "true_label"]
    available = [c for c in candidates if c in merged.columns]
    if not available:
        raise ValueError("No ground-truth label column found after merge.")

    for c in available:
        if merged[c].notna().any():
            merged["true_label"] = safe_int_series(merged[c])
            return c
    raise ValueError("Ground-truth columns exist but are empty.")


# ---------------------------------------------------------------------------
# Modes yang dijalankan secara sekuensial dalam satu eksekusi
# ---------------------------------------------------------------------------
ALL_MODES = ["or", "and", "confidence_gate"]


def run_all_modes(
    cnn: pd.DataFrame,
    suri: pd.DataFrame,
    merged_base: pd.DataFrame,
    cnn_low: float,
    cnn_high: float,
    gate_tau: float,
) -> tuple:
    """Jalankan ketiga mode secara sekuensial pada merged_base yang sama.

    Returns
    -------
    merged_out : pd.DataFrame
        DataFrame gabungan dengan kolom hybrid_label_or, hybrid_label_and,
        hybrid_label_confidence_gate, dan kolom true_label.
    all_metrics : dict
        Dict berisi metrik CNN & Suricata (sekali) + metrik tiap mode hybrid.
    """
    merged = merged_base.copy()

    y_true = safe_int_series(merged["true_label"]).fillna(0).astype(int).to_numpy()
    y_cnn  = safe_int_series(merged["cnn_label"]).fillna(0).astype(int).to_numpy()
    y_suri = safe_int_series(merged["suricata_label"]).fillna(0).astype(int).to_numpy()

    all_metrics: Dict[str, object] = {
        "join_method": merged["join_method"].iloc[0],
        "row_counts": {
            "cnn": int(len(cnn)),
            "suricata": int(len(suri)),
            "merged": int(len(merged)),
        },
        "cnn_thresholds": {"low": cnn_low, "high": cnn_high},
        "gate_tau": gate_tau,
        # Metrik komponen (dihitung sekali, berlaku untuk semua mode)
        "cnn_metrics": evaluate_binary(y_true, y_cnn),
        "suricata_metrics": evaluate_binary(y_true, y_suri),
        "cnn_per_attack": per_attack_metrics(merged, "cnn_label"),
        "suricata_per_attack": per_attack_metrics(merged, "suricata_label"),
        "hybrid_modes": {},
    }

    for mode in ALL_MODES:
        print(f"[→] Menjalankan mode: {mode} ...")
        col = f"hybrid_label_{mode}"
        merged[col] = merged.apply(
            build_hybrid_label,
            axis=1,
            mode=mode,
            low=cnn_low,
            high=cnn_high,
            gate_tau=gate_tau,
        ).astype(int)

        y_hybrid = safe_int_series(merged[col]).fillna(0).astype(int).to_numpy()
        all_metrics["hybrid_modes"][mode] = {  # type: ignore[index]
            "overall": evaluate_binary(y_true, y_hybrid),
            "per_attack": per_attack_metrics(merged, col),
        }

        f1 = all_metrics["hybrid_modes"][mode]["overall"]["f1"]  # type: ignore[index]
        print(f"    F1={f1:.6f}")

    return merged, all_metrics


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Combine CNN dan Suricata secara sekuensial dengan tiga metode fusion "
            "(OR, AND, Confidence Gate) dan hasilkan satu CSV + satu JSON."
        )
    )
    ap.add_argument("--cnn",        required=True, help="Path ke CNN predictions CSV.")
    ap.add_argument("--suricata",   required=True, help="Path ke Suricata predictions CSV.")
    ap.add_argument("--output-dir", required=True, help="Direktori output.")
    ap.add_argument("--cnn-low",  type=float, default=0.4,  help="Batas bawah probabilitas untuk mode cnn_priority.")
    ap.add_argument("--cnn-high", type=float, default=0.6,  help="Batas atas probabilitas untuk mode cnn_priority.")
    ap.add_argument("--gate-tau", type=float, default=0.90, help="Ambang batas τ untuk confidence_gate (default: 0.90).")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Muat dan merge data (sekali saja)
    # ------------------------------------------------------------------ #
    print("[*] Memuat data CNN dan Suricata ...")
    cnn  = prepare_cnn(pd.read_csv(args.cnn))
    suri = prepare_suricata(pd.read_csv(args.suricata))

    merged_base = direct_join(cnn, suri)
    if merged_base is None:
        merged_base = align_by_source_and_order(cnn, suri)

    gt_source = find_ground_truth(merged_base)
    print(f"[*] Ground-truth source : {gt_source}")
    print(f"[*] Join method         : {merged_base['join_method'].iloc[0]}")
    print(f"[*] Baris merged        : {len(merged_base)}")

    # Harmonisasi metadata
    if "attack_type_cnn" in merged_base.columns and merged_base["attack_type_cnn"].notna().any():
        merged_base["attack_type"] = merged_base["attack_type_cnn"]
    elif "attack_type_suri" in merged_base.columns:
        merged_base["attack_type"] = merged_base["attack_type_suri"]

    if "source_pcap_cnn" in merged_base.columns and merged_base["source_pcap_cnn"].notna().any():
        merged_base["source_pcap"] = merged_base["source_pcap_cnn"]
    elif "source_pcap_suri" in merged_base.columns:
        merged_base["source_pcap"] = merged_base["source_pcap_suri"]

    # ------------------------------------------------------------------ #
    # 2. Jalankan ketiga mode secara sekuensial
    # ------------------------------------------------------------------ #
    merged_final, all_metrics = run_all_modes(
        cnn=cnn,
        suri=suri,
        merged_base=merged_base,
        cnn_low=args.cnn_low,
        cnn_high=args.cnn_high,
        gate_tau=args.gate_tau,
    )
    all_metrics["ground_truth_source"] = gt_source

    # ------------------------------------------------------------------ #
    # 3. Simpan output tunggal
    # ------------------------------------------------------------------ #
    merged_out  = out_dir / "hybrid_predictions_all_modes.csv"
    metrics_out = out_dir / "metrics_all_modes.json"

    merged_final.to_csv(merged_out, index=False)
    metrics_out.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ #
    # 4. Ringkasan ke konsol
    # ------------------------------------------------------------------ #
    print("\n" + "="*60)
    print("[✓] Semua mode selesai dijalankan")
    print(f"[*] CSV output   : {merged_out}")
    print(f"[*] JSON metrics : {metrics_out}")
    print("-"*60)
    print(f"{'Komponen':<22} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}")
    print("-"*60)

    def _row(label, m):
        return (f"{label:<22} "
                f"{m['accuracy']:>8.4f} "
                f"{m['precision']:>8.4f} "
                f"{m['recall']:>8.4f} "
                f"{m['f1']:>8.4f}")

    print(_row("CNN",      all_metrics["cnn_metrics"]))
    print(_row("Suricata", all_metrics["suricata_metrics"]))
    print("-"*60)
    for mode in ALL_MODES:
        label = f"Hybrid ({mode})"
        m = all_metrics["hybrid_modes"][mode]["overall"]
        print(_row(label, m))
    print("="*60)


if __name__ == "__main__":
    main()
