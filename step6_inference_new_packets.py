#!/usr/bin/env python3
"""
step6_inference_new_packets.py

Run inference on new/unseen PCAP files using trained CNN model.
This script loads the trained model from step2 and applies it to new packet captures.

Features:
- Load pre-trained CNN model and scaler
- Extract features from new PCAP files
- Run batch inference
- Output predictions ready for hybrid fusion with Suricata

Usage:
    python3 step6_inference_new_packets.py \\
        --pcap new_traffic.pcap \\
        --model-dir training_outputs_holdout \\
        --output-dir inference_outputs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import torch
from scapy.all import IP, UDP, PcapReader  # type: ignore

# Constants
GTP_PORT = 2152


class FeatureExtractor:
    """Extract features from PCAP for inference."""

    def __init__(self, gtp_port: int = 2152) -> None:
        self.gtp_port = int(gtp_port)

    @staticmethod
    def calculate_entropy(data: bytes) -> float:
        if not data:
            return 0.0
        byte_counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
        probabilities = byte_counts[byte_counts > 0] / len(data)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return float(entropy)

    @staticmethod
    def parse_gtp_from_udp_payload(payload: bytes) -> Dict[str, int]:
        """Parse minimal GTP header manually from UDP payload."""
        result = {
            "has_gtp": 0,
            "gtp_flags": 0,
            "gtp_version": 0,
            "gtp_protocol_type": 0,
            "gtp_type": 0,
            "gtp_length": 0,
            "gtp_teid": 0,
            "gtp_seq": 0,
            "gtp_npdu": 0,
            "gtp_next_ex": 0,
            "gtp_has_optional_fields": 0,
        }

        if len(payload) < 8:
            return result

        flags = int(payload[0])
        version = (flags >> 5) & 0x07
        protocol_type = (flags >> 4) & 0x01
        has_optional_fields = 1 if (flags & 0x07) != 0 else 0
        message_type = int(payload[1])
        gtp_length = int.from_bytes(payload[2:4], byteorder="big", signed=False)
        teid = int.from_bytes(payload[4:8], byteorder="big", signed=False)

        if version not in (1, 2):
            return result
        if protocol_type not in (0, 1):
            return result
        if gtp_length < 0:
            return result

        result.update(
            {
                "has_gtp": 1,
                "gtp_flags": flags,
                "gtp_version": version,
                "gtp_protocol_type": protocol_type,
                "gtp_type": message_type,
                "gtp_length": gtp_length,
                "gtp_teid": teid,
                "gtp_has_optional_fields": has_optional_fields,
            }
        )

        if has_optional_fields and len(payload) >= 12:
            result["gtp_seq"] = int.from_bytes(payload[8:10], byteorder="big", signed=False)
            result["gtp_npdu"] = int(payload[10])
            result["gtp_next_ex"] = int(payload[11])

        return result

    def extract_packet_features(self, pkt) -> Dict:
        """Extract features from single packet."""
        features: Dict = {}

        # Basic packet features
        features["pkt_size"] = int(len(pkt))
        features["ip_len"] = int(pkt[IP].len) if pkt.haslayer(IP) and getattr(pkt[IP], "len", None) is not None else 0
        features["ttl"] = int(pkt[IP].ttl) if pkt.haslayer(IP) and getattr(pkt[IP], "ttl", None) is not None else 0
        features["ip_proto"] = int(pkt[IP].proto) if pkt.haslayer(IP) and getattr(pkt[IP], "proto", None) is not None else 0
        features["frag_offset"] = int(pkt[IP].frag) if pkt.haslayer(IP) and getattr(pkt[IP], "frag", None) is not None else 0
        features["is_fragment"] = 1 if pkt.haslayer(IP) and int(getattr(pkt[IP], "frag", 0)) > 0 else 0

        # UDP features
        features["src_port"] = int(pkt[UDP].sport) if pkt.haslayer(UDP) else 0
        features["dst_port"] = int(pkt[UDP].dport) if pkt.haslayer(UDP) else 0
        features["udp_len"] = int(pkt[UDP].len) if pkt.haslayer(UDP) and getattr(pkt[UDP], "len", None) is not None else 0

        # Payload
        udp_payload = bytes(pkt[UDP].payload) if pkt.haslayer(UDP) else b""
        features["payload_len"] = int(len(udp_payload))
        features["payload_entropy"] = self.calculate_entropy(udp_payload)

        # GTP parsing
        gtp_info = self.parse_gtp_from_udp_payload(udp_payload)
        features.update(gtp_info)

        # Temporal features (filled later)
        features["pkt_rate"] = 0.0
        features["byte_rate"] = 0.0
        features["inter_arrival"] = 0.0

        return features

    def extract_from_pcap(self, pcap_file: str) -> List[Dict]:
        """Extract features from PCAP file."""
        pcap_path = Path(pcap_file)
        print(f"[*] Extracting features from: {pcap_path.name}")

        if not pcap_path.exists():
            print(f"[!] File not found: {pcap_path}")
            return []

        features_list: List[Dict] = []
        total_packets = 0
        matched_udp2152 = 0

        try:
            with PcapReader(str(pcap_path)) as packets:
                for pcap_idx, pkt in enumerate(packets, start=1):
                    total_packets += 1

                    if not (
                        pkt.haslayer(UDP)
                        and (int(pkt[UDP].dport) == self.gtp_port or int(pkt[UDP].sport) == self.gtp_port)
                    ):
                        continue

                    matched_udp2152 += 1
                    timestamp = float(getattr(pkt, "time", 0.0))

                    features = self.extract_packet_features(pkt)
                    features["pcap_packet_index"] = int(pcap_idx)
                    features["gtp_packet_index"] = int(matched_udp2152)
                    features["packet_time"] = float(timestamp)

                    features_list.append(features)

        except Exception as e:
            print(f"[!] Error reading {pcap_path}: {e}")
            return []

        print(f"[+] Extracted {matched_udp2152} / {total_packets} packets")
        return features_list


def load_model_artifacts(model_dir: Path) -> tuple:
    """Load trained model, scaler, and feature columns."""
    print(f"[*] Loading model artifacts from: {model_dir}")

    model_path = model_dir / "cnn_model.pt"
    scaler_path = model_dir / "feature_scaler.joblib"
    features_path = model_dir / "feature_columns.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"Feature columns not found: {features_path}")

    # Load feature columns
    with open(features_path, "r") as f:
        feature_cols = json.load(f)
    print(f"[✓] Feature columns: {len(feature_cols)} features")

    # Load scaler
    scaler = joblib.load(scaler_path)
    print("[✓] Feature scaler loaded")

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")

    # Simple 1D CNN model
    class PacketCNN(torch.nn.Module):
        def __init__(self, num_features: int):
            super().__init__()
            self.features = torch.nn.Sequential(
                torch.nn.Conv1d(1, 16, kernel_size=3, padding=1),
                torch.nn.BatchNorm1d(16),
                torch.nn.ReLU(),
                torch.nn.Conv1d(16, 32, kernel_size=3, padding=1),
                torch.nn.BatchNorm1d(32),
                torch.nn.ReLU(),
                torch.nn.MaxPool1d(kernel_size=2, stride=2),
                torch.nn.Dropout(0.15),
                torch.nn.Conv1d(32, 64, kernel_size=3, padding=1),
                torch.nn.BatchNorm1d(64),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool1d(4),
            )
            self.classifier = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(64 * 4, 64),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.20),
                torch.nn.Linear(64, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.classifier(self.features(x)).squeeze(1)

    model = PacketCNN(num_features=len(feature_cols)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("[✓] Model loaded")

    return model, scaler, feature_cols, device


def run_inference(
    model,
    scaler,
    feature_cols: List[str],
    df: pd.DataFrame,
    device,
    batch_size: int = 256
) -> np.ndarray:
    """Run inference on features."""
    print(f"[*] Running inference on {len(df)} packets...")

    x = df[feature_cols].astype(np.float32).to_numpy()
    x = scaler.transform(x)

    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1)

    probs = []
    with torch.no_grad():
        for i in range(0, len(x_tensor), batch_size):
            batch = x_tensor[i : i + batch_size].to(device)
            logits = model(batch)
            batch_probs = torch.sigmoid(logits).cpu().numpy()
            probs.append(batch_probs)

    return np.concatenate(probs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on new PCAP files")
    parser.add_argument("--pcap", required=True, help="Path to PCAP file for inference")
    parser.add_argument("--model-dir", required=True, help="Directory with trained model artifacts")
    parser.add_argument("--output-dir", default="inference_outputs", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for inference")
    return parser.parse_args()


def main():
    args = parse_args()

    pcap_path = Path(args.pcap)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)

    print("=" * 72)
    print("  GTP IDS Inference on New Packets")
    print("=" * 72)

    if not pcap_path.exists():
        print(f"[!] PCAP not found: {pcap_path}")
        return

    if not model_dir.exists():
        print(f"[!] Model directory not found: {model_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model artifacts
    model, scaler, feature_cols, device = load_model_artifacts(model_dir)

    # Extract features
    extractor = FeatureExtractor(gtp_port=GTP_PORT)
    features_list = extractor.extract_from_pcap(str(pcap_path))

    if not features_list:
        print("[!] No features extracted from PCAP")
        return

    df = pd.DataFrame(features_list)
    print(f"[*] Total packets: {len(df)}")

    # Run inference
    y_prob = run_inference(model, scaler, feature_cols, df, device, args.batch_size)
    y_pred = (y_prob >= 0.5).astype(int)

    df["pred_prob"] = y_prob
    df["pred_label"] = y_pred

    # Save predictions
    output_csv = output_dir / "inference_predictions.csv"
    df.to_csv(output_csv, index=False)

    print("\n" + "=" * 72)
    print("[✓] Inference completed")
    print(f"[*] Predictions saved: {output_csv}")
    print(f"[*] Total detections (pred_label=1): {(y_pred == 1).sum()}")
    print(f"[*] Normal packets (pred_label=0): {(y_pred == 0).sum()}")
    print(f"[*] Average confidence: {y_prob.mean():.4f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
