#!/usr/bin/env python3
"""
step1_extract_features_gtp_fixed.py

Revisi feature extraction packet-level untuk traffic GTP-U.
Perbaikan utama dibanding versi sebelumnya:
- Parsing GTP dilakukan manual dari payload UDP/2152, jadi tidak bergantung pada
  pkt.haslayer(GTPHeader) yang bisa gagal pada beberapa PCAP.
- Menambahkan metadata yang diperlukan untuk audit dataset.
- Menghitung fitur temporal sederhana per source PCAP.
- Tidak melakukan balancing pada tahap extraction.

Catatan penting untuk training:
- Gunakan `label` sebagai target.
- JANGAN masukkan metadata berikut ke model:
  attack_type, source_pcap, direction, pcap_packet_index, gtp_packet_index,
  packet_time, relative_time.
- Lakukan split train/test setelah extraction dengan menjaga source_pcap dan urutan
  paket agar mengurangi leakage.
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scapy.all import IP, UDP, PcapReader  # type: ignore


DEFAULT_SPECS = [
    {"pcap": "normal_traffic.pcap", "label": 0, "attack_type": "normal"},
    {"pcap": "attack_flood.pcap", "label": 1, "attack_type": "flood"},
    {"pcap": "attack_invalid_teid.pcap", "label": 1, "attack_type": "invalid_teid"},
    {"pcap": "attack_malformed.pcap", "label": 1, "attack_type": "malformed"},
    {"pcap": "attack_spoofing.pcap", "label": 1, "attack_type": "spoofing"},
    # Aktifkan bila memang dipakai di eksperimen final.
    # {"pcap": "attack_fragmented.pcap", "label": 1, "attack_type": "fragmented"},
]


class FeatureExtractor:
    def __init__(self, gtp_port: int = 2152, rate_window_seconds: float = 1.0) -> None:
        self.gtp_port = int(gtp_port)
        self.rate_window_seconds = float(rate_window_seconds)
        self.features_list: List[Dict] = []
        self.file_summaries: List[Dict] = []

    @staticmethod
    def calculate_entropy(data: bytes) -> float:
        if not data:
            return 0.0
        byte_counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
        probabilities = byte_counts[byte_counts > 0] / len(data)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return float(entropy)

    def infer_direction(self, pkt) -> str:
        if not pkt.haslayer(UDP):
            return "unknown"

        sport = int(pkt[UDP].sport)
        dport = int(pkt[UDP].dport)

        if dport == self.gtp_port and sport != self.gtp_port:
            return "request"
        if sport == self.gtp_port and dport != self.gtp_port:
            return "response"
        if sport == self.gtp_port and dport == self.gtp_port:
            return "peer_gtp"
        return "unknown"

    @staticmethod
    def parse_gtp_from_udp_payload(payload: bytes) -> Dict[str, int]:
        """Parse minimal GTP header manually from UDP payload.

        Supports common GTPv1-U layout:
        - Octet 1: flags
        - Octet 2: message type
        - Octet 3-4: payload length
        - Octet 5-8: TEID
        - Optional 4 octets follow if any E/S/PN flag is set:
          sequence number (2B), N-PDU (1B), next extension header (1B)

        Returns zeros when payload does not look like a valid GTP header.
        """
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

        # Conservative heuristic: accept only payloads that look like GTP.
        # GTPv1/GTPv2 commonly use version 1 or 2 and protocol type 1.
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

        # Payload bytes from UDP layer
        udp_payload = bytes(pkt[UDP].payload) if pkt.haslayer(UDP) else b""
        features["payload_len"] = int(len(udp_payload))
        features["payload_entropy"] = self.calculate_entropy(udp_payload)

        # Manual GTP parsing
        gtp_info = self.parse_gtp_from_udp_payload(udp_payload)
        features.update(gtp_info)

        # Temporal features filled later
        features["pkt_rate"] = 0.0
        features["byte_rate"] = 0.0
        features["inter_arrival"] = 0.0

        return features

    def extract_from_pcap(self, pcap_file: str, label: int, attack_type: str) -> None:
        pcap_path = Path(pcap_file)
        print(f"[*] Processing {pcap_path.name} | label={label} | attack_type={attack_type}")

        if not pcap_path.exists():
            print(f"[!] File not found: {pcap_path}")
            return

        prev_time: Optional[float] = None
        first_time: Optional[float] = None
        total_packets = 0
        matched_udp2152_packets = 0
        parsed_gtp_packets = 0

        # rolling window for rate features
        rate_window: Deque[Tuple[float, int]] = deque()
        rate_window_total_bytes = 0

        try:
            with PcapReader(str(pcap_path)) as packets:
                for pcap_packet_index, pkt in enumerate(packets, start=1):
                    total_packets += 1

                    if not (
                        pkt.haslayer(UDP)
                        and (int(pkt[UDP].dport) == self.gtp_port or int(pkt[UDP].sport) == self.gtp_port)
                    ):
                        continue

                    matched_udp2152_packets += 1
                    timestamp = float(getattr(pkt, "time", 0.0))
                    if first_time is None:
                        first_time = timestamp

                    features = self.extract_packet_features(pkt)
                    if int(features.get("has_gtp", 0)) == 1:
                        parsed_gtp_packets += 1

                    # inter-arrival per source PCAP among UDP/2152 packets
                    if prev_time is not None:
                        features["inter_arrival"] = max(0.0, timestamp - prev_time)
                    prev_time = timestamp

                    # trailing rates per source PCAP
                    rate_window.append((timestamp, features["pkt_size"]))
                    rate_window_total_bytes += int(features["pkt_size"])
                    while rate_window and (timestamp - rate_window[0][0]) > self.rate_window_seconds:
                        _, old_size = rate_window.popleft()
                        rate_window_total_bytes -= int(old_size)

                    features["pkt_rate"] = len(rate_window) / self.rate_window_seconds
                    features["byte_rate"] = rate_window_total_bytes / self.rate_window_seconds

                    # metadata
                    features["label"] = int(label)
                    features["attack_type"] = attack_type
                    features["source_pcap"] = pcap_path.name
                    features["direction"] = self.infer_direction(pkt)
                    features["pcap_packet_index"] = int(pcap_packet_index)
                    features["gtp_packet_index"] = int(matched_udp2152_packets)
                    features["packet_time"] = float(timestamp)
                    features["relative_time"] = float(timestamp - first_time) if first_time is not None else 0.0

                    self.features_list.append(features)

        except Exception as exc:
            print(f"[!] Error reading {pcap_path}: {exc}")
            return

        self.file_summaries.append(
            {
                "source_pcap": pcap_path.name,
                "attack_type": attack_type,
                "label": int(label),
                "total_packets": total_packets,
                "matched_udp2152_packets": matched_udp2152_packets,
                "parsed_gtp_packets": parsed_gtp_packets,
            }
        )

        print(
            f"[+] UDP/2152 packets: {matched_udp2152_packets} / {total_packets} | "
            f"parsed_gtp: {parsed_gtp_packets}"
        )

    def save_to_csv(self, output_file: str) -> None:
        if not self.features_list:
            print("[!] No features extracted!")
            return

        df = pd.DataFrame(self.features_list)

        feature_columns = [
            "pkt_size",
            "ip_len",
            "ttl",
            "ip_proto",
            "frag_offset",
            "is_fragment",
            "src_port",
            "dst_port",
            "udp_len",
            "payload_len",
            "payload_entropy",
            "has_gtp",
            "gtp_flags",
            "gtp_version",
            "gtp_protocol_type",
            "gtp_type",
            "gtp_length",
            "gtp_teid",
            "gtp_seq",
            "gtp_npdu",
            "gtp_next_ex",
            "gtp_has_optional_fields",
            "pkt_rate",
            "byte_rate",
            "inter_arrival",
        ]

        metadata_columns = [
            "label",
            "attack_type",
            "source_pcap",
            "direction",
            "pcap_packet_index",
            "gtp_packet_index",
            "packet_time",
            "relative_time",
        ]

        ordered_columns = [col for col in feature_columns + metadata_columns if col in df.columns]
        df = df[ordered_columns]
        df.to_csv(output_file, index=False)

        print("\n[*] Checking feature correlations with label...")
        correlations = []
        for col in feature_columns:
            if col in df.columns:
                corr = df[col].corr(df["label"])
                if not np.isnan(corr):
                    correlations.append((col, abs(float(corr))))
        correlations.sort(key=lambda x: x[1], reverse=True)

        print("\nTop 8 feature correlations:")
        for col, corr in correlations[:8]:
            status = "⚠️ SUSPICIOUS" if corr > 0.40 else "✓"
            print(f"  {status} {col:25s}: {corr:.4f}")

        print(f"\n{'=' * 72}")
        print(f"[✓] Features saved to: {output_file}")
        print(f"{'=' * 72}")
        print(f"Total samples:             {len(df)}")
        print(f"Normal (label=0):          {len(df[df['label'] == 0])}")
        print(f"Attack (label=1):          {len(df[df['label'] == 1])}")
        print(f"Parsed has_gtp=1 samples:  {len(df[df['has_gtp'] == 1])}")
        print(f"Feature columns:           {len(feature_columns)}")
        print(f"Metadata columns:          {len(metadata_columns)}")
        print(f"{'=' * 72}")

        print("\n[*] Dataset summary per source_pcap:")
        if self.file_summaries:
            summary_df = pd.DataFrame(self.file_summaries)
            print(summary_df.to_string(index=False))


def build_specs(include_fragmented: bool) -> List[Dict]:
    specs = list(DEFAULT_SPECS)
    if include_fragmented:
        specs.append({"pcap": "attack_fragmented.pcap", "label": 1, "attack_type": "fragmented"})
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract GTP packet-level features from PCAP files.")
    parser.add_argument("--output", default="training_dataset_gtp_fixed.csv", help="Output CSV filename.")
    parser.add_argument(
        "--gtp-port",
        type=int,
        default=2152,
        help="UDP port to treat as GTP traffic (default: 2152).",
    )
    parser.add_argument(
        "--rate-window",
        type=float,
        default=1.0,
        help="Trailing window in seconds for pkt_rate and byte_rate (default: 1.0).",
    )
    parser.add_argument(
        "--include-fragmented",
        action="store_true",
        help="Include attack_fragmented.pcap in extraction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 72)
    print("  GTP Feature Extraction - MANUAL GTP PARSER VERSION")
    print("  (Manual TEID / message type parsing from UDP/2152 payload)")
    print("=" * 72)

    extractor = FeatureExtractor(gtp_port=args.gtp_port, rate_window_seconds=args.rate_window)
    specs = build_specs(include_fragmented=args.include_fragmented)

    for spec in specs:
        extractor.extract_from_pcap(
            pcap_file=spec["pcap"],
            label=int(spec["label"]),
            attack_type=str(spec["attack_type"]),
        )

    extractor.save_to_csv(args.output)

    print("\n[✓] Feature extraction completed!")
    print("[*] Next step: training script harus DROP metadata non-fitur sebelum fitting model.")
    print("[*] Jangan gunakan packet_time / relative_time / packet indexes sebagai input model.")


if __name__ == "__main__":
    main()