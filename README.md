# HIDS-GTP: Hybrid Intrusion Detection System for GTP Traffic

This repository stores the source code **Implementasi Sistem Deteksi Intrusi Hybrid Untuk Protokol GTP di Jaringan Seluler**
which is a thesis in Universitas Indonesia.

A multi-stage packet-level attack detection system for GTP-U (GPRS Tunneling Protocol) traffic, combining **CNN-based deep learning** with **Suricata rule-based detection** for comprehensive anomaly detection.

## Overview

This project implements a hybrid detection pipeline that:
- Extracts packet-level features from GTP-U traffic (port 2152)
- Trains a CNN binary classifier to detect anomalies
- Validates detections using Suricata IDS rules
- Combines both approaches using configurable fusion strategies
- Supports holdout-attack evaluation for generalization testing

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GTP PCAP Files                           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │ Step 1: Feature Extraction │  (step1_extract_features_gtp_fixed.py)
        │ - Manual GTP parsing       │
        │ - 25 packet-level features │
        │ - Temporal features        │
        └────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  Training Dataset CSV      │
        │  (with metadata)           │
        └────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
    ┌─────────────┐      ┌──────────────────┐
    │ Step 2: CNN │      │ Step 3: Suricata │
    │  Training   │      │ Rule Detection   │
    └─────────────┘      └──────────────────┘
         │                       │
         ▼                       ▼
    ┌──────────────┐      ┌──────────────────┐
    │ CNN Model    │      │ Step 4:Suricata  │
    │              │      │  Labels          │
    │ + Scaler     │      │ (per-packet)     │
    └──────────────┘      └──────────────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
        ┌────────────────────────────┐
        │  Step 5: Hybrid Fusion     │  (step5_combine_cnn_suricata_hybrid.py)
        │  - Merge predictions       │
        │  - Apply fusion strategy   │
        │  - Generate metrics        │
        └────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  Final Detections + Metrics│
        └────────────────────────────┘
```

---

## Dataset Structure

### Input: PCAP Files

Default dataset specifications (configured in scripts):

| PCAP File | Label | Attack Type | Purpose |
|-----------|-------|-------------|---------|
| `normal_traffic.pcap` | 0 | normal | Benign baseline |
| `attack_flood.pcap` | 1 | flood | GTP flooding attacks |
| `attack_invalid_teid.pcap` | 1 | invalid_teid | Invalid TEID exploitation |
| `attack_malformed.pcap` | 1 | malformed | Malformed GTP packets |
| `attack_spoofing.pcap` | 1 | spoofing | Source/TEID spoofing |
| `attack_fragmented.pcap` | 1 | fragmented | Fragmented packets (optional) |

### Output: Feature Set (25 Features)

**Packet-Level Features:**
- `pkt_size`: Total packet length (bytes)
- `ip_len`, `ttl`, `ip_proto`, `frag_offset`, `is_fragment`: IP layer
- `src_port`, `dst_port`, `udp_len`: UDP layer
- `payload_len`, `payload_entropy`: Payload analysis

**GTP-Specific Features:**
- `has_gtp`: Valid GTP header detected (0/1)
- `gtp_flags`, `gtp_version`, `gtp_protocol_type`: GTP header fields
- `gtp_type`, `gtp_length`, `gtp_teid`: GTP message type and tunnel endpoint ID
- `gtp_seq`, `gtp_npdu`, `gtp_next_ex`: Optional GTP extension fields
- `gtp_has_optional_fields`: Extension headers present (0/1)

**Temporal Features (per source PCAP):**
- `pkt_rate`: Packets per second (1-second trailing window)
- `byte_rate`: Bytes per second (1-second trailing window)
- `inter_arrival`: Time since previous UDP/2152 packet (seconds)

**Metadata (not used for training):**
- `label`: Binary target (0=normal, 1=attack)
- `attack_type`, `source_pcap`, `direction`: Audit/grouping
- `pcap_packet_index`, `gtp_packet_index`: Original packet position
- `packet_time`, `relative_time`: Timestamps

---
## Installation & Dependencies

### Requirements

```bash
# Python 3.8+
python3 -m venv venv
source venv/bin/activate

# Core dependencies
pip install numpy pandas scikit-learn

# Deep learning
pip install torch torchvision torchaudio

# Utilities
pip install matplotlib joblib scapy

# Optional: Suricata (system package)
sudo apt-get install suricata
```

### Scapy Setup (if needed)

```bash
pip install scapy

# For PCAP reading with Scapy on Linux
sudo apt-get install libpcap-dev
```

## Step-by-Step Usage

### Step 1: Feature Extraction

Extract packet-level features from PCAP files:

```bash
python3 step1_extract_features_gtp_fixed.py
```

### Step 2: CNN Model Training

Train a CNN classifier with strict train/test separation by attack type:

```bash
python3 step2_holdout_attack_train.py 
```

### Step 3: Suricata Rule-Based Detection

sudo suricata -T -c /etc/suricata/suricata.yaml -S /etc/suricata/rules/gtp-custom.rules

rm -rf /home/ubuntuml/suricata_out_holdout
mkdir -p /home/ubuntuml/suricata_out_holdout
rm -f /home/ubuntuml/suricata_holdout_packet_labels.csv

suricata -r /home/ubuntuml/holdout_test_stream.pcap \
  -c /etc/suricata/suricata.yaml \
  -S /etc/suricata/rules/gtp-custom.rules \
  -l /home/ubuntuml/suricata_out_holdout

---

### Step 4: Convert Suricata Alerts to Per-Packet Labels

Extract per-packet labels from Suricata `eve.json` output:

```bash
python3 step4_suricata_packet_labels.py \
    --eve suricata_out_holdout/eve.json \
    --output suricata_labels.csv \
    --stats suricata_out_holdout/stats.log \
    --ground-truth training_dataset_gtp_fixed.csv
```

### Step 5: Hybrid CNN + Suricata Fusion

Combine CNN and Suricata predictions using configurable fusion strategies:

```bash
step5 combine cnn_suricata
python3 step5_combine_cnn_suricata_hybrid.py \
  --cnn test_predictions_cnn.csv \
  --suricata test_predictions_suricata.csv \
  --output-dir hybrid_outputs_or \
  --mode or
```

## Contact & Support

For issues or questions:
- Open a GitHub issue
- Check existing documentation above
- Review script docstrings for implementation details

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-05-11 | Initial release: manual GTP parsing, CNN + Suricata hybrid detection |

---

**Last Updated:** 2026-05-11  
**Maintained by:** [@fikri210892](https://github.com/fikri210892)
