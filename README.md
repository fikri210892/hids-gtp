# HIDS-GTP: Hybrid Intrusion Detection System for GTP Traffic

A multi-stage packet-level attack detection system for GTP-U (GPRS Tunneling Protocol) traffic, combining **CNN-based deep learning** with **Suricata rule-based detection** for comprehensive anomaly detection.

## Overview

This project implements a hybrid detection pipeline that:
- Extracts packet-level features from GTP-U traffic (port 2152)
- Trains a CNN binary classifier to detect anomalies
- Validates detections using Suricata IDS rules
- Combines both approaches using configurable fusion strategies
- Supports holdout-attack evaluation for generalization testing

### Key Features

✅ **Manual GTP Parsing** – Robust extraction from UDP/2152 payload without scapy layer dependencies  
✅ **Temporal Features** – Packet rates, byte rates, inter-arrival times  
✅ **Entropy Analysis** – Payload entropy calculation for anomaly detection  
✅ **Holdout-Attack Design** – Train on attack types A, test on unseen attack type B  
✅ **Hybrid Fusion** – Multiple strategies (OR, AND, CNN-priority, Suricata-priority)  
✅ **Per-Attack Metrics** – Detailed evaluation broken down by attack type  

---

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
    │ CNN Model    │      │ Suricata Labels  │
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

## Step-by-Step Usage

### Step 1: Feature Extraction

Extract packet-level features from PCAP files:

```bash
python3 step1_extract_features_gtp_fixed.py \
    --output training_dataset_gtp_fixed.csv \
    --gtp-port 2152 \
    --rate-window 1.0 \
    --include-fragmented
```

**Parameters:**
- `--output`: Output CSV file path (default: `training_dataset_gtp_fixed.csv`)
- `--gtp-port`: UDP port for GTP traffic (default: 2152)
- `--rate-window`: Temporal window in seconds for rate features (default: 1.0)
- `--include-fragmented`: Include fragmented attack PCAP if available

**Output:**
- `training_dataset_gtp_fixed.csv` – Full dataset with 25 features + metadata
- Console statistics showing packet parsing summary per PCAP

**⚠️ Important:**
- PCAP files must be in current directory or provide full paths
- Do NOT use metadata columns for model training
- Metadata exists for audit and proper train/test splitting

---

### Step 2: CNN Model Training (Holdout-Attack Split)

Train a CNN classifier with strict train/test separation by attack type:

```bash
python3 step2_holdout_attack_train.py \
    --input training_dataset_gtp_fixed.csv \
    --output-dir training_outputs_holdout \
    --normal-train-ratio 0.50 \
    --val-ratio-within-train 0.15 \
    --train-attack-types flood,malformed \
    --test-attack-types invalid_teid,spoofing \
    --epochs 30 \
    --batch-size 256 \
    --learning-rate 0.001 \
    --patience 6 \
    --seed 42
```

**Parameters:**
- `--input`: Input CSV from step 1
- `--output-dir`: Directory for model artifacts (default: `training_outputs_holdout`)
- `--normal-train-ratio`: % of normal traffic for train/val (default: 0.50)
- `--val-ratio-within-train`: % of training data reserved for validation (default: 0.15)
- `--train-attack-types`: Comma-separated attack types for training
- `--test-attack-types`: Comma-separated attack types for testing (unseen during training)
- `--epochs`, `--batch-size`, `--learning-rate`: Hyperparameters
- `--patience`: Early stopping patience (default: 6 epochs)
- `--seed`: Random seed for reproducibility

**Output Files:**
```
training_outputs_holdout/
├── cnn_model.pt                # Trained PyTorch model
├── feature_scaler.joblib       # StandardScaler for feature normalization
├── feature_columns.json        # List of features used for training
├── removed_columns.json        # Columns excluded (leakage/non-numeric)
├── training_history.json       # Loss curves per epoch
├── metrics.json                # Accuracy, precision, recall, F1, confusion matrix
├── test_predictions.csv        # Per-packet predictions on test set
└── confusion_matrix.png        # Confusion matrix visualization
```

**Model Architecture:**
- Input: 1D CNN (25 features)
- Conv layers: 3 (16→32→64 channels) with BatchNorm + ReLU
- Pooling: MaxPool + AdaptiveAvgPool
- Classifier: Fully connected (64→1) with Sigmoid
- Loss: BCEWithLogitsLoss with positive class weighting

**Holdout-Attack Design:**
- **Train**: 50% normal (first half) + all flood/malformed attacks
- **Validation**: 15% tail samples from train partitions
- **Test**: 50% normal (second half) + all invalid_teid/spoofing attacks
- **Purpose**: Evaluate generalization to unseen attack types

---

### Step 3: Suricata Rule-Based Detection

Configure and run Suricata IDS on test PCAP for comparison:

#### 3a. Test Suricata Configuration

```bash
sudo suricata -T -c /etc/suricata/suricata.yaml \
    -S /etc/suricata/rules/gtp-custom.rules
```

#### 3b. Run Suricata on Test PCAP

```bash
rm -rf suricata_out_holdout
mkdir -p suricata_out_holdout

suricata -r /path/to/test_pcap.pcap \
    -c /etc/suricata/suricata.yaml \
    -S /etc/suricata/rules/gtp-custom.rules \
    -l suricata_out_holdout
```

**Note:** Create GTP-specific rules file at `/etc/suricata/rules/gtp-custom.rules` with detection signatures.

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

**Parameters:**
- `--eve`: Path to Suricata eve.json (JSONL alert log)
- `--output`: Output CSV with per-packet labels
- `--stats`: Optional Suricata stats.log to auto-read total packet count
- `--ground-truth`: Optional ground-truth CSV for validation merge
- `--total-packets`: Explicit total packet count (if stats not available)

**Output CSV Columns:**
- `pcap_cnt`: Packet index (1-based)
- `suricata_label`: 0=no alert, 1=alert triggered
- `alert_count`: Number of unique alerts for packet
- `signature_ids`: Alert signature IDs (pipe-separated)
- `signatures`: Alert descriptions
- `severity_max`: Maximum severity level
- `src_ips`, `dst_ips`: Unique source/destination IPs
- `true_label`, `attack_type`, `source_pcap`: (if ground-truth provided)

---

### Step 5: Hybrid CNN + Suricata Fusion

Combine CNN and Suricata predictions using configurable fusion strategies:

```bash
python3 step5_combine_cnn_suricata_hybrid.py \
    --cnn training_outputs_holdout/test_predictions.csv \
    --suricata suricata_labels.csv \
    --output-dir hybrid_results \
    --mode cnn_priority \
    --cnn-low 0.4 \
    --cnn-high 0.6
```

**Parameters:**
- `--cnn`: CNN predictions CSV from step 2
- `--suricata`: Suricata labels CSV from step 4
- `--output-dir`: Directory for hybrid outputs
- `--mode`: Fusion strategy (see below)
- `--cnn-low`, `--cnn-high`: Probability thresholds for cnn_priority mode

**Fusion Modes:**

| Mode | Logic | Use Case |
|------|-------|----------|
| `or` | Detect if CNN=1 **or** Suricata=1 | High sensitivity, catch all alerts |
| `and` | Detect if CNN=1 **and** Suricata=1 | High specificity, mutual confirmation |
| `cnn_priority` | CNN high-confidence overrides Suricata; uncertain cases deferred to Suricata | Default: balance speed & accuracy |
| `suricata_priority` | Any Suricata alert forces detection; CNN for additional signals | Expert rule-trust mode |

**CNN-Priority Logic:**
```
if cnn_prob >= high:        return 1  (confident attack)
if cnn_prob <= low:         return 0  (confident normal)
else:                       return suricata_label  (deferred to rules)
```

**Output Files:**
```
hybrid_results/
├── hybrid_predictions.csv  # Merged predictions from both models
└── metrics.json            # Detailed fusion metrics and per-attack analysis
```

**Output CSV Columns:**
- All CNN features and predictions
- All Suricata fields (signature_ids, alerts, etc.)
- `hybrid_label`: Final fused prediction (0/1)
- `join_method`: How rows were aligned ("pcap_cnt" or "source_pcap+source_seq")

**Metrics JSON Contents:**
```json
{
  "join_method": "pcap_cnt",
  "ground_truth_source": "true_label_cnn",
  "fusion_mode": "cnn_priority",
  "cnn_metrics": { "accuracy": 0.95, "precision": 0.88, ... },
  "suricata_metrics": { "accuracy": 0.92, ... },
  "hybrid_metrics": { "accuracy": 0.97, "f1": 0.94, ... },
  "cnn_per_attack": { "spoofing": { "recall": 0.92, ... }, ... },
  "suricata_per_attack": { ... },
  "hybrid_per_attack": { ... }
}
```

---

## Performance Metrics

### Evaluation Metrics

- **Accuracy**: (TP + TN) / Total
- **Precision**: TP / (TP + FP) – False alarm rate
- **Recall**: TP / (TP + FN) – Detection rate
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)
- **ROC-AUC**: Receiver Operating Characteristic area (CNN only)
- **Confusion Matrix**: TP/TN/FP/FN breakdown

### Per-Attack Metrics

For each attack type, computed against normal traffic:
- Attack-specific recall (% of attack packets detected)
- Attack-specific precision
- Sample counts (attack vs. normal)

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

---

## Data Leakage Prevention

⚠️ **Critical:** The following columns contain information that could leak attack patterns into the training set. **DO NOT use these for model input:**

```
attack_type, source_pcap, direction, pcap_packet_index, 
gtp_packet_index, packet_time, relative_time
```

These are **metadata only** for:
- Dataset audit and reproducibility
- Proper train/test splitting
- Per-attack evaluation

The training scripts automatically exclude these columns. If using the CSV directly, filter them manually:

```python
feature_cols = [c for c in df.columns if c not in LEAKAGE_COLUMNS and c != 'label']
```

---

## Example Workflow

### Full Pipeline (Single Commands)

```bash
# 1. Extract features from PCAP files (in current directory)
python3 step1_extract_features_gtp_fixed.py

# 2. Train CNN with holdout-attack split
python3 step2_holdout_attack_train.py

# 3. Run Suricata on holdout test PCAP
suricata -r /path/to/test_pcap.pcap -c /etc/suricata/suricata.yaml \
    -S /etc/suricata/rules/gtp-custom.rules \
    -l suricata_out_holdout

# 4. Extract Suricata per-packet labels
python3 step4_suricata_packet_labels.py \
    --eve suricata_out_holdout/eve.json \
    --output suricata_labels.csv \
    --stats suricata_out_holdout/stats.log

# 5. Fuse CNN and Suricata predictions
python3 step5_combine_cnn_suricata_hybrid.py \
    --cnn training_outputs_holdout/test_predictions.csv \
    --suricata suricata_labels.csv \
    --output-dir hybrid_results \
    --mode cnn_priority
```

### Expected Outputs

```
training_dataset_gtp_fixed.csv
training_outputs_holdout/
├── cnn_model.pt
├── feature_scaler.joblib
├── metrics.json          # CNN metrics
└── test_predictions.csv
suricata_labels.csv
hybrid_results/
├── hybrid_predictions.csv
└── metrics.json          # Fusion metrics
```

---

## Configuration & Customization

### Adjust Attack Types

Modify the train/test split in `step2_holdout_attack_train.py`:

```bash
python3 step2_holdout_attack_train.py \
    --train-attack-types flood,invalid_teid \
    --test-attack-types malformed,spoofing
```

### Tune CNN Hyperparameters

```bash
python3 step2_holdout_attack_train.py \
    --epochs 50 \
    --batch-size 128 \
    --learning-rate 0.0005 \
    --patience 10
```

### Change Fusion Strategy

```bash
# High sensitivity (catch everything)
python3 step5_combine_cnn_suricata_hybrid.py --mode or

# High specificity (mutual confirmation)
python3 step5_combine_cnn_suricata_hybrid.py --mode and

# Trust Suricata rules more
python3 step5_combine_cnn_suricata_hybrid.py --mode suricata_priority
```

---

## Troubleshooting

### Issue: "PCAP file not found"
**Solution:** Ensure PCAP files are in the current working directory or provide absolute paths in the dataset configuration.

### Issue: "No valid GTP packets found"
**Solution:** Verify PCAP contains UDP traffic on port 2152 with valid GTP headers. Check with:
```bash
tcpdump -r file.pcap 'udp port 2152' | head -20
```

### Issue: "CNN model training very slow"
**Solution:** 
- Use CUDA GPU if available (PyTorch auto-detects)
- Reduce batch size or dataset size
- Decrease epochs or increase learning rate

### Issue: "Suricata eve.json parsing fails"
**Solution:** Verify eve.json format with:
```bash
head -1 eve.json | jq .
```
Ensure each line is valid JSON. Repair malformed logs:
```bash
grep -v '^$' eve.json | while read line; do jq -r . <<< "$line" 2>/dev/null && echo; done > eve_clean.json
```

### Issue: "CNN and Suricata row counts don't match"
**Solution:** 
- Verify both CSVs reference the same PCAP/test data
- Check for missing packets (compare `pcap_cnt` ranges)
- Use `--total-packets` in step 4 if stats.log unavailable

---

## Citation & References

- **GTP Protocol**: 3GPP TS 29.060 (GTP v1-U)
- **CNN Architecture**: Based on 1D convolutional networks for time-series
- **Suricata IDS**: https://suricata.io/
- **PyTorch**: https://pytorch.org/

---

## License

[Specify your license - e.g., MIT, GPL-3.0, etc.]

---

## Contributing

Pull requests and issue reports welcome. Please ensure:
- [ ] Scripts pass without errors on sample PCAP
- [ ] No data leakage columns used for training
- [ ] Metadata columns preserved in outputs
- [ ] Holdout-attack split properly enforced

---

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
