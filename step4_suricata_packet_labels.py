#!/usr/bin/env python3
import argparse, json, csv
from collections import defaultdict
from pathlib import Path
import re


def read_total_packets_from_stats(path: Path):
    text = path.read_text(encoding='utf-8', errors='ignore')
    m = re.search(r'decoder\.pkts\s*\|\s*Total\s*\|\s*(\d+)', text)
    if m:
        return int(m.group(1))
    return None


def load_alerts(eve_path: Path):
    by_pcap = defaultdict(lambda: {
        'alert_count': 0,
        'signature_ids': set(),
        'signatures': set(),
        'severity_max': None,
        'src_ips': set(),
        'dst_ips': set(),
    })
    seen = set()  # dedupe identical duplicate alert rows by (pcap_cnt, sid, timestamp, src, dst)

    with eve_path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get('event_type') != 'alert':
                continue
            pcap_cnt = obj.get('pcap_cnt')
            alert = obj.get('alert', {}) or {}
            sid = alert.get('signature_id')
            sig = alert.get('signature', '')
            ts = obj.get('timestamp', '')
            src = obj.get('src_ip', '')
            dst = obj.get('dest_ip', '')
            dedupe_key = (pcap_cnt, sid, ts, src, dst)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            if pcap_cnt is None:
                continue
            row = by_pcap[int(pcap_cnt)]
            row['alert_count'] += 1
            if sid is not None:
                row['signature_ids'].add(int(sid))
            if sig:
                row['signatures'].add(sig)
            sev = alert.get('severity')
            if sev is not None:
                row['severity_max'] = sev if row['severity_max'] is None else max(row['severity_max'], sev)
            if src:
                row['src_ips'].add(src)
            if dst:
                row['dst_ips'].add(dst)
    return by_pcap


def load_ground_truth(path: Path):
    gt = {}
    if not path:
        return gt
    import csv
    with path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        idx_col = None
        for cand in ('mixed_packet_index', 'packet_index', 'pcap_cnt'):
            if cand in reader.fieldnames:
                idx_col = cand
                break
        if not idx_col:
            raise ValueError('Ground truth CSV must have one of: mixed_packet_index, packet_index, pcap_cnt')
        for row in reader:
            try:
                idx = int(row[idx_col])
            except Exception:
                continue
            gt[idx] = row
    return gt


def main():
    ap = argparse.ArgumentParser(description='Convert Suricata eve.json alerts into per-packet CSV labels')
    ap.add_argument('--eve', required=True, help='Path to Suricata eve.json')
    ap.add_argument('--output', required=True, help='Output CSV path')
    ap.add_argument('--total-packets', type=int, default=None, help='Total packet count in the PCAP')
    ap.add_argument('--stats', default=None, help='Optional Suricata stats.log to auto-read total packets')
    ap.add_argument('--ground-truth', default=None, help='Optional ground truth CSV to append attack_type/true_label')
    args = ap.parse_args()

    eve_path = Path(args.eve)
    out_path = Path(args.output)
    total = args.total_packets
    if total is None and args.stats:
        total = read_total_packets_from_stats(Path(args.stats))
    alerts = load_alerts(eve_path)
    if total is None:
        total = max(alerts.keys()) if alerts else 0
    gt = load_ground_truth(Path(args.ground_truth)) if args.ground_truth else {}

    fieldnames = [
        'pcap_cnt', 'suricata_label', 'alert_count', 'signature_ids', 'signatures', 'severity_max',
        'src_ips', 'dst_ips'
    ]
    if gt:
        fieldnames += ['true_label', 'attack_type', 'source_pcap']

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i in range(1, total + 1):
            a = alerts.get(i)
            row = {
                'pcap_cnt': i,
                'suricata_label': 1 if a else 0,
                'alert_count': a['alert_count'] if a else 0,
                'signature_ids': '|'.join(map(str, sorted(a['signature_ids']))) if a else '',
                'signatures': '|'.join(sorted(a['signatures'])) if a else '',
                'severity_max': a['severity_max'] if a and a['severity_max'] is not None else '',
                'src_ips': '|'.join(sorted(a['src_ips'])) if a else '',
                'dst_ips': '|'.join(sorted(a['dst_ips'])) if a else '',
            }
            if gt:
                g = gt.get(i, {})
                row['true_label'] = g.get('label', '')
                row['attack_type'] = g.get('attack_type', '')
                row['source_pcap'] = g.get('source_pcap', '')
            w.writerow(row)

    print(f'[✓] Wrote per-packet Suricata labels to: {out_path}')
    print(f'[*] Total packets: {total}')
    print(f'[*] Unique alerted packets: {len(alerts)}')


if __name__ == '__main__':
    main()