#!/usr/bin/env python3
"""
ivc_proto_dravidian_decoder_class_only.py

This simplified decoder uses the class labels from decoded seals as candidates
to produce non-blank output. It ignores symbol-to-Tamil mappings for now.
"""

import os
import json
import argparse
from collections import namedtuple

# -----------------------------
# Input files
# -----------------------------
COMPARE_JSON = os.path.join('dataset', 'output', 'lexicon_compare_report.json')
SEALS_JSON = os.path.join('dataset', 'output', 'ivc_decoded_seals_advanced.json')

# Output files
OUTPUT_CANDIDATES = 'dataset\\output\\decoded_candidates.json'
OUTPUT_REPORT = 'dataset\\output\\decoding_report.json'

# Named tuple
Candidate = namedtuple('Candidate', ['word', 'score'])

# -----------------------------
# Load JSON
# -----------------------------
def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# -----------------------------
# Extract seals
# -----------------------------
def extract_seals(seals_json):
    results = []
    if isinstance(seals_json, dict):
        for k, v in seals_json.items():
            if not v: continue
            entry = v[0]  # take first sequence
            slots = entry.get('slots') or entry.get('sequence') or []
            results.append({'id': k, 'slots': slots, 'raw': entry})
    return results

# -----------------------------
# Decode seals (class-only)
# -----------------------------
def decode_seal_slots(seal):
    slots = seal.get('slots') or []
    if not slots:
        return []

    hyps = []
    # Each slot has a single candidate (its class)
    words = [Candidate(word=s, score=1.0) for s in slots]
    hyp = {
        'words': [c.word for c in words],
        'score': sum(c.score for c in words),
        'per_slot_meta': [{'slot': s, 'candidate_score': 1.0} for s in slots]
    }
    hyps.append(hyp)
    return hyps

# -----------------------------
# Main
# -----------------------------
def main(args):
    seals_json = load_json(args.seals)
    seals = extract_seals(seals_json)

    all_results = {}
    stats = {'total_seals': len(seals), 'decoded': 0}

    for i, seal in enumerate(seals):
        sid = seal.get('id') or f'seal_{i}'
        hyps = decode_seal_slots(seal)
        all_results[sid] = hyps
        if hyps:
            stats['decoded'] += 1

    # Write output
    with open(args.output_candidates, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    with open(args.output_report, 'w', encoding='utf-8') as f:
        json.dump({'summary': stats}, f, indent=2, ensure_ascii=False)

    print(f"Decoded {stats['decoded']} / {stats['total_seals']} seals. Results -> {args.output_candidates}")

# -----------------------------
# CLI
# -----------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IVC class-only decoder')
    parser.add_argument('--compare', default=COMPARE_JSON)
    parser.add_argument('--seals', default=SEALS_JSON)
    parser.add_argument('--output_candidates', default=OUTPUT_CANDIDATES)
    parser.add_argument('--output_report', default=OUTPUT_REPORT)
    args = parser.parse_args()
    main(args)
