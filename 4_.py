#!/usr/bin/env python3
"""
compare.py

Compare two lexicon_summary.json files (IVC vs. Old Tamil)
using multiple similarity metrics:
- entropy similarity
- label order similarity
- semantic overlap
- structural similarity of seal arrangements (multi-label per seal)
- SVO syntactic pattern similarity

Outputs a detailed report and overall similarity score.
"""

import json
import os
import math
from collections import Counter
from typing import Dict, List

OUT_DIR_DEFAULT = os.path.join('dataset', 'output')
IVC_LEXICON_JSON = os.path.join(OUT_DIR_DEFAULT, "IVC",'lexicon_summary.json')
TAMIL_LEXICON_JSON = os.path.join(OUT_DIR_DEFAULT, "pali",'lexicon_summary.json')
REPORT_JSON = os.path.join(OUT_DIR_DEFAULT, 'lexicon_compare_report.json')

# Map labels to SVO roles
ROLE_MAP = {
    "proper_name": "S",   # Subject
    "title": "V",         # Verb / Action
    "commodity": "O"      # Object
}


def load_lexicon(path: str) -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Lexicon JSON not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_entropy(counts: Dict[str, int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)


def compare_entropies(ent1: float, ent2: float) -> float:
    if ent1 == 0 and ent2 == 0:
        return 1.0
    return 1.0 - abs(ent1 - ent2) / max(ent1, ent2)


def label_distribution(lexicon: Dict) -> List[str]:
    """Flatten all labels in order for order similarity."""
    seq = []
    for cls, reps in lexicon.get("representatives", {}).items():
        for rep in reps:
            # If multi-label sequence exists, flatten it
            label_seq = rep.get("label_sequence", [rep.get("assigned_label")])
            seq.extend(label_seq)
    return seq


def semantic_overlap(lex1: Dict, lex2: Dict) -> float:
    """Compute overlap of classes assigned to sequences."""
    classes1 = set(lex1.get("class_counts", {}).keys())
    classes2 = set(lex2.get("class_counts", {}).keys())
    if not classes1 and not classes2:
        return 1.0
    return len(classes1.intersection(classes2)) / max(len(classes1.union(classes2)), 1)


def extract_seal_structures(lexicon: Dict) -> List[List[str]]:
    """Extract multi-label sequences for each seal."""
    seals = []
    for cls, reps in lexicon.get("representatives", {}).items():
        for rep in reps:
            label_seq = rep.get("label_sequence", [rep.get("assigned_label")])
            seals.append(label_seq)
    return seals


def structural_similarity(seals1: List[List[str]], seals2: List[List[str]]) -> float:
    """Compare two sets of seal label sequences using Dice coefficient."""
    if not seals1 or not seals2:
        return 0.0

    def pattern_counts(seals: List[List[str]]) -> Counter:
        return Counter(tuple(s) for s in seals)

    count1 = pattern_counts(seals1)
    count2 = pattern_counts(seals2)
    all_patterns = set(count1.keys()).union(count2.keys())

    similarity_sum = 0.0
    for p in all_patterns:
        f1 = count1.get(p, 0)
        f2 = count2.get(p, 0)
        if f1 + f2 == 0:
            sim = 1.0
        else:
            sim = 2 * min(f1, f2) / (f1 + f2)
        similarity_sum += sim

    return similarity_sum / len(all_patterns)


def compute_cosine_similarity(counter1: Counter, counter2: Counter) -> float:
    all_keys = set(counter1.keys()).union(counter2.keys())
    v1 = [counter1.get(k, 0) for k in all_keys]
    v2 = [counter2.get(k, 0) for k in all_keys]
    dot = sum(a*b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a*a for a in v1))
    norm2 = math.sqrt(sum(b*b for b in v2))
    return dot / (norm1 * norm2) if norm1 and norm2 else 0.0


# --- SVO Pattern Analysis ---
def seal_to_svo_pattern(seal_labels: List[str]) -> str:
    """Convert seal's multi-label sequence to SVO pattern."""
    return ''.join(ROLE_MAP.get(label, '?') for label in seal_labels)


def analyze_svo_patterns(lexicon: Dict) -> Counter:
    """Return counts of SVO patterns across all seals."""
    patterns = []
    for cls, reps in lexicon.get("representatives", {}).items():
        for rep in reps:
            label_seq = rep.get("label_sequence", [rep.get("assigned_label")])
            patterns.append(seal_to_svo_pattern(label_seq))
    return Counter(patterns)


def svo_pattern_similarity(counter1: Counter, counter2: Counter) -> float:
    """Dice similarity of SVO patterns."""
    all_patterns = set(counter1.keys()).union(counter2.keys())
    similarity_sum = 0.0
    for p in all_patterns:
        f1 = counter1.get(p, 0)
        f2 = counter2.get(p, 0)
        if f1 + f2 == 0:
            sim = 1.0
        else:
            sim = 2 * min(f1, f2) / (f1 + f2)
        similarity_sum += sim
    return similarity_sum / len(all_patterns)


# --- Main ---
def main():
    print("[INFO] Loading IVC lexicon JSON...")
    ivc_lex = load_lexicon(IVC_LEXICON_JSON)
    print("[INFO] Loading Tamil lexicon JSON...")
    tamil_lex = load_lexicon(TAMIL_LEXICON_JSON)

    # 1. Entropy similarity
    ent_ivc = compute_entropy(ivc_lex.get("class_counts", {}))
    ent_tamil = compute_entropy(tamil_lex.get("class_counts", {}))
    entropy_similarity = compare_entropies(ent_ivc, ent_tamil)

    # 2. Order similarity
    ivc_labels = label_distribution(ivc_lex)
    tamil_labels = label_distribution(tamil_lex)
    order_similarity = compute_cosine_similarity(Counter(ivc_labels), Counter(tamil_labels))

    # 3. Semantic similarity
    semantic_sim = semantic_overlap(ivc_lex, tamil_lex)

    # 4. Structural similarity
    ivc_structs = extract_seal_structures(ivc_lex)
    tamil_structs = extract_seal_structures(tamil_lex)
    struct_similarity = structural_similarity(ivc_structs, tamil_structs)

    # 5. SVO similarity
    ivc_svo_counts = analyze_svo_patterns(ivc_lex)
    tamil_svo_counts = analyze_svo_patterns(tamil_lex)
    svo_similarity = svo_pattern_similarity(ivc_svo_counts, tamil_svo_counts)

    # Overall score
    weights = {
        "entropy": 0.2,
        "order": 0.2,
        "semantic": 0.2,
        "structure": 0.2,
        "svo": 0.2
    }
    overall_score = (
        entropy_similarity * weights["entropy"] +
        order_similarity * weights["order"] +
        semantic_sim * weights["semantic"] +
        struct_similarity * weights["structure"] +
        svo_similarity * weights["svo"]
    ) * 100

    report = {
        "entropy_similarity": round(entropy_similarity, 4),
        "order_similarity": round(order_similarity, 4),
        "semantic_similarity": round(semantic_sim, 4),
        "structure_similarity": round(struct_similarity, 4),
        "svo_similarity": round(svo_similarity, 4),
        "overall_score_percent": round(overall_score, 2),
        "ivc_entropy": round(ent_ivc, 4),
        "tamil_entropy": round(ent_tamil, 4),
        "ivc_class_counts": ivc_lex.get("class_counts", {}),
        "tamil_class_counts": tamil_lex.get("class_counts", {}),
        "ivc_seg_count": len(ivc_labels),
        "tamil_seg_count": len(tamil_labels)
    }

    print("\n=== Lexicon Comparison Report ===")
    for k, v in report.items():
        print(f"{k:<28}: {v}")

    with open(REPORT_JSON, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print(f"\n[INFO] Detailed report saved to {REPORT_JSON}")


if __name__ == "__main__":
    main()
