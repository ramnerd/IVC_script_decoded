#!/usr/bin/env python3
"""
decode_advanced.py

Advanced probabilistic decoding of IVC seals using:
- Segment probabilities
- Proto-grammar templates
- Old Tamil lexicon mapping
- Entropy-weighted confidence
- Multi-hypothesis decoding
"""

import json
import os
from collections import defaultdict, Counter
from itertools import product

# --- INPUT PATHS ---
IVC_LEXICON_JSON = os.path.join('dataset', 'output', "IVC", 'lexicon_summary.json')
TAMIL_LEXICON_JSON = os.path.join('dataset', 'output', "tamil", "lexicon_summary.json")
TEMPLATES_JSON = os.path.join('dataset', 'output', "IVC","grammar.json")

OUTPUT_JSON = os.path.join('dataset', 'output', 'ivc_decoded_seals_advanced.json')

# Parameters
TOP_K = 3  # number of candidate decodings per seal
SEMANTIC_CLASSES = ['commodity', 'title', 'proper_name']

# --- UTILITY FUNCTIONS ---

def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_segment_probs(ivc_lex):
    """
    Returns segment probability dict: {seg_id: {class: prob}}
    """
    seg_probs = defaultdict(dict)
    for cls, reps in ivc_lex.get("representatives", {}).items():
        for rep in reps:
            seg_id = rep.get("seg_id")
            post_key = f"post_{cls}"
            seg_probs[seg_id][cls] = rep.get(post_key, 0.0)
            # entropy adjustment (lower prob if high positional entropy)
            entropy = rep.get("entropy", 0.0)
            seg_probs[seg_id][cls] *= (1 - entropy)
    return seg_probs


def map_segments_to_classes(seg_probs):
    """
    Map segments to candidate classes sorted by probability
    Returns: {seg_id: [(class, prob), ...]}
    """
    mapping = {}
    for seg, probs in seg_probs.items():
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        mapping[seg] = sorted_probs
    return mapping


def match_to_tamil(seg_class, tamil_lex):
    """
    Map IVC semantic class to most probable Old Tamil words
    Returns top 2 candidates for each class
    """
    reps = tamil_lex.get("representatives", {}).get(seg_class, [])
    counter = Counter()
    for r in reps:
        label_seq = r.get("label_sequence", [r.get("assigned_label")])
        counter.update(label_seq)
    if counter:
        return [word for word, _ in counter.most_common(2)]
    return [seg_class]


def decode_seal_candidates(seg_sequence, seg_class_probs, tamil_lex, top_k=TOP_K):
    """
    Generate top K candidate decodings for a seal.
    """
    # 1. For each segment, get top 2 candidate classes
    seg_candidates = []
    for seg in seg_sequence:
        classes = seg_class_probs.get(seg, [(seg, 1.0)])
        # weight probabilities
        seg_candidates.append(classes[:2])

    # 2. Generate all combinations
    all_combinations = list(product(*seg_candidates))

    scored_decodings = []
    for combo in all_combinations:
        classes, probs = zip(*combo)
        # joint probability = product of segment probabilities
        joint_prob = 1.0
        for p in probs:
            joint_prob *= p

        # Map classes to Tamil words
        tamil_words_candidates = []
        for cls in classes:
            tamil_words_candidates.append(match_to_tamil(cls, tamil_lex)[0])
        scored_decodings.append((tamil_words_candidates, joint_prob))

    # Sort by joint probability
    scored_decodings.sort(key=lambda x: x[1], reverse=True)
    return scored_decodings[:top_k]


# --- MAIN PIPELINE ---

def main():
    print("[INFO] Loading IVC lexicon...")
    ivc_lex = load_json(IVC_LEXICON_JSON)

    print("[INFO] Loading Tamil lexicon...")
    tamil_lex = load_json(TAMIL_LEXICON_JSON)

    print("[INFO] Loading templates...")
    templates = load_json(TEMPLATES_JSON)

    # Compute segment probabilities (entropy adjusted)
    seg_probs = compute_segment_probs(ivc_lex)

    # Candidate classes per segment
    seg_class_probs = map_segments_to_classes(seg_probs)

    # Decode each seal
    decoded_seals = {}
    for cls, reps in ivc_lex.get("representatives", {}).items():
        for rep in reps:
            seg_id = rep.get("seg_id")
            seg_sequence = rep.get("label_sequence", [seg_id])
            top_candidates = decode_seal_candidates(seg_sequence, seg_class_probs, tamil_lex)
            decoded_seals[seg_id] = [{"sequence": seq, "confidence": round(prob,4)} for seq, prob in top_candidates]

    # Save decoded seals
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(decoded_seals, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Decoded seals saved to {OUTPUT_JSON}")
    print("[INFO] Example decoded seals:")
    for k, v in list(decoded_seals.items())[:5]:
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
