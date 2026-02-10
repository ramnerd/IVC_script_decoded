#!/usr/bin/env python3
"""
decode.py — Model-Driven Contextual IVC → Tamil Decoder (v2.2)
Enhanced for reliability (fixed lex_summary parsing bug)
"""

import os
import json
import csv
import re
import unicodedata
import math
from difflib import SequenceMatcher
from collections import defaultdict
from datetime import datetime

# ---------------------------
# CONFIG
# ---------------------------
BASE_DIR = "dataset"
BOUNDARY_PATH = os.path.join(BASE_DIR, "output", "IVC", "boundary_IVC.txt")
LEXICON_SUMMARY_PATH = os.path.join(BASE_DIR, "output", "tamil", "lexicon_summary.json")
TAMIL_DATASET_PATH = os.path.join("script", "tamil.csv")
TAMIL_LEXICON_PATH = os.path.join("script", "tamil_2.csv")
OUTPUT_DIR = "Decoded"

TOP_K = 3

# ---------------------------
# UTILITIES
# ---------------------------
def make_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def normalize_text(text: str) -> str:
    if text is None: return ""
    text = re.sub(r"[^\u0B80-\u0BFFa-zA-Z0-9\s]", "", str(text).strip().lower())
    return " ".join(text.split())

def strip_diacritics(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")

def sequence_similarity(a, b):
    return SequenceMatcher(None, str(a), str(b)).ratio()

def phonetic_similarity(a, b):
    return SequenceMatcher(None, strip_diacritics(a), strip_diacritics(b)).ratio()

def entropy_similarity(a_len, b_len):
    if a_len <= 0 or b_len <= 0:
        return 1.0 if a_len == b_len else 0.0
    diff = abs(a_len - b_len)
    return max(0.0, 1.0 - diff / max(a_len, b_len))

def structure_similarity(ivc_seq, tamil_word):
    ivc_parts = len([p for p in ivc_seq.split(",") if p.strip()])
    tamil_chars = len(normalize_text(tamil_word))
    return entropy_similarity(ivc_parts, max(1, tamil_chars // 3))

# ---------------------------
# LOADERS
# ---------------------------
def load_lines(path):
    if not os.path.exists(path): return []
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]

def load_json(path):
    if not os.path.exists(path): return []
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            # Ensure the data is a list of dicts
            if isinstance(data, dict):
                data = [data]
            elif isinstance(data, list):
                data = [d for d in data if isinstance(d, dict)]
            else:
                data = []
            return data
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON Decode Error in {path}: {e}")
            return []

def load_tamil_corpus(path):
    if not os.path.exists(path): return []
    with open(path, "r", encoding="utf-8") as f:
        lines = [normalize_text(l) for l in f if normalize_text(l)]
    return list(dict.fromkeys(lines))

def load_tamil_lexicon(path):
    if not os.path.exists(path): return {}
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            k = normalize_text(row.get("seal_id") or row.get("seg_key") or "")
            v = normalize_text(row.get("words") or row.get("word") or row.get("tamil") or "")
            if k and v:
                mapping[k] = v
    return mapping

def bucket_tamil_by_length(corpus):
    buckets = defaultdict(list)
    for w in corpus:
        key = max(1, len(w) // 3)
        buckets[key].append(w)
    return buckets

# ---------------------------
# LEXICON / CLASS UTILITIES
# ---------------------------
def get_segment_features(seg_key, lex_summary):
    seg_key = normalize_text(seg_key)
    for e in lex_summary:
        if not isinstance(e, dict):
            continue
        k = normalize_text(e.get("seg_key") or e.get("seal_key") or "")
        if k == seg_key:
            lbl = e.get("assigned_label") or e.get("label") or "unknown"
            prob_key = f"post_{lbl}" if lbl else None
            prob = float(e.get(prob_key, 0.0)) if prob_key else 0.0
            return lbl, prob
    return "unknown", 0.0

def class_weights(seg_class):
    sc = (seg_class or "").lower()
    if sc == "title": return 0.25, 0.20, 0.25, 0.30
    if sc == "commodity": return 0.35, 0.30, 0.20, 0.15
    if sc == "proper_name": return 0.25, 0.20, 0.35, 0.20
    return 0.30, 0.25, 0.25, 0.20

def linguistic_weight(word, seg_class):
    suf_map = {
        "title": ["ar", "karar", "nāyakan", "mudaliar", "rājan"],
        "proper_name": ["vel", "an", "al"],
        "commodity": ["am", "vu", "mai"]
    }
    w = normalize_text(word)
    mult = 1.0
    for suf in suf_map.get(seg_class, []):
        if w.endswith(suf): mult *= 1.1
    if len(w) <= 2: mult *= 0.9
    return max(0.75, min(1.25, mult))

def final_score(ivc, word, seg_class, seg_prob):
    E, S, SEM, CL = class_weights(seg_class)
    e = entropy_similarity(len(ivc), len(word))
    s = structure_similarity(ivc, word)
    sem = 0.6 * phonetic_similarity(ivc, word) + 0.4 * sequence_similarity(ivc, word)
    base = (E * e) + (S * s) + (SEM * sem)
    combined = base * (1 - CL) + (CL * max(0.15, seg_prob))
    return round(combined * linguistic_weight(word, seg_class), 4)

# ---------------------------
# MATCHING
# ---------------------------
def top_k(ivc_seq, seg_class, seg_prob, tamil_buckets, tamil_full, k=TOP_K):
    key = max(1, len(ivc_seq) // 3)
    cands = []
    for kk in range(max(1, key - 2), key + 3):
        cands.extend(tamil_buckets.get(kk, []))
    if not cands:
        cands = tamil_full
    scored = [(w, final_score(ivc_seq, w, seg_class, seg_prob)) for w in cands]
    scored.sort(key=lambda x: x[1], reverse=True)
    seen, out = set(), []
    for w, s in scored:
        if w not in seen:
            seen.add(w)
            out.append((w, s))
            if len(out) >= k: break
    return out

# ---------------------------
# POSTPROCESSOR
# ---------------------------
def synthesize_tamil_sentence(words):
    words = [normalize_text(w) for w in words if w]
    if not words: return ""
    merged = []
    for w in words:
        if merged and len(w) <= 2:
            merged[-1] += w
        else:
            merged.append(w)
    words = merged
    nouns, verbs, particles = [], [], []
    for w in words:
        if re.search("(kiṟ|viṭṭ|pōn|varu|iru|seyy)", w):
            verbs.append(w)
        elif re.search("(il|in|ai|ku|kku|am|vu|mai)$", w):
            nouns.append(w)
        else:
            particles.append(w)
    sentence = nouns + verbs + particles
    sentence = [w for i, w in enumerate(sentence) if i == 0 or w != sentence[i - 1]]
    out = " ".join(sentence).strip()
    if out:
        out = out[0].upper() + out[1:]
    return out

# ---------------------------
# MAIN DECODE
# ---------------------------
def decode():
    make_output_dir()
    print("🔍 Decoding initiated...")

    boundaries = load_lines(BOUNDARY_PATH)
    lex_summary = load_json(LEXICON_SUMMARY_PATH)
    tamil_corpus = load_tamil_corpus(TAMIL_DATASET_PATH)
    tamil_lexicon = load_tamil_lexicon(TAMIL_LEXICON_PATH)
    tamil_buckets = bucket_tamil_by_length(tamil_corpus)

    decoded_lines, decoded_words = [], []

    for line_idx, line in enumerate(boundaries):
        segs = [normalize_text(s) for s in line.split("|") if s.strip()]
        tamil_out = []

        for seg in segs:
            if seg in tamil_lexicon:
                tamil_out.append(tamil_lexicon[seg])
                continue
            seg_class, seg_prob = get_segment_features(seg, lex_summary)
            top = top_k(seg, seg_class, seg_prob, tamil_buckets, tamil_corpus)
            tamil_out.append(top[0][0] if top else "unknown")
            decoded_words.append({
                "ivc_segment": seg,
                "top_candidates": [{"tamil": w, "score": s} for w, s in top]
            })

        sentence = synthesize_tamil_sentence(tamil_out)
        conf = round(sum([s[1] for s in top_k(line, "unknown", 0.5, tamil_buckets, tamil_corpus)]) / TOP_K * 100, 2)

        decoded_lines.append({
            "ivc_line": line,
            "decoded_tamil": sentence,
            "confidence_percent": conf,
            "line_index": line_idx
        })

    with open(os.path.join(OUTPUT_DIR, "decoded_words.json"), "w", encoding="utf-8") as f:
        json.dump(decoded_words, f, ensure_ascii=False, indent=2)

    with open(os.path.join(OUTPUT_DIR, "decoded.txt"), "w", encoding="utf-8") as f:
        for item in decoded_lines:
            f.write(f"{item['decoded_tamil']}\n")

    print(f"✅ Decoding complete. Meaning-enhanced Tamil sentences saved in '{OUTPUT_DIR}/decoded.txt'")

if __name__ == "__main__":
    decode()
