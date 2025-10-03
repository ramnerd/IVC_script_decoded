#!/usr/bin/env python3
"""
grammar.py — Customized grammar induction tuned for sparse data (no clustering)

Design goals (as requested):
 - Compatible with boundary.py segmentation output (dataset/output/boundary.txt)
 - Avoid clustering algorithms (DBSCAN/other) because data is very sparse (556 lines)
 - Use robust, conservative grouping using normalized Levenshtein + context-consensus heuristics
 - Generate templates with up to 2 slots, prefer high-support generalizations, strong MDL-like penalty
 - Emphasize regularization to avoid overfitting (validation holdout, parameter penalties,
   minimum improvement thresholds)
 - Produce grammar.json with segments, groups, templates and diagnostic outputs
 - Deterministic where possible (seeded RNG only for train/val split)

Notes on approach:
 - We mine contiguous and small gapped patterns, then generalize by turning variant positions
   into "slots" only when there is coherent evidence across occurrences (majority group +
   low intra-group diversity measured by Levenshtein on tokens).
 - Grouping is produced by an agglomerative, thresholded, conservative merge based mostly on
   normalized Levenshtein (sequence-level) but constrained by occurrence-context agreement. No
   clustering libraries are used.
 - Template scoring uses an approximate likelihood gain vs unigram baseline, penalized by
   model complexity (number of slots & distinct slot groups); validation counts add small
   positive weight. We also require minimum net gain to accept a template.

Compatibility: Python 3.8+, requires numpy,pandas
Author: Assistant (customized for user)
Version: 1.0-custom-no-cluster
"""

from __future__ import annotations

import os
import sys
import re
import json
import math
import random
import argparse
import logging
import itertools
import traceback
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Any, Optional, Set

import numpy as np
import pandas as pd

# ---------------------------
# Defaults & constants
# ---------------------------
DEFAULT_BOUNDARY_PATH = os.path.join("dataset", "output","IVC", "boundary_IVC.txt")
OUT_DIR_DEFAULT = os.path.join("dataset", "output", "IVC")
SEGMENTS_CSV = "segments.csv"
SEGMENT_GROUPS_CSV = "segment_groups.csv"
SEQ_TEMPLATES_TXT = "sequence_templates.txt"
GRAMMAR_JSON = "grammar.json"
VALIDATION_TXT = "validation.txt"
DIAG_DIR = "diagnostics"

# DEFAULT_BOUNDARY_PATH = os.path.join("dataset", "output", "boundary_IVC.txt")
# OUT_DIR_DEFAULT = os.path.join("dataset", "output", "IVC")
# SEGMENTS_CSV = "segments.csv"
# SEGMENT_GROUPS_CSV = "segment_groups.csv"
# SEQ_TEMPLATES_TXT = "sequence_templates.txt"
# GRAMMAR_JSON = "grammar.json"
# VALIDATION_TXT = "validation.txt"
# DIAG_DIR = "diagnostics"

DEFAULT_MIN_SUPPORT = 4
DEFAULT_MAX_NGRAM = 4
DEFAULT_MAX_GAPPED_NGRAM = 3
DEFAULT_LEV_THRESHOLD = 0.22
DEFAULT_CONTEXT_WINDOW = 2
DEFAULT_HOLDOUT = 0.20
DEFAULT_MAX_LEV_CANDIDATES = 1000
DEFAULT_SLOT_COHESION = 0.38
DEFAULT_RANDOM_SEED = 12345

MAX_OCCURRENCES_RECORD = 2000
MAX_CONTEXT_VOCAB = 400

# Scoring hyperparams
DEFAULT_ALPHA = 1.0   # complexity penalty factor
DEFAULT_BETA = 0.25   # validation weight
DEFAULT_MAX_SLOTS = 2  # max slot positions to consider when generalizing
MIN_DELTA_GAIN = 2.0  # minimal loglik gain required to accept a template

# ---------------------------
# Logging
# ---------------------------

def configure_logging(debug: bool = False):
    fmt = "[%(levelname)s] %(asctime)s - %(message)s"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=fmt, level=level)

# ---------------------------
# Parsing helpers
# ---------------------------
SPLIT_RE = re.compile(r"[;,|\s]+")


def parse_segmented_line(line: str) -> List[List[int]]:
    """Parse one segmented line "a, b | c | d,e" -> [[a,b],[c],[d,e]] as ints."""
    if line is None:
        return []
    s = str(line).strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split("|")]
    out = []
    for p in parts:
        if p == "":
            out.append([])
            continue
        toks = SPLIT_RE.split(p)
        seq = []
        for t in toks:
            if not t:
                continue
            m = re.search(r"(-?\d+)", t)
            if m:
                try:
                    seq.append(int(m.group(1)))
                except Exception:
                    continue
        out.append(seq)
    return out


def read_boundary_file(path: str) -> List[List[List[int]]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Boundary file not found: {path}")
    seqs: List[List[List[int]]] = []
    with open(path, 'r', encoding='utf-8') as fh:
        for ln in fh:
            ln = ln.rstrip('\n')
            if ln.strip() == "":
                seqs.append([])
            else:
                seqs.append(parse_segmented_line(ln))
    return seqs

# ---------------------------
# Inventory & ngram mining
# ---------------------------

def canonical_seg_key(tokens: List[int]) -> str:
    return ",".join(str(x) for x in tokens)


def build_segment_inventory(sequences: List[List[List[int]]]) -> Tuple[Dict[str, Dict[str, Any]], List[List[str]]]:
    counts = Counter()
    examples = defaultdict(list)
    seq_seg_keys = []
    for sidx, seq in enumerate(sequences):
        keys = []
        for pos, seg in enumerate(seq):
            k = canonical_seg_key(seg)
            counts[k] += 1
            if len(examples[k]) < 5:
                examples[k].append((sidx, pos))
            keys.append(k)
        seq_seg_keys.append(keys)
    segments_map = {}
    for i, (k, f) in enumerate(sorted(counts.items(), key=lambda x: (-x[1], x[0])), start=1):
        seg_id = f"seg_{i:05d}"
        tokens = [int(x) for x in k.split(",")] if k != "" else []
        segments_map[k] = {"seg_id": seg_id, "tokens": tokens, "freq": int(f), "examples": examples.get(k, [])}
    return segments_map, seq_seg_keys


def mine_contiguous_ngrams(seq_seg_keys: List[List[str]], max_n: int = 4, min_support: int = 2) -> Dict[Tuple[str, ...], Dict[str, Any]]:
    counts = {}
    for sidx, seq in enumerate(seq_seg_keys):
        L = len(seq)
        for start in range(0, L):
            for n in range(1, max_n + 1):
                end = start + n
                if end > L:
                    break
                ng = tuple(seq[start:end])
                if ng not in counts:
                    counts[ng] = {"count": 0, "occurrences": []}
                counts[ng]['count'] += 1
                if len(counts[ng]['occurrences']) < MAX_OCCURRENCES_RECORD:
                    counts[ng]['occurrences'].append((sidx, start))
    filtered = {ng: v for ng, v in counts.items() if v['count'] >= min_support}
    return filtered


def mine_small_gapped_patterns(seq_seg_keys: List[List[str]], max_len: int = 3, min_support: int = 2, max_gap: int = 1) -> Dict[Tuple[str, ...], Dict[str, Any]]:
    # Conservative gapped mining (only small gaps) to keep sparsity manageable
    counts = {}
    for sidx, seq in enumerate(seq_seg_keys):
        L = len(seq)
        if L == 0:
            continue
        # sliding window up to max_len + max_gap
        for i in range(0, L):
            for span in range(1, min(L - i, max_len + max_gap) + 1):
                window = seq[i:i+span]
                # enumerate subsequences up to length max_len with limited gaps
                for k in range(1, min(max_len, len(window)) + 1):
                    for positions in itertools.combinations(range(len(window)), k):
                        # check total gaps
                        span_span = positions[-1] - positions[0]
                        if span_span - (k - 1) > max_gap:
                            continue
                        ng = tuple(window[pos] for pos in positions)
                        if ng not in counts:
                            counts[ng] = {"count": 0, "occurrences": []}
                        counts[ng]['count'] += 1
                        if len(counts[ng]['occurrences']) < MAX_OCCURRENCES_RECORD:
                            counts[ng]['occurrences'].append((sidx, i + positions[0]))
    filtered = {ng: v for ng, v in counts.items() if v['count'] >= min_support}
    return filtered

# ---------------------------
# Normalized Levenshtein distance for integer sequences
# ---------------------------

def normalized_levenshtein(a: List[int], b: List[int]) -> float:
    na, nb = len(a), len(b)
    if na == 0 and nb == 0:
        return 0.0
    if na == 0 or nb == 0:
        return 1.0
    # use classic DP
    prev = list(range(nb + 1))
    for i in range(1, na + 1):
        curr = [i] + [0] * nb
        ai = a[i-1]
        for j in range(1, nb + 1):
            cost = 0 if ai == b[j-1] else 1
            curr[j] = min(prev[j] + 1, curr[j-1] + 1, prev[j-1] + cost)
        prev = curr
    dist = prev[nb]
    return float(dist / max(na, nb))

# ---------------------------
# Grouping without clustering: conservative agglomerative merges guided by lev + context
# ---------------------------

def initial_groups_by_frequency(segments_map: Dict[str, Any], min_support: int) -> Dict[str, str]:
    """Each segment key initially its own group; we will only merge when evidence is strong."""
    seg_to_group = {}
    gid = 1
    for k in segments_map.keys():
        seg_to_group[k] = f"g{gid:05d}"
        gid += 1
    return seg_to_group


def build_context_profiles(seq_seg_keys: List[List[str]], window: int = DEFAULT_CONTEXT_WINDOW):
    left = defaultdict(Counter)
    right = defaultdict(Counter)
    for seq in seq_seg_keys:
        L = len(seq)
        for i, s in enumerate(seq):
            for w in range(1, window+1):
                j = i - w
                if j < 0: break
                left[s][seq[j]] += 1
            for w in range(1, window+1):
                j = i + w
                if j >= L: break
                right[s][seq[j]] += 1
    return left, right


def profile_similarity(a: Counter, b: Counter) -> float:
    # cosine-like similarity on counts (small robust version)
    if not a and not b:
        return 1.0
    keys = set(a.keys()) | set(b.keys())
    num = 0.0
    na = 0.0
    nb = 0.0
    for k in keys:
        va = float(a.get(k,0)); vb = float(b.get(k,0))
        num += va * vb
        na += va * va
        nb += vb * vb
    denom = math.sqrt(na) * math.sqrt(nb) + 1e-12
    return num / denom


def conservative_agglomerative_grouping(segments_map: Dict[str, Any], seq_seg_keys: List[List[str]], min_support: int = 2, lev_thresh: float = DEFAULT_LEV_THRESHOLD, context_window: int = DEFAULT_CONTEXT_WINDOW, cohesion_thresh: float = DEFAULT_SLOT_COHESION):
    """Agglomerative merging based mostly on Levenshtein plus context agreement.

    Strategy:
     - Consider only segment keys with freq >= min_support for merging candidates.
     - For each pair (a,b) of candidates (limited by top-K freq), compute lev distance on token lists.
     - If lev <= lev_thresh AND context profiles are similar above threshold, merge.
     - Merges are transitive (union-find). Low-risk, conservative merges only.
    """
    # build list of candidate keys ordered by frequency
    items = [(k, info['freq']) for k, info in segments_map.items()]
    items.sort(key=lambda x: (-x[1], x[0]))
    candidates = [k for k,f in items if f >= min_support]
    # limit for combinatorial safety
    if len(candidates) > DEFAULT_MAX_LEV_CANDIDATES:
        candidates = candidates[:DEFAULT_MAX_LEV_CANDIDATES]
    n = len(candidates)

    # union-find
    parent = {k:k for k in candidates}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a,b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # build context profiles
    left_prof, right_prof = build_context_profiles(seq_seg_keys, window=context_window)

    # compute pairwise comparisons but limited by frequency ranking neighborhood (compare only within top-K neighbors)
    K_NEIGHBOR = 200  # compare only to nearest by rank to limit comparisons
    for i in range(n):
        ki = candidates[i]
        ai = segments_map[ki]['tokens']
        for j in range(i+1, min(n, i + K_NEIGHBOR)):
            kj = candidates[j]
            bj = segments_map[kj]['tokens']
            # fast length-based filter
            len_ratio = max(1e-12, len(ai)) / max(1, len(bj))
            if len(ai) == 0 or len(bj) == 0:
                continue
            if max(len(ai), len(bj)) > 20:
                # avoid extremely long sequences
                continue
            d = normalized_levenshtein(ai, bj)
            if d > lev_thresh:
                continue
            # check context similarity
            left_sim = profile_similarity(left_prof.get(ki, Counter()), left_prof.get(kj, Counter()))
            right_sim = profile_similarity(right_prof.get(ki, Counter()), right_prof.get(kj, Counter()))
            if left_sim >= 0.45 and right_sim >= 0.45:
                # candidate strong merge
                union(ki, kj)

    groups = defaultdict(list)
    for k in candidates:
        groups[find(k)].append(k)
    # build seg_to_group mapping
    seg_to_group = {}
    gid = 1
    for r, members in groups.items():
        gidname = f"grp_{gid:04d}"
        gid += 1
        for m in members:
            seg_to_group[m] = gidname
    # remaining keys not in candidates -> singleton groups
    for k in segments_map.keys():
        if k not in seg_to_group:
            seg_to_group[k] = f"sg_{abs(hash(k))%1000000}"
    return seg_to_group

# ---------------------------
# Candidate generalization (generate templates with slots)
# ---------------------------

def estimate_baseline_unigram_probs(seq_seg_keys: List[List[str]], alpha: float = 1.0):
    uni = Counter()
    for seq in seq_seg_keys:
        uni.update(seq)
    V = max(1, len(uni))
    total = sum(uni.values())
    denom = total + alpha * V
    probs = {k: (v + alpha) / denom for k, v in uni.items()}
    unk = alpha / denom
    return probs, unk


def estimate_delta_loglik_approx(ngram: Tuple[str, ...], c_train: int, unigram_probs: Dict[str, float], unk_prob: float, total_positions: int):
    # approximate: delta = c_train * (log p_template - log p_baseline)
    p_baseline = 1.0
    for tok in ngram:
        p_baseline *= unigram_probs.get(tok, unk_prob)
    p_baseline = max(p_baseline, 1e-300)
    p_template = max(1e-300, c_train / max(1, total_positions))
    delta = c_train * (math.log(p_template) - math.log(p_baseline))
    return float(delta)


def candidate_generalizations(ngram: Tuple[str, ...], occurrences: List[Tuple[int, int]], seq_seg_keys: List[List[str]], seg_to_group: Dict[str, str], segments_map: Dict[str, Any], min_support: int = 2, max_slots: int = 2, cohesion_threshold: float = DEFAULT_SLOT_COHESION):
    n = len(ngram)
    # gather variants per position across occurrences
    pos_variants = [set() for _ in range(n)]
    for sidx, start in occurrences:
        if sidx < 0 or sidx >= len(seq_seg_keys):
            continue
        seq = seq_seg_keys[sidx]
        if start + n > len(seq):
            continue
        for i in range(n):
            pos_variants[i].add(seq[start + i])
    variable_positions = [i for i, s in enumerate(pos_variants) if len(s) > 1]
    results = []
    # consider exact template always
    if len(occurrences) >= min_support:
        results.append({
            'pattern': list(ngram),
            'slot_map': {},
            'occurrences': occurrences.copy(),
            'count': len(occurrences)
        })
    if not variable_positions:
        return results
    # enumerate slot combinations up to max_slots conservatively
    for r in range(1, min(max_slots, len(variable_positions)) + 1):
        for combo in itertools.combinations(variable_positions, r):
            # for each position, compute majority group for variants
            slot_map = {}
            ok = True
            for pos in combo:
                variants = pos_variants[pos]
                # map variants to groups
                grp_counts = Counter(seg_to_group.get(v, '') for v in variants)
                if not grp_counts:
                    ok = False; break
                top_group, cnt = grp_counts.most_common(1)[0]
                frac = cnt / max(1, len(variants))
                # require strong majority
                if frac < 0.65:
                    ok = False; break
                # compute cohesion of tokens within group -> use lev distances
                members = [v for v in variants if seg_to_group.get(v) == top_group]
                if len(members) < 2:
                    cohesion = 0.0
                else:
                    # average pairwise lev among tokens' integer sequences
                    total = 0.0; c = 0
                    mm = members
                    for i1 in range(len(mm)):
                        for j1 in range(i1+1, len(mm)):
                            total += normalized_levenshtein(segments_map[mm[i1]]['tokens'], segments_map[mm[j1]]['tokens'])
                            c += 1
                    cohesion = (total / max(1, c)) if c > 0 else 0.0
                if cohesion > cohesion_threshold:
                    ok = False; break
                slot_map[pos] = top_group
            if not ok:
                continue
            # build generalized pattern and re-scan corpus to find occurrences
            pattern = []
            for i in range(n):
                if i in slot_map:
                    pattern.append({'slot': slot_map[i]})
                else:
                    pattern.append(ngram[i])
            gen_occ = []
            # rescan corpus for matches
            for sidx, seq in enumerate(seq_seg_keys):
                L = len(seq)
                for start in range(0, L - n + 1):
                    match = True
                    for i in range(n):
                        seg_at = seq[start + i]
                        pat = pattern[i]
                        if isinstance(pat, dict):
                            if seg_to_group.get(seg_at) != pat['slot']:
                                match = False; break
                        else:
                            if seg_at != pat:
                                match = False; break
                    if match:
                        gen_occ.append((sidx, start))
            if len(gen_occ) >= min_support:
                results.append({'pattern': pattern, 'slot_map': {f'slot_{p}': slot_map[p] for p in slot_map}, 'occurrences': gen_occ, 'count': len(gen_occ)})
    return results

# ---------------------------
# Weighted interval scheduling (per sequence DP)
# ---------------------------

def weighted_interval_scheduling(cands: List[Tuple[int,int,float,int]]):
    if not cands:
        return []
    # we will adapt classical algorithm but since lists small, use simple DP
    # sort by end
    idxs = list(range(len(cands)))
    idxs.sort(key=lambda i: cands[i][1])
    ends = [cands[i][1] for i in idxs]
    P = [None] * len(idxs)
    for ii, i in enumerate(idxs):
        s_i, e_i, sc_i, tid = cands[i]
        # find rightmost j with end <= s_i
        j = -1
        lo = 0; hi = ii - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if cands[idxs[mid]][1] <= s_i:
                j = mid; lo = mid + 1
            else:
                hi = mid - 1
        P[ii] = j
    dp = [0.0] * (len(idxs) + 1)
    take = [False] * len(idxs)
    for i in range(1, len(idxs) + 1):
        idx = idxs[i-1]
        s,e,sc,tidx = cands[idx]
        incl = sc + (dp[P[i-1] + 1] if P[i-1] != -1 else 0.0)
        excl = dp[i-1]
        if incl > excl:
            dp[i] = incl; take[i-1] = True
        else:
            dp[i] = excl
    # backtrack
    chosen = []
    i = len(idxs) - 1
    while i >= 0:
        if take[i]:
            chosen.append(idxs[i])
            i = P[i]
        else:
            i -= 1
    chosen.reverse()
    return chosen

# ---------------------------
# Core pipeline
# ---------------------------

def run_pipeline(boundary_path: str = DEFAULT_BOUNDARY_PATH,
                 out_dir: str = OUT_DIR_DEFAULT,
                 min_support: int = DEFAULT_MIN_SUPPORT,
                 max_ngram: int = DEFAULT_MAX_NGRAM,
                 use_gapped: bool = False,
                 max_gapped: int = DEFAULT_MAX_GAPPED_NGRAM,
                 max_slots: int = DEFAULT_MAX_SLOTS,
                 lev_threshold: float = DEFAULT_LEV_THRESHOLD,
                 context_window: int = DEFAULT_CONTEXT_WINDOW,
                 holdout_frac: float = DEFAULT_HOLDOUT,
                 seed: int = DEFAULT_RANDOM_SEED,
                 cohesion_threshold: float = DEFAULT_SLOT_COHESION,
                 alpha: float = DEFAULT_ALPHA,
                 beta: float = DEFAULT_BETA,
                 debug: bool = False,
                 compare_paths: Optional[List[str]] = None,
                 example_mode: bool = False):
    configure_logging(debug)
    random.seed(seed); np.random.seed(seed)
    logging.info("grammar (no-cluster) started")

    if example_mode:
        sequences = [ [[1],[2],[3],[4]], [[1],[2],[5],[4]], [[1],[6],[3],[4]], [[7],[2],[3],[4]] ]
    else:
        logging.info("Reading segmented sequences from %s", boundary_path)
        sequences = read_boundary_file(boundary_path)

    nseq = len(sequences)
    logging.info("Total sequences: %d", nseq)

    segments_map, seq_seg_keys = build_segment_inventory(sequences)
    logging.info("Unique segments: %d", len(segments_map))

    # train/val split
    idxs = list(range(nseq))
    random.shuffle(idxs)
    nval = max(1, int(math.ceil(nseq * holdout_frac)))
    val_idxs = set(idxs[:nval])
    train_idxs = set(idxs[nval:])
    logging.info("Train: %d, Val: %d", len(train_idxs), len(val_idxs))

    train_seq_keys = [seq_seg_keys[i] for i in range(nseq) if i in train_idxs]

    # mine patterns
    ngrams = mine_contiguous_ngrams(train_seq_keys, max_n=max_ngram, min_support=min_support)
    logging.info("Mined %d contiguous ngrams (support>=%d)", len(ngrams), min_support)
    if use_gapped:
        gapped = mine_small_gapped_patterns(train_seq_keys, max_len=max_gapped, min_support=min_support, max_gap=1)
        for k,v in gapped.items():
            if k in ngrams:
                ngrams[k]['count'] += v['count']; ngrams[k]['occurrences'].extend(v['occurrences'])
            else:
                ngrams[k] = v
        logging.info("Including gapped patterns total candidates now %d", len(ngrams))

    # grouping WITHOUT clustering: conservative agglomerative merges
    seg_to_group = conservative_agglomerative_grouping(segments_map, seq_seg_keys, min_support=min_support, lev_thresh=lev_threshold, context_window=context_window, cohesion_thresh=cohesion_threshold)
    logging.info("Grouping created %d groups (sample)", len(set(seg_to_group.values())))

    # baseline unigram probs
    unigram_probs, unk_prob = estimate_baseline_unigram_probs(train_seq_keys, alpha=1.0)
    total_train_positions = sum(len(s) for s in train_seq_keys)

    # generate candidate templates
    all_candidates = []
    for ng, info in ngrams.items():
        occs = info.get('occurrences', [])
        cands = candidate_generalizations(ng, occs, seq_seg_keys, seg_to_group, segments_map, min_support=min_support, max_slots=max_slots, cohesion_threshold=cohesion_threshold)
        for c in cands:
            all_candidates.append(c)
    logging.info("Generated %d raw candidate templates", len(all_candidates))

    # scoring and filtering
    scored_candidates = []
    for cand in all_candidates:
        c_train = cand['count']
        est_delta = estimate_delta_loglik_approx(tuple([el if not isinstance(el, dict) else '__SLOT__' for el in cand['pattern']]), c_train, unigram_probs, unk_prob, total_train_positions)
        num_params = len(cand.get('slot_map', {}))
        penalty = alpha * num_params * math.log(max(2, total_train_positions))
        # validation occurrences
        c_val = sum(1 for (sidx, st) in cand['occurrences'] if sidx in val_idxs)
        score = est_delta - penalty + beta * c_val
        cand['_score'] = score
        cand['_est_delta'] = est_delta
        cand['_c_val'] = c_val
        # require some minimal evidence and non-trivial gain
        if cand['count'] >= min_support and score >= MIN_DELTA_GAIN:
            scored_candidates.append(cand)
    scored_candidates.sort(key=lambda x: x.get('_score', 0.0), reverse=True)
    logging.info("Kept %d candidates after scoring and thresholding", len(scored_candidates))

    # Greedy selection with DP placement (conservative)
    selected_templates = []
    covered_positions = [set() for _ in range(nseq)]
    ITER_LIMIT = min(200, len(scored_candidates))
    it = 0
    while scored_candidates and it < ITER_LIMIT:
        it += 1
        cand = scored_candidates.pop(0)
        # compute marginal gain by placing greedily (per sequence interval scheduling)
        placements = defaultdict(list)  # sidx -> list of (start,end,score)
        plen = len(cand['pattern'])
        for (sidx, start) in cand['occurrences']:
            if sidx not in train_idxs and sidx not in val_idxs:
                continue
            # skip if overlaps covered positions
            overlap = any((p in covered_positions[sidx]) for p in range(start, start+plen))
            if overlap:
                continue
            placements[sidx].append((start, start+plen, cand['_score'], 0))
        marginal = 0.0
        chosen_spans = {}
        for sidx, spans in placements.items():
            if not spans: continue
            chosen_idx = weighted_interval_scheduling(spans)
            for idx in chosen_idx:
                s,e,sc,_ = spans[idx]
                marginal += sc
                chosen_spans.setdefault(sidx, []).append((s,e))
        if marginal <= 0:
            continue
        # accept template
        selected_templates.append(cand)
        # mark covered positions
        for sidx, spans in chosen_spans.items():
            for s,e in spans:
                for p in range(s,e): covered_positions[sidx].add(p)
        # remove candidates that now have zero non-overlapping occurrences
        new_list = []
        for other in scored_candidates:
            still = False
            oplen = len(other['pattern'])
            for (sidx, start) in other['occurrences']:
                if any((p in covered_positions[sidx]) for p in range(start, start+oplen)):
                    continue
                still = True; break
            if still:
                new_list.append(other)
        scored_candidates = new_list
        logging.info("Selected template #%d score=%.4f count=%d marginal=%.4f", len(selected_templates), cand['_score'], cand['count'], marginal)

    logging.info("Selected %d templates", len(selected_templates))

    # final placement on all sequences using DP with template scores
    tpl_scores = {i: t.get('_score', t.get('count',1)) for i,t in enumerate(selected_templates)}
    seq_matches = [[] for _ in range(nseq)]
    total_covered = 0
    for sidx, seq in enumerate(seq_seg_keys):
        spans = []
        for tidx, t in enumerate(selected_templates):
            pat = t['pattern']; plen = len(pat)
            if plen == 0 or plen > len(seq): continue
            for start in range(0, len(seq) - plen + 1):
                ok = True
                for i in range(plen):
                    el = pat[i]; seg_at = seq[start + i]
                    if isinstance(el, dict) and 'slot' in el:
                        if seg_to_group.get(seg_at) != el['slot']:
                            ok = False; break
                    else:
                        if seg_at != el:
                            ok = False; break
                if ok:
                    spans.append((start, start+plen, tpl_scores.get(tidx,1.0), tidx))
        chosen_idx = weighted_interval_scheduling(spans)
        covered = set()
        chosen_spans = []
        for idx in chosen_idx:
            s,e,sc,tidx = spans[idx]
            chosen_spans.append((tidx, s, e-s))
            for p in range(s,e): covered.add(p)
        seq_matches[sidx] = chosen_spans
        total_covered += len(covered)

    total_positions = sum(len(s) for s in seq_seg_keys)
    coverage_pct = (total_covered / total_positions * 100.0) if total_positions > 0 else 0.0
    logging.info("Final coverage: %d / %d (%.2f%%)", total_covered, total_positions, coverage_pct)

    # write outputs
    os.makedirs(out_dir, exist_ok=True)
    # segments.csv
    seg_rows = []
    for key, info in segments_map.items():
        seg_rows.append({
            'seg_key': key,
            'seg_id': info['seg_id'],
            'tokens': '|'.join(str(x) for x in info['tokens']),
            'freq': info['freq'],
            'group_id': seg_to_group.get(key, ''),
            'examples': ';'.join(f"{a}:{b}" for a,b in info.get('examples', []))
        })
    pd.DataFrame(seg_rows).to_csv(os.path.join(out_dir, SEGMENTS_CSV), index=False)
    # group file
    group_members = defaultdict(list)
    for k,g in seg_to_group.items(): group_members[g].append(k)
    grp_rows = []
    for gid, members in group_members.items():
        grp_rows.append({'group_id': gid, 'members': '|'.join(members), 'size': len(members)})
    pd.DataFrame(grp_rows).to_csv(os.path.join(out_dir, SEGMENT_GROUPS_CSV), index=False)
    # templates text
    with open(os.path.join(out_dir, SEQ_TEMPLATES_TXT), 'w', encoding='utf-8') as fh:
        for i,t in enumerate(selected_templates, start=1):
            fh.write(f"TEMPLATE_{i:05d}\tcount={t['count']}\tscore={t.get('_score',0.0)}\tpattern={t['pattern']}\n")
            fh.write("  slot_map=" + json.dumps(t.get('slot_map', {})) + "\n")
            fh.write("  examples=" + json.dumps(t.get('occurrences', [])[:50]) + "\n\n")
    # grammar.json
    grammar = {
        'metadata': {'generated_by': 'grammar_no_cluster', 'params': {'min_support': min_support, 'max_ngram': max_ngram, 'use_gapped': use_gapped, 'max_slots': max_slots, 'alpha': alpha, 'beta': beta}},
        'segments': {}, 'groups': {}, 'templates': []
    }
    for k, info in segments_map.items():
        grammar['segments'][info['seg_id']] = {'seg_key': k, 'tokens': info['tokens'], 'freq': info['freq'], 'group': seg_to_group.get(k)}
    for gid, members in group_members.items():
        grammar['groups'][gid] = {'members': [segments_map[m]['seg_id'] for m in members], 'size': len(members)}
    for i,t in enumerate(selected_templates, start=1):
        grammar['templates'].append({'template_id': f't{i:05d}', 'pattern': t['pattern'], 'slot_map': t.get('slot_map', {}), 'count': t['count'], 'score': t.get('_score',0.0), 'examples': t.get('occurrences', [])[:200]})
    with open(os.path.join(out_dir, GRAMMAR_JSON), 'w', encoding='utf-8') as fh:
        json.dump(grammar, fh, indent=2)

    # validation.txt
    def coverage_for_idxs(indices: Set[int]):
        tot_pos = 0; cov_pos = 0; seqs_with = 0
        for sidx in indices:
            if sidx < 0 or sidx >= len(seq_seg_keys): continue
            L = len(seq_seg_keys[sidx]); tot_pos += L
            matches = seq_matches[sidx]
            if matches:
                seqs_with += 1
                covered = set()
                for tidx, start, plen in matches:
                    for p in range(start, start+plen): covered.add(p)
                cov_pos += len(covered)
        pct_pos = cov_pos / tot_pos * 100.0 if tot_pos>0 else 0.0
        pct_seq = seqs_with / max(1, len(indices)) * 100.0
        return cov_pos, tot_pos, pct_pos, pct_seq
    tr_cov = coverage_for_idxs(train_idxs)
    v_cov = coverage_for_idxs(val_idxs)
    with open(os.path.join(out_dir, VALIDATION_TXT), 'w', encoding='utf-8') as fh:
        fh.write("Validation Report\n")
        fh.write("=================\n")
        fh.write(f"Total sequences: {nseq}\n")
        fh.write(f"Unique segments: {len(segments_map)}\n")
        fh.write(f"Selected templates: {len(selected_templates)}\n")
        fh.write("\nTraining coverage:\n")
        fh.write(f" Covered positions: {tr_cov[0]} / {tr_cov[1]} -> {tr_cov[2]:.2f}%\n")
        fh.write(f" Sequences with match: {tr_cov[3]:.2f}%\n")
        fh.write("\nValidation coverage:\n")
        fh.write(f" Covered positions: {v_cov[0]} / {v_cov[1]} -> {v_cov[2]:.2f}%\n")
        fh.write(f" Sequences with match: {v_cov[3]:.2f}%\n")

    # diagnostics
    diag_dir = os.path.join(out_dir, DIAG_DIR); os.makedirs(diag_dir, exist_ok=True)
    with open(os.path.join(diag_dir, 'candidates.json'), 'w', encoding='utf-8') as fh:
        json.dump([{'pattern': c['pattern'], 'count': c['count'], 'score': c.get('_score',0.0)} for c in scored_candidates[:2000]], fh, indent=2)
    with open(os.path.join(diag_dir, 'selected_templates.json'), 'w', encoding='utf-8') as fh:
        json.dump(selected_templates, fh, indent=2)
    logging.info("Wrote outputs to %s", out_dir)

    # compare if requested
    if compare_paths:
        for p in compare_paths:
            try:
                metrics = compare_grammars(os.path.join(out_dir, GRAMMAR_JSON), p)
                logging.info("Compared to %s -> %s", p, metrics)
            except Exception as e:
                logging.warning("Compare failed for %s: %s", p, e)

    logging.info("grammar run finished")
    return out_dir

# ---------------------------
# Compare grammars (lightweight)
# ---------------------------

def compare_grammars(path_a: str, path_b: str) -> Dict[str, float]:
    if not os.path.exists(path_a) or not os.path.exists(path_b):
        return {}
    A = json.load(open(path_a, 'r', encoding='utf-8'))
    B = json.load(open(path_b, 'r', encoding='utf-8'))
    sa = set(info.get('seg_key') for info in A.get('segments', {}).values() if info.get('seg_key'))
    sb = set(info.get('seg_key') for info in B.get('segments', {}).values() if info.get('seg_key'))
    inter = len(sa & sb); union = len(sa | sb)
    seg_j = inter / union if union>0 else 0.0
    pa = set(json.dumps(t.get('pattern',[]), sort_keys=True) for t in A.get('templates', []))
    pb = set(json.dumps(t.get('pattern',[]), sort_keys=True) for t in B.get('templates', []))
    ip = len(pa & pb); up = len(pa | pb)
    patt_j = ip / up if up>0 else 0.0
    ca = len(A.get('templates', [])); cb = len(B.get('templates', []))
    tmpl_ratio = 1.0 if max(ca,cb) == 0 else min(ca,cb) / max(ca,cb)
    return {'segment_key_jaccard': seg_j, 'template_pattern_jaccard': patt_j, 'template_count_ratio': tmpl_ratio}

# ---------------------------
# Basic unit tests
# ---------------------------

def run_unit_tests():
    print('Running minimal unit tests...')
    assert normalized_levenshtein([],[]) == 0.0
    assert normalized_levenshtein([], [1]) == 1.0
    assert normalized_levenshtein([1], [1]) == 0.0
    assert parse_segmented_line('1,2 | 3') == [[1,2],[3]]
    print('Basic tests passed')

# ---------------------------
# CLI
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Grammar induction (no clustering)')
    p.add_argument('--boundary', type=str, default=DEFAULT_BOUNDARY_PATH)
    p.add_argument('--out', type=str, default=OUT_DIR_DEFAULT)
    p.add_argument('--min-support', type=int, default=DEFAULT_MIN_SUPPORT)
    p.add_argument('--max-ngram', type=int, default=DEFAULT_MAX_NGRAM)
    p.add_argument('--use-gapped', action='store_true')
    p.add_argument('--max-gapped', type=int, default=DEFAULT_MAX_GAPPED_NGRAM)
    p.add_argument('--max-slots', type=int, default=DEFAULT_MAX_SLOTS)
    p.add_argument('--lev-threshold', type=float, default=DEFAULT_LEV_THRESHOLD)
    p.add_argument('--context-window', type=int, default=DEFAULT_CONTEXT_WINDOW)
    p.add_argument('--holdout', type=float, default=DEFAULT_HOLDOUT)
    p.add_argument('--seed', type=int, default=DEFAULT_RANDOM_SEED)
    p.add_argument('--cohesion', type=float, default=DEFAULT_SLOT_COHESION)
    p.add_argument('--alpha', type=float, default=DEFAULT_ALPHA)
    p.add_argument('--beta', type=float, default=DEFAULT_BETA)
    p.add_argument('--debug', action='store_true')
    p.add_argument('--compare', nargs='*')
    p.add_argument('--example', action='store_true')
    p.add_argument('--run-tests', action='store_true')
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args.run_tests:
        run_unit_tests(); sys.exit(0)
    try:
        run_pipeline(boundary_path=args.boundary, out_dir=args.out, min_support=args.min_support, max_ngram=args.max_ngram, use_gapped=args.use_gapped, max_gapped=args.max_gapped, max_slots=args.max_slots, lev_threshold=args.lev_threshold, context_window=args.context_window, holdout_frac=args.holdout, seed=args.seed, cohesion_threshold=args.cohesion, alpha=args.alpha, beta=args.beta, debug=args.debug, compare_paths=args.compare, example_mode=args.example)
    except Exception as e:
        logging.exception('Fatal error in grammar pipeline: %s', e)
        traceback.print_exc()
        sys.exit(2)
