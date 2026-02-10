#!/usr/bin/env python3
"""
boundary.py  (fixed & enhanced)

Boundary detector (v2.1) for sequence dataset — corrected, hardened, and extended.

Features:
 - Robust sequence parsing (many separators/formats)
 - Safe CSV and plain-text loaders
 - Positional entropy & diversity map loaders (robust to headers/whitespace)
 - Unigram/bigram/trigram counts with Laplace smoothing
 - Bigram & trigram NPMI calculations (stable numerics)
 - Candidate-boundary feature extraction (10+ features)
 - BayesianGaussianMixture detector with stable scoring
 - DBSCAN detector with safe fallbacks for degenerate inputs
 - Z-score based detector with robust handling of constant arrays
 - Reliability-weighted ensemble combination (skips missing detectors)
 - Guaranteed at least one boundary per sequence
 - Diagnostic plotting (optional) and verbose logging
 - Extensive CLI options and unit-testable helper functions

Compatibility: Python 3.8+; requires numpy,pandas,scipy,scikit-learn,matplotlib (optional),tqdm

Author: Generated & debugged for user
Version: 2.1 (patched)
"""

# Standard library
import os
import sys
import re
import math
import argparse
import logging
from typing import List, Tuple, Dict, Any, Optional
from collections import Counter, defaultdict

# Third-party
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.mixture import BayesianGaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Optional for plotting diagnostics
try:
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except Exception:
    PLOTTING_AVAILABLE = False

# ---------------------------
# Constants & Defaults
# ---------------------------
HERE = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
ENTROPY_FILE_DEFAULT = os.path.join("dataset", "boundary", "sanskrit","normalized_positional_entropy.txt")
DIVERSITY_FILE_DEFAULT = os.path.join("dataset", "boundary", "sanskrit","distribution_score.txt")
SEQUENCE_FILE_DEFAULT = os.path.join("dataset", "dataset_sanskrit.csv")
OUTPUT_FILE_DEFAULT = os.path.join("dataset", "output", "sanskrit","boundary_sanskrit.txt")
DIAG_DIR_DEFAULT = os.path.join("dataset", "output", "sanskrit","diagnostics")



# ENTROPY_FILE_DEFAULT = os.path.join("dataset", "boundary","IVC", "normalized_positional_entropy.txt")
# DIVERSITY_FILE_DEFAULT = os.path.join("dataset", "boundary", "IVC", "distribution_score.txt")
# SEQUENCE_FILE_DEFAULT = os.path.join("dataset", "dataset_tamil.csv")
# OUTPUT_FILE_DEFAULT = os.path.join("dataset", "output","IVC", "boundary_IVC.txt")
# DIAG_DIR_DEFAULT = os.path.join("dataset", "output", "IVC", "diagnostics")

# Logging configuration helper (customizable by CLI)
def configure_logging(level=logging.INFO):
    """Configure root logger once. Avoid adding multiple handlers on repeated calls."""
    logger = logging.getLogger()
    if logger.handlers:
        # adjust level only
        logger.setLevel(level)
        return
    fmt = "[%(levelname)s] %(asctime)s - %(message)s"
    logging.basicConfig(format=fmt, level=level)


# ---------------------------
# Utility: parse sequence lines robustly
# Accepts many separators and bracket formats.
# Returns list[int]
# ---------------------------

def parse_sequence_line(line: str) -> List[int]:
    """Parse a single line containing a token sequence and return list of ints.

    Accepts forms like:
      - "1 2 3"
      - "1,2,3"
      - "(1, 2, 3)"
      - "[1;2;3]"
      - "1|2|3"
    Non-digit characters around numbers are ignored. Empty or whitespace-only lines -> [].
    """
    if line is None:
        return []
    s = str(line).strip()
    if not s:
        return []
    # Remove outer brackets/parentheses if they enclose the entire content
    s = s.strip()
    if (s.startswith("(") and s.endswith(")")) or (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
        s = s[1:-1].strip()
    # Replace common separators with a single space
    s = re.sub(r"[;,|]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    parts = s.split(" ")
    out: List[int] = []
    for p in parts:
        if not p:
            continue
        # find the first contiguous integer in token
        m = re.search(r"(-?\d+)", p)
        if m:
            try:
                out.append(int(m.group(1)))
            except Exception:
                # skip tokens that can't be parsed
                continue
    return out


# ---------------------------
# Load scalar map: robust CSV loader
# Expects two columns: symbol,value OR symbol,value header names
# Returns Dict[int, float]
# ---------------------------

def load_scalar_map(path: str) -> Dict[int, float]:
    """Load a two-column file into a symbol->float map.

    - Accepts CSV with headers (symbol,value) or two columns without headers.
    - Strips whitespace and attempts to cast to int/float safely.
    - If file missing, returns empty dict but logs a warning.
    """
    res: Dict[int, float] = {}
    if not path:
        return res
    if not os.path.exists(path):
        logging.warning("Scalar map file not found: %s", path)
        return res
    # Helper to coerce a row pair into k,v
    def try_parse_pair(rawk: Any, rawv: Any) -> Optional[Tuple[int, float]]:
        try:
            if rawk is None or rawv is None:
                return None
            # direct cast attempt
            k = int(str(rawk).strip())
            v = float(str(rawv).strip())
            return k, v
        except Exception:
            # regex fallback
            mk = re.search(r"(-?\d+)", str(rawk))
            mv = re.search(r"(-?\d+(?:\.\d+)?)", str(rawv))
            if mk and mv:
                try:
                    return int(mk.group(1)), float(mv.group(1))
                except Exception:
                    return None
            return None

    try:
        # First attempt: try header=0, but detect if first row looked like data or header
        df = pd.read_csv(path, dtype=str, header=0)
        # If file has fewer than two columns, fallback
        if df.shape[1] < 2:
            raise ValueError("Not enough columns")
        # guess header names
        cols = [c.strip().lower() for c in df.columns]
        if 'symbol' in cols and 'value' in cols:
            sym_col = df.columns[cols.index('symbol')]
            val_col = df.columns[cols.index('value')]
        else:
            # detect header-like first row: if first row values are non-numeric for both columns, re-read without header
            first_row = df.iloc[0].tolist() if not df.empty else []
            numeric_count = 0
            for v in first_row[:2]:
                if re.search(r"-?\d", str(v)):
                    numeric_count += 1
            if numeric_count < 1:
                # re-read without header
                df = pd.read_csv(path, dtype=str, header=None)
                if df.shape[1] < 2:
                    logging.warning("Scalar map file %s has fewer than two columns; returning empty map.", path)
                    return res
                sym_col = 0
                val_col = 1
            else:
                sym_col = df.columns[0]
                val_col = df.columns[1]

        for _, row in df.iterrows():
            rawk = row[sym_col]
            rawv = row[val_col]
            try:
                if pd.isna(rawk) or pd.isna(rawv):
                    continue
            except Exception:
                pass
            parsed = try_parse_pair(rawk, rawv)
            if parsed is not None:
                k, v = parsed
                res[k] = v
        return res
    except Exception as e:
        logging.warning("Failed to parse scalar map %s with pandas or header heuristic: %s", path, e)
        # Fallback: parse manually line by line
        try:
            with open(path, 'r', encoding='utf-8') as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    # split by common separators (tab, comma, semicolon)
                    parts = re.split(r"[\t,;]+", line)
                    if len(parts) < 2:
                        # try whitespace split
                        parts = line.split()
                        if len(parts) < 2:
                            continue
                    rawk, rawv = parts[0], parts[1]
                    parsed = try_parse_pair(rawk, rawv)
                    if parsed is not None:
                        res[parsed[0]] = parsed[1]
            return res
        except Exception as e2:
            logging.warning("Failed to load scalar map %s: %s", path, e2)
            return {}


# ---------------------------
# Compute counts: unigram, bigram, trigram and start/end counters
# ---------------------------

def compute_counts(sequences: List[List[int]]) -> Tuple[Counter, Counter, Counter, Tuple[Counter, Counter]]:
    uni = Counter()
    bi = Counter()
    tri = Counter()
    starts = Counter()
    ends = Counter()
    for seq in sequences:
        if not seq:
            continue
        n = len(seq)
        uni.update(seq)
        starts[seq[0]] += 1
        ends[seq[-1]] += 1
        for i in range(n - 1):
            bi[(seq[i], seq[i + 1])] += 1
        for i in range(n - 2):
            tri[(seq[i], seq[i + 1], seq[i + 2])] += 1
    return uni, bi, tri, (starts, ends)


# ---------------------------
# NPMI computations (stable numerics)
# ---------------------------

def safe_log(x: float) -> float:
    # safe log wrapper
    return math.log(max(x, 1e-300))


def compute_npmi_for_bigrams(uni: Counter, bi: Counter, alpha: float = 1.0) -> Dict[Tuple[int, int], float]:
    """Compute NPMI for observed bigrams with Laplace smoothing.

    Returns dict mapping (a,b) -> npmi.
    Unseen bigrams are not stored; user code should handle missing keys.
    Notes: uses unigram product as denominator (PMI = log p(ab) - log p(a) - log p(b)).
    """
    V = max(1, len(uni))
    total_unigrams = max(1, sum(uni.values()))
    total_bigrams = max(1, sum(bi.values()))

    def p_unigram(x: int) -> float:
        return (uni.get(x, 0) + alpha) / (total_unigrams + alpha * V)

    def p_bigram(x: int, y: int) -> float:
        # Laplace smoothing over bigram space with V^2 vocabulary
        return (bi.get((x, y), 0) + alpha) / (total_bigrams + alpha * (V ** 2))

    npmi_map: Dict[Tuple[int, int], float] = {}
    for (x, y), _ in bi.items():
        pxy = p_bigram(x, y)
        px = p_unigram(x)
        py = p_unigram(y)
        # PMI = log(pxy / (px * py))
        denom = -safe_log(pxy)
        # safer zero check
        if denom <= 1e-12:
            npmi_map[(x, y)] = 0.0
            continue
        pmi = safe_log(pxy) - (safe_log(px) + safe_log(py))
        npmi_map[(x, y)] = pmi / denom
    return npmi_map


def compute_npmi_for_trigrams(uni: Counter, tri: Counter, alpha: float = 1.0) -> Dict[Tuple[int, int, int], float]:
    """Compute trigram NPMI using joint v.s. unigram product PMI.

    This implementation uses PMI = log p(abc) - (log p(a)+log p(b)+log p(c)).
    Alternative formulations (e.g., conditioning on ab) may be more local but
    we're keeping the unigram-denominator variant for consistency with bigram NPMI.
    """
    V = max(1, len(uni))
    total_trigrams = max(1, sum(tri.values()))
    total_unigrams = max(1, sum(uni.values()))

    def p_trigram(a: int, b: int, c: int) -> float:
        return (tri.get((a, b, c), 0) + alpha) / (total_trigrams + alpha * (V ** 3))

    def p_unigram(x: int) -> float:
        return (uni.get(x, 0) + alpha) / (total_unigrams + alpha * V)

    npmi_tri: Dict[Tuple[int, int, int], float] = {}
    for (a, b, c), _ in tri.items():
        p3 = p_trigram(a, b, c)
        p1 = p_unigram(a)
        p2 = p_unigram(b)
        p3u = p_unigram(c)
        denom = -safe_log(p3)
        if denom <= 1e-12:
            npmi_tri[(a, b, c)] = 0.0
            continue
        pmi = safe_log(p3) - (safe_log(p1) + safe_log(p2) + safe_log(p3u))
        npmi_tri[(a, b, c)] = pmi / denom
    return npmi_tri


# ---------------------------
# Feature extraction
# ---------------------------

def extract_features_for_all_sequences(
    sequences: List[List[int]],
    uni: Counter,
    bi: Counter,
    tri: Counter,
    npmi_bi: Dict[Tuple[int, int], float],
    npmi_tri: Dict[Tuple[int, int, int], float],
    entropy_map: Dict[int, float],
    diversity_map: Dict[int, float],
    starts: Counter,
    ends: Counter,
    alpha: float = 1.0,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Extract features for every candidate boundary in every sequence.

    Candidate positions are integers 1..n-1 for a sequence of length n.
    Returns: (X, indices) where X is (n_candidates, n_features) and indices maps row->(seq_idx, pos)
    """
    all_features: List[List[float]] = []
    indices: List[Tuple[int, int]] = []

    total_starts = max(1, sum(starts.values()))
    total_ends = max(1, sum(ends.values()))

    def start_prob(s: int) -> float:
        return (starts.get(s, 0) + alpha) / (total_starts + alpha * (len(uni) if uni else 1))

    def end_prob(s: int) -> float:
        return (ends.get(s, 0) + alpha) / (total_ends + alpha * (len(uni) if uni else 1))

    avg_diversity_default = float(np.mean(list(diversity_map.values()))) if diversity_map else 0.0
    avg_entropy_default = float(np.mean(list(entropy_map.values()))) if entropy_map else 0.0

    for sidx, seq in enumerate(sequences):
        n = len(seq)
        if n < 2:
            # No internal boundaries — skip candidates; we'll force one later in output generator
            continue
        for pos in range(1, n):
            left = seq[pos - 1]
            right = seq[pos]
            # Across-boundary bigram
            bi_npmi = float(npmi_bi.get((left, right), -1.0))
            # context neighbor npmi
            left_neighbor_npmi = float(npmi_bi.get((seq[pos - 2], left), -1.0)) if pos - 2 >= 0 else -1.0
            right_neighbor_npmi = float(npmi_bi.get((right, seq[pos + 1]), -1.0)) if pos + 1 < n else -1.0
            # tri-gram npmi around boundary
            tri_npmi = -1.0
            if pos - 2 >= 0:
                tri_npmi = max(tri_npmi, float(npmi_tri.get((seq[pos - 2], seq[pos - 1], seq[pos]), -1.0)))
            if pos + 1 < n:
                tri_npmi = max(tri_npmi, float(npmi_tri.get((seq[pos - 1], seq[pos], seq[pos + 1]), -1.0)))
            # entropy/diversity
            left_entropy = float(entropy_map.get(left, avg_entropy_default))
            right_entropy = float(entropy_map.get(right, avg_entropy_default))
            delta_entropy = right_entropy - left_entropy
            abs_delta_entropy = abs(delta_entropy)
            avg_div = (float(diversity_map.get(left, avg_diversity_default)) + float(diversity_map.get(right, avg_diversity_default))) / 2.0
            # start/end probs
            sp_left = start_prob(left)
            ep_right = end_prob(right)
            # position encoding
            cont_pos = ((pos - 1) / max(1, (n - 2))) if n > 2 else 0.5
            # Additional derived features (helpful to classifiers):
            # - ratio of left/right unigram frequency (smoothed)
            left_freq = uni.get(left, 0)
            right_freq = uni.get(right, 0)
            freq_ratio = (left_freq + alpha) / (right_freq + alpha)
            # - is left frequent start? is right frequent end?
            is_left_common_start = 1.0 if starts.get(left, 0) > (0.01 * total_starts) else 0.0
            is_right_common_end = 1.0 if ends.get(right, 0) > (0.01 * total_ends) else 0.0

            feat = [
                bi_npmi,
                left_neighbor_npmi,
                right_neighbor_npmi,
                tri_npmi,
                abs_delta_entropy,
                delta_entropy,
                avg_div,
                sp_left,
                ep_right,
                cont_pos,
                freq_ratio,
                is_left_common_start,
                is_right_common_end,
            ]
            all_features.append(feat)
            indices.append((sidx, pos))

    if not all_features:
        X = np.zeros((0, 13), dtype=float)
        return X, indices
    X = np.array(all_features, dtype=float)
    return X, indices


# ---------------------------
# Detector: Bayesian GMM
# ---------------------------

def run_bayesian_gmm(X: np.ndarray, n_components: int = 8, random_state: int = 42, max_iter: int = 500) -> Tuple[np.ndarray, float]:
    """Fit a BayesianGaussianMixture on standardized features and score boundaryness.

    Heuristic: components with lower mean of bigram-npmi (feature 0) are more likely boundaries.
    Returns (p_boundary, reliability)
    """
    if X is None or X.shape[0] == 0:
        return np.array([]), 0.0

    # If too few samples, fallback to zscore (safer than forcing multi-component fit)
    if X.shape[0] < 2:
        return run_zscore_npmi(X)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    # cap components to number of samples (but at least 1)
    n_comp = max(1, min(n_components, X.shape[0]))
    try:
        bgmm = BayesianGaussianMixture(
            n_components=n_comp,
            weight_concentration_prior_type='dirichlet_process',
            covariance_type='full',
            random_state=random_state,
            max_iter=max_iter,
        )
        bgmm.fit(Xs)
        post = bgmm.predict_proba(Xs)
    except Exception as e:
        logging.warning("BGMM failed: %s", e)
        # fallback: use zscore on feature 0 as approximate 'probability'
        arr = X[:, 0].astype(float)
        if arr.std() < 1e-8:
            p = np.full(arr.shape[0], 0.5, dtype=float)
            return p, 0.1
        zs = (arr - arr.mean()) / (arr.std() + 1e-12)
        p = np.clip(1.0 - norm.cdf(zs), 0.0, 1.0)
        return p, 0.1

    # compute component-level scores based on original (non-scaled) feature 0
    comp_scores = []
    for k in range(post.shape[1]):
        weights_k = post[:, k]
        if weights_k.sum() == 0:
            comp_scores.append(0.0)
            continue
        mean_feat0 = float((weights_k @ X[:, 0]) / (weights_k.sum() + 1e-12))
        comp_scores.append(mean_feat0)
    comp_scores = np.array(comp_scores)
    # invert so that lower mean -> higher score
    mx = comp_scores.max()
    mn = comp_scores.min()
    if mx - mn < 1e-12:
        comp_boundary_score = np.ones_like(comp_scores)
    else:
        comp_boundary_score = (mx - comp_scores) / (mx - mn)
    # normalize 0..1
    mn_s = comp_boundary_score.min()
    rng = comp_boundary_score.max() - mn_s
    if rng < 1e-12:
        comp_boundary_score = np.clip(comp_boundary_score, 0.0, 1.0)
    else:
        comp_boundary_score = (comp_boundary_score - mn_s) / (rng + 1e-12)

    p_boundary = post @ comp_boundary_score
    reliability = float(np.mean(post.max(axis=1)))
    p_boundary = np.clip(p_boundary, 0.0, 1.0)
    return p_boundary, reliability


# ---------------------------
# Detector: DBSCAN based
# ---------------------------

def run_dbscan_based(X: np.ndarray, eps: Optional[float] = None, min_samples: int = 5, random_state: Optional[int] = None) -> Tuple[np.ndarray, float]:
    """Run DBSCAN on standardized features and compute per-sample boundary probability.

    If no meaningful clusters are found, fallback to robust zscore heuristic.
    random_state used for deterministic sampling when estimating eps if eps is None.
    """
    if X is None or X.shape[0] == 0:
        return np.array([]), 0.0
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    # Heuristic for eps: median pairwise distance * 0.5 if not provided
    if eps is None:
        try:
            # compute approximate pairwise distances on a sample to avoid O(n^2) blowup
            m = min(500, Xs.shape[0])  # reduced default sample for speed and memory
            rng = np.random.default_rng(random_state)
            if m < Xs.shape[0]:
                sample = Xs[rng.choice(Xs.shape[0], m, replace=False)]
            else:
                sample = Xs
            dists = np.sqrt(((sample[:, None, :] - sample[None, :, :]) ** 2).sum(axis=2))
            eps = float(np.median(dists)) * 0.5 if dists.size else 0.5
        except Exception:
            eps = 0.5
    try:
        db = DBSCAN(eps=eps, min_samples=max(2, min_samples))
        labels = db.fit_predict(Xs)
    except Exception as e:
        logging.warning("DBSCAN failed: %s", e)
        # fallback: zscore heuristic
        arr = X[:, 0].astype(float)
        if arr.std() < 1e-8:
            p = np.full(arr.shape[0], 0.5, dtype=float)
            return p, 0.1
        from scipy.stats import zscore
        p = np.clip(1.0 - zscore(arr), 0.0, 1.0)
        return p, 0.1

    unique_labels = [l for l in set(labels) if l != -1]
    if not unique_labels:
        # all noise — fallback
        arr = X[:, 0].astype(float)
        if arr.std() < 1e-8:
            p = np.full(arr.shape[0], 0.5, dtype=float)
            return p, 0.1
        from scipy.stats import zscore
        p = np.clip(1.0 - zscore(arr), 0.0, 1.0)
        return p, 0.2

    # compute centroids in scaled space and cluster-level scores based on original feature 0
    cluster_centroids = {}
    cluster_mean_feat0 = {}
    for lab in unique_labels:
        idxs = np.where(labels == lab)[0]
        cluster_centroids[lab] = Xs[idxs].mean(axis=0)
        cluster_mean_feat0[lab] = float(X[idxs, 0].mean())
    arr = np.array([cluster_mean_feat0[l] for l in unique_labels])
    mx, mn = arr.max(), arr.min()
    if mx - mn < 1e-12:
        cl_weights = {l: 1.0 for l in unique_labels}
    else:
        cl_weights = {l: float((mx - cluster_mean_feat0[l]) / (mx - mn)) for l in unique_labels}
    # normalize weights 0..1
    vals = np.array(list(cl_weights.values()))
    vmin, vmax = vals.min(), vals.max()
    if vmax - vmin < 1e-12:
        cl_weights = {l: 1.0 for l in unique_labels}
    else:
        cl_weights = {l: float((cl_weights[l] - vmin) / (vmax - vmin)) for l in unique_labels}

    p = np.zeros(X.shape[0], dtype=float)
    confidences: List[float] = []
    for i in range(X.shape[0]):
        lab = labels[i]
        if lab == -1:
            # noise -> distance to nearest cluster centroid
            dists = [np.linalg.norm(Xs[i] - cluster_centroids[l]) for l in unique_labels]
            nearest = unique_labels[int(np.argmin(dists))]
            dist = float(np.min(dists))
            conf = 1.0 / (1.0 + dist)
            p[i] = cl_weights.get(nearest, 0.0) * conf
            confidences.append(conf)
        else:
            dist = np.linalg.norm(Xs[i] - cluster_centroids[lab])
            conf = 1.0 / (1.0 + dist)
            p[i] = cl_weights.get(lab, 0.0) * conf
            confidences.append(conf)
    reliability = float(np.mean(confidences)) if confidences else 0.0
    p = np.clip(p, 0.0, 1.0)
    return p, reliability


# ---------------------------
# Detector: Z-score on bigram NPMI (feature 0)
# ---------------------------

def run_zscore_npmi(X: np.ndarray) -> Tuple[np.ndarray, float]:
    if X is None or X.shape[0] == 0:
        return np.array([]), 0.0
    arr = X[:, 0].astype(float)
    if arr.size == 0:
        return np.array([]), 0.0
    std = arr.std()
    if std < 1e-8:
        # degenerate
        p = np.full(arr.shape[0], 0.5, dtype=float)
        return p, 0.2
    zs = (arr - arr.mean()) / (std + 1e-12)
    p = np.clip(1.0 - norm.cdf(zs), 0.0, 1.0)
    reliability = float(np.mean(np.abs(zs) / (1 + np.abs(zs))))
    return p, reliability


# ---------------------------
# Ensemble combination (skip missing detectors gracefully)
# ---------------------------

def combine_ensemble(probs_list: List[np.ndarray], reliabilities: List[float]) -> np.ndarray:
    # Filter out empty arrays and corresponding reliabilities
    filtered = [(p, r) for p, r in zip(probs_list, reliabilities) if p is not None and p.size > 0]
    if not filtered:
        return np.array([])
    ps, rs = zip(*filtered)
    n = ps[0].shape[0]
    # Ensure all arrays have same length
    lengths = [p.shape[0] for p in ps]
    if len(set(lengths)) != 1:
        raise ValueError(f"All probability arrays must have equal length. Found detector output lengths: {lengths}")
    rel = np.array(rs, dtype=float)
    # avoid zeros: add small epsilon
    if rel.sum() <= 0:
        weights = np.ones_like(rel)
    else:
        weights = rel / (rel.sum() + 1e-12)
    stacked = np.vstack(ps)
    weighted = (weights.reshape(-1, 1) * stacked).sum(axis=0) / (weights.sum() + 1e-12)
    return np.clip(weighted, 0.0, 1.0)


# ---------------------------
# Convert predicted boundaries into output segments string
# Guarantee at least one boundary per line
# ---------------------------
def generate_output_lines(sequences: List[List[int]], indices: List[Tuple[int, int]], final_probs: np.ndarray, threshold: float = 0.5) -> List[str]:
    per_seq = defaultdict(list)
    # indices aligns with final_probs
    if final_probs is None:
        final_probs = np.array([])
    for (seq_idx, pos), prob in zip(indices, final_probs):
        per_seq[seq_idx].append((pos, float(prob)))
    output_lines: List[str] = []
    for sidx, seq in enumerate(sequences):
        n = len(seq)
        if n == 0:
            output_lines.append("")
            continue
        candidates = sorted(per_seq.get(sidx, []), key=lambda x: x[0])
        chosen_positions: List[Tuple[int, float]] = []
        if candidates:
            for pos, prob in candidates:
                if prob >= threshold:
                    chosen_positions.append((pos, prob))
            if not chosen_positions:
                # choose highest-prob candidate
                best = max(candidates, key=lambda x: x[1])
                chosen_positions = [best]
        else:
            # no candidates (short seqs) -> force a mid split
            mid = max(1, n // 2)
            chosen_positions = [(mid, 0.0)]
        chosen_positions.sort(key=lambda x: x[0])
        bps = [p for p, _ in chosen_positions]
        segments: List[str] = []
        start = 0
        for bp in bps:
            seg = seq[start:bp]
            if seg:
                segments.append(", ".join(str(x) for x in seg))
            start = bp
        last = seq[start:]
        if last:
            segments.append(", ".join(str(x) for x in last))
        line = " | ".join(segments)
        output_lines.append(line)
    return output_lines


# ---------------------------
# Diagnostics (plotting)
# ---------------------------

def make_diagnostics(output_dir: str, X: np.ndarray, indices: List[Tuple[int, int]], probs_dict: Dict[str, np.ndarray], sequences: List[List[int]]):
    """Create diagnostic plots and CSVs to help tune the pipeline.

    - Saves histograms, scatterplots, and a CSV of candidates.
    - If matplotlib not available, gracefully skip plotting.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Save candidate CSV
    try:
        # Efficient CSV creation without building a huge list in memory
        if X is not None and X.shape[0] > 0:
            cols = [f'f{i}' for i in range(X.shape[1])]
            df = pd.DataFrame(X, columns=cols)
            df['seq_idx'] = [s for (s, p) in indices]
            df['pos'] = [p for (s, p) in indices]
            df.to_csv(os.path.join(output_dir, 'candidates.csv'), index=False)
    except Exception as e:
        logging.warning("Failed to write candidates CSV: %s", e)
    if not PLOTTING_AVAILABLE:
        logging.info("Matplotlib not available; skipping plots.")
        return
    try:
        # Histogram for feature 0
        if X is not None and X.shape[0] > 0:
            plt.figure(figsize=(6, 4))
            plt.hist(X[:, 0], bins=60)
            plt.title('Histogram: bigram NPMI (feature 0)')
            plt.xlabel('NPMI')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'hist_npmi_feature0.png'))
            plt.close()

            # Scatter: feature0 vs abs_delta_entropy (f4)
            plt.figure(figsize=(6, 4))
            plt.scatter(X[:, 0], X[:, 4], s=6)
            plt.title('feature0 vs abs_delta_entropy')
            plt.xlabel('feature0 (bigram npmi)')
            plt.ylabel('abs_delta_entropy')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'scatter_npmi_entropy.png'))
            plt.close()

        # Detector probability distributions
        for name, p in probs_dict.items():
            if p is None or p.size == 0:
                continue
            plt.figure(figsize=(6, 4))
            plt.hist(p, bins=50)
            plt.title(f'Probability histogram: {name}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'hist_probs_{name}.png'))
            plt.close()
    except Exception as e:
        logging.warning("Diagnostics plotting failed: %s", e)


# ---------------------------
# Main pipeline
# ---------------------------

def main(
    entropy_path: str,
    diversity_path: str,
    seq_path: str,
    out_path: str,
    diag_dir: Optional[str] = None,
    threshold: float = 0.5,
    dbg: bool = False,
    save_diagnostics: bool = True,
    random_seed: int = 42,
):
    configure_logging(logging.DEBUG if dbg else logging.INFO)
    logging.info("boundary.py started")

    entropy_map = load_scalar_map(entropy_path) if entropy_path else {}
    diversity_map = load_scalar_map(diversity_path) if diversity_path else {}
    if not entropy_map:
        logging.warning("Entropy map empty. Entropy features will fall back to mean=0.0")
    if not diversity_map:
        logging.warning("Diversity map empty. Diversity features will fall back to mean=0.0")

    # Load sequences robustly
    sequences: List[List[int]] = []
    if not seq_path or not os.path.exists(seq_path):
        raise FileNotFoundError(f"Sequence file not found: {seq_path}")
    # Try pandas first (handles CSV files)
    try:
        df = pd.read_csv(seq_path, header=None, dtype=str)
        for _, row in df.iterrows():
            val = None
            # first non-null cell
            for c in row:
                if pd.notna(c):
                    val = str(c)
                    break
            if val is None:
                sequences.append([])
            else:
                sequences.append(parse_sequence_line(val))
    except Exception as e:
        logging.warning("pandas.read_csv failed for sequences: %s; falling back to line reader", e)
        with open(seq_path, 'r', encoding='utf-8') as fh:
            for line in fh:
                sequences.append(parse_sequence_line(line))

    logging.info("Loaded %d sequences", len(sequences))

    # Compute corpus stats
    uni, bi, tri, (starts, ends) = compute_counts(sequences)
    logging.info("Corpus: %d unique symbols, %d bigrams, %d trigrams", len(uni), len(bi), len(tri))

    npmi_bi = compute_npmi_for_bigrams(uni, bi, alpha=1.0)
    npmi_tri = compute_npmi_for_trigrams(uni, tri, alpha=1.0)
    logging.info("Computed NPMI maps (bigram/trigram) with sizes %d/%d", len(npmi_bi), len(npmi_tri))

    # Extract features
    X, indices = extract_features_for_all_sequences(
        sequences, uni, bi, tri, npmi_bi, npmi_tri, entropy_map, diversity_map, starts, ends, alpha=1.0
    )
    logging.info("Extracted features: X.shape=%s, indices=%d", X.shape, len(indices))

    # Run detectors
    # set random seed for reproducibility
    np.random.seed(random_seed)
    p_bgm, rel_bgm = run_bayesian_gmm(X, n_components=8, random_state=random_seed, max_iter=500)
    logging.debug("BGM reliability=%.4f", rel_bgm)
    p_db, rel_db = run_dbscan_based(X, eps=None, min_samples=4, random_state=random_seed)
    logging.debug("DBSCAN reliability=%.4f", rel_db)
    p_z, rel_z = run_zscore_npmi(X)
    logging.debug("Zscore reliability=%.4f", rel_z)

    # Combine ensemble
    final_p = combine_ensemble([p_bgm, p_db, p_z], [rel_bgm, rel_db, rel_z])
    logging.info("Combined probabilities computed for %d candidates", final_p.shape[0] if final_p is not None else 0)

    # Generate outputs
    output_lines = generate_output_lines(sequences, indices, final_p, threshold=threshold)

    # Ensure directory
    out_dir = os.path.dirname(out_path) or '.'
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as fh:
        for line in output_lines:
            fh.write(line.strip() + '\n')
    logging.info("Wrote %d lines to %s", len(output_lines), out_path)

    # Save diagnostics
    if save_diagnostics:
        diag_dir = diag_dir or DIAG_DIR_DEFAULT
        probs_dict = {'bgm': p_bgm, 'dbscan': p_db, 'zscore': p_z, 'final': final_p}
        try:
            make_diagnostics(diag_dir, X, indices, probs_dict, sequences)
            logging.info("Saved diagnostics to %s", diag_dir)
        except Exception as e:
            logging.warning("Failed to create diagnostics: %s", e)

    logging.info("boundary.py finished")
    return output_lines


# ---------------------------
# Lightweight tests / example runner
# ---------------------------

def _example_run():
    # create a small artificial dataset and run pipeline for smoke test
    sequences = [
        [1, 2, 3, 4],
        [5, 6, 7],
        [1, 9, 2, 8, 3],
        [],
        [10],
    ]
    # create dummy entropy/diversity maps
    entropy_map = {i: float((i % 5) + 0.1) for seq in sequences for i in seq}
    diversity_map = {i: float((i % 7) + 0.2) for seq in sequences for i in seq}
    uni, bi, tri, (starts, ends) = compute_counts(sequences)
    npmi_bi = compute_npmi_for_bigrams(uni, bi)
    npmi_tri = compute_npmi_for_trigrams(uni, tri)
    X, indices = extract_features_for_all_sequences(sequences, uni, bi, tri, npmi_bi, npmi_tri, entropy_map, diversity_map, starts, ends)
    p_bgm, r_bgm = run_bayesian_gmm(X)
    p_db, r_db = run_dbscan_based(X, random_state=42)
    p_z, r_z = run_zscore_npmi(X)
    final = combine_ensemble([p_bgm, p_db, p_z], [r_bgm, r_db, r_z])
    lines = generate_output_lines(sequences, indices, final, threshold=0.5)
    print("Example output:")
    for l in lines:
        print(l)


# ---------------------------
# CLI entrypoint
# ---------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Boundary detector v2.1 (fixed & extended)")
    parser.add_argument("--entropy", type=str, default=ENTROPY_FILE_DEFAULT, help="Positional entropy file (CSV: symbol,value)")
    parser.add_argument("--diversity", type=str, default=DIVERSITY_FILE_DEFAULT, help="Diversity scores file (CSV: symbol,value)")
    parser.add_argument("--sequences", type=str, default=SEQUENCE_FILE_DEFAULT, help="Sequences file (CSV or plain text). Each row one sequence.")
    parser.add_argument("--output", type=str, default=OUTPUT_FILE_DEFAULT, help="Output file path")
    parser.add_argument("--diag", type=str, default=DIAG_DIR_DEFAULT, help="Diagnostics output directory")
    parser.add_argument("--threshold", type=float, default=0.5, help="Boundary probability threshold")
    parser.add_argument("--no-diag", dest='no_diag', action='store_true', help="Do not save diagnostics")
    parser.add_argument("--debug", dest='debug', action='store_true', help="Enable debug logging")
    parser.add_argument("--example", dest='example', action='store_true', help="Run example smoke test and exit")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for stochastic components (DBSCAN sampling, BGM)")
    args = parser.parse_args()

    if args.example:
        configure_logging(logging.DEBUG)
        _example_run()
        sys.exit(0)

    configure_logging(logging.DEBUG if args.debug else logging.INFO)

    try:
        main(
            entropy_path=args.entropy,
            diversity_path=args.diversity,
            seq_path=args.sequences,
            out_path=args.output,
            diag_dir=None if args.no_diag else args.diag,
            threshold=args.threshold,
            dbg=args.debug,
            save_diagnostics=not args.no_diag,
            random_seed=args.random_seed,
        )
    except Exception as e:
        logging.exception("Fatal error in boundary pipeline: %s", e)
        sys.exit(2)

# End of file
