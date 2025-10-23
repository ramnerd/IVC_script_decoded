#!/usr/bin/env python3
"""
main.py

Computes:
1. Distribution Score for each symbol.
2. Normalized Positional Entropy for each symbol.

Assumes you have a dataset of inscriptions in a text/CSV file, where
each line is a sequence of space-separated symbols (or numbers).
"""

import os
import math
from collections import defaultdict, Counter

# ---------- CONFIGURATION ----------
INPUT_FILE = "dataset\\dataset_sanskrit.csv"  # Each line: symbol1 symbol2 symbol3 ...
OUTPUT_DISTRIBUTION = "dataset\\boundary\\sanskrit\\distribution_score.txt"
OUTPUT_ENTROPY = "dataset\\boundary\\sanskrit\\normalized_positional_entropy.txt"

# INPUT_FILE = "dataset\\dataset_tamil.csv"  # Each line: symbol1 symbol2 symbol3 ...
# OUTPUT_DISTRIBUTION = "dataset\\boundary\\IVC\\distribution_score.txt"
# OUTPUT_ENTROPY = "dataset\\boundary\\IVC\\normalized_positional_entropy.txt"


def load_sequences(file_path):
    """Loads symbol sequences from file."""
    sequences = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            # Remove trailing commas and extra whitespace
            line = line.replace(",", " ").strip()
            tokens = line.split()
            if tokens:
                sequences.append(tokens)
    return sequences


def compute_distribution_score(sequences):
    """Computes frequency-based distribution score."""
    all_symbols = [s for seq in sequences for s in seq]
    total = len(all_symbols)
    counts = Counter(all_symbols)

    # Simple normalized frequency as distribution score
    dist_scores = {sym: counts[sym] / total for sym in counts}
    return dist_scores


def compute_normalized_positional_entropy(sequences):
    """
    Computes normalized positional entropy for each symbol.
    Entropy measures how uniformly a symbol is distributed across positions.
    """
    position_counts = defaultdict(lambda: defaultdict(int))
    max_len = max(len(seq) for seq in sequences)

    for seq in sequences:
        for pos, sym in enumerate(seq):
            position_counts[sym][pos] += 1

    entropies = {}
    for sym, pos_dict in position_counts.items():
        total_occurrences = sum(pos_dict.values())
        probs = [count / total_occurrences for count in pos_dict.values()]

        # Compute Shannon entropy
        entropy = -sum(p * math.log2(p) for p in probs)

        # Normalize by max possible entropy (log2 of max_len positions)
        normalized_entropy = entropy / math.log2(max_len) if max_len > 1 else 0
        entropies[sym] = normalized_entropy

    return entropies


def save_scores(scores, file_path):
    """Saves scores as text file: symbol<TAB>score"""
    with open(file_path, "w", encoding="utf-8") as f:
        for sym, score in sorted(scores.items(), key=lambda x: str(x[0])):
            f.write(f"{sym}\t{score:.6f}\n")


def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    print("[INFO] Loading sequences...")
    sequences = load_sequences(INPUT_FILE)

    print("[INFO] Computing distribution score...")
    dist_scores = compute_distribution_score(sequences)
    save_scores(dist_scores, OUTPUT_DISTRIBUTION)
    print(f"[INFO] Distribution scores saved to {OUTPUT_DISTRIBUTION}")

    print("[INFO] Computing normalized positional entropy...")
    entropy_scores = compute_normalized_positional_entropy(sequences)
    save_scores(entropy_scores, OUTPUT_ENTROPY)
    print(f"[INFO] Normalized positional entropy saved to {OUTPUT_ENTROPY}")

    print("[INFO] Done!")


if __name__ == "__main__":
    main()
