#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create a candidate-word CSV where each segment contains a set of words.

This removes duplicates and avoids using the subtitle order as the final order.
It is useful for assisted annotation on an interpreter ROI.
"""

import argparse
import re
import pandas as pd


def norm_word(w):
    w = str(w).lower().strip()
    w = w.replace("ş", "ș").replace("ţ", "ț")
    w = re.sub(r"[^\wăâîșț]+", "", w, flags=re.UNICODE)
    return w


def make_word_set(x):
    words = [norm_word(w) for w in str(x).split()]
    words = sorted(set([w for w in words if len(w) >= 3]))
    return " ".join(words)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input CSV with candidate_words column.")
    parser.add_argument("--output", required=True, help="Output CSV.")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if "candidate_words" not in df.columns:
        raise ValueError("Input CSV must contain candidate_words column.")

    df["candidate_words"] = df["candidate_words"].apply(make_word_set)
    df["num_candidates"] = df["candidate_words"].apply(lambda x: len(str(x).split()))
    df["source"] = "word_set"

    df.to_csv(args.output, index=False, encoding="utf-8")
    print("Saved:", args.output)
    print("Rows:", len(df))


if __name__ == "__main__":
    main()
