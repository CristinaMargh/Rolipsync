Încearcă AI direct în aplicațiile preferate … Folosește Gemini pentru a genera schițe și a rafina conținut și beneficiază de Gemini Pro cu acces la AI de ultimă generație de la Google
1
100 %
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse, re
import pandas as pd
import sentencepiece as spm
from tqdm import tqdm

def norm_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="CSV with column text")
    ap.add_argument("--spm", required=True, help="SentencePiece .model path")
    ap.add_argument("--out", required=True, help="Output corpus text file")
    args = ap.parse_args()

    df = pd.read_csv(args.meta, encoding="utf-8")
    if "text" not in df.columns:
        raise ValueError("metadata must contain 'text' column")

    sp = spm.SentencePieceProcessor()
    sp.load(args.spm)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for t in tqdm(df["text"].astype(str).tolist(), desc="write_corpus"):
            t = norm_text(t)
            if not t:
                continue
            pieces = sp.encode(t, out_type=str)  # list of pieces (with ▁)
            if not pieces:
                continue
            f.write(" ".join(pieces) + "\n")
            n += 1

    print("[DONE] lines:", n)
    print("[OUT]", args.out)

if __name__ == "__main__":
    main()
