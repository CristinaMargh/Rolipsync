Încearcă AI direct în aplicațiile preferate … Folosește Gemini pentru a genera schițe și a rafina conținut și beneficiază de Gemini Pro cu acces la AI de ultimă generație de la Google
1
100 %
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, random, re
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = "/content/drive/MyDrive/Licenta/ro_vsr"
if ROOT not in sys.path:
    sys.path.append(ROOT)

from ctc_keyword_spot import KeywordSpotter, normalize_word

# -----------------------------
# Config
# -----------------------------
ALIGN_CSV = os.path.join(ROOT, "word_alignments_whisperx_partial.csv")
# Dacă ai generat CSV complet, poți schimba în:
# ALIGN_CSV = os.path.join(ROOT, "word_alignments_whisperx.csv")

NPZ_DIR = os.path.join(ROOT, "mouth_npz")
CKPT = os.path.join(ROOT, "runs", "ctc_conformer_selftrain", "best.pt")
SPM = os.path.join(ROOT, "spm", "ro_vsr.model")

VOCAB_PATH = os.path.join(ROOT, "ro_keyword_vocab_5k.txt")
SYN_PATH = os.path.join(ROOT, "ro_synonyms_seed.csv")

OUT_CSV = os.path.join(ROOT, "keyword_spotting_v3_segment_results.csv")

RANDOM_SEED = 42

MAX_SEGMENTS = 200          # crește dacă vrei mai multe segmente
MAX_POSITIVES = 8           # câte cuvinte reale din segment
NUM_NEGATIVES = 15          # câți distractori din vocabular
TOP_N_SCORE = 50

MIN_WORD_LEN = 5
MIN_ALIGN_SCORE = 0.70
LENGTH_TOL = 3
SCORE_NORM = "token_len"    # "token_len" sau "none"

RO_STOPWORDS = {
    "și","si","să","sa","că","ca","de","din","la","în","in","pe","cu","un","o","a","al","ai","ale",
    "este","e","era","sunt","am","ai","are","au","mai","foarte","nu","da","îmi","imi","mă","ma",
    "te","va","vă","vrei","vreau","pentru","dar","iar","sau","or","către","spre","însă","insa",
    "acum","atunci","aici","acolo","când","cand","unde","cum","ce","cine","fost","fără","fara",
    "asta","ăsta","aceasta","acest","aceste","ăla","ala","tot","toate","toți","toata","lui","mea",
    "mele","tău","tau","lor","noi","voi","ei","ele","mie","tine","mine","care"
}

def clean_word(w):
    w = normalize_word(str(w))
    w = re.sub(r"^\-+|\-+$", "", w)
    return w

def valid_word(w):
    if not w:
        return False
    if len(w) < MIN_WORD_LEN:
        return False
    if w in RO_STOPWORDS:
        return False
    if w.isdigit():
        return False
    if re.search(r"[^a-zăâîșț\-]", w):
        return False
    return True

def unique_keep_order(words):
    seen = set()
    out = []
    for w in words:
        if w not in seen:
            out.append(w)
            seen.add(w)
    return out

def normalize_scores(scored):
    out = []
    for x in scored:
        x = dict(x)
        if SCORE_NORM == "token_len":
            x["score_norm"] = x["score"] / max(1, x.get("token_len", 1))
        else:
            x["score_norm"] = x["score"]
        out.append(x)
    out.sort(key=lambda d: d["score_norm"], reverse=True)
    return out

def recall_at_k(ranked_words, positives, k):
    positives = set(positives)
    return int(any(w in positives for w in ranked_words[:k]))

def reciprocal_rank(ranked_words, positives):
    positives = set(positives)
    for i, w in enumerate(ranked_words, start=1):
        if w in positives:
            return 1.0 / i
    return 0.0

def first_positive_rank(ranked_words, positives):
    positives = set(positives)
    for i, w in enumerate(ranked_words, start=1):
        if w in positives:
            return i
    return None

def load_vocab():
    if not os.path.exists(VOCAB_PATH):
        raise FileNotFoundError(
            f"Nu există vocabularul {VOCAB_PATH}. Creează-l întâi cu scriptul de vocab 5k."
        )
    words = []
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        for line in f:
            w = clean_word(line.strip())
            if valid_word(w):
                words.append(w)
    words = sorted(set(words))
    return words

def load_synonyms():
    """
    Format ro_synonyms_seed.csv:
    doctor,medic
    spital,clinică,clinica
    """
    syn = {}
    if not os.path.exists(SYN_PATH):
        return syn

    with open(SYN_PATH, "r", encoding="utf-8") as f:
        for line in f:
            parts = [clean_word(x) for x in line.strip().split(",")]
            parts = [x for x in parts if valid_word(x)]
            if len(parts) < 2:
                continue
            group = set(parts)
            for p in parts:
                syn.setdefault(p, set()).update(group)
    return syn

def expand_semantic_set(words, synonym_map):
    out = set(words)
    for w in words:
        if w in synonym_map:
            out.update(synonym_map[w])
    return out

def choose_negatives(positives, segment_words, global_words, n=25):
    """
    Distractori:
    - nu apar în segment
    - nu sunt pozitive
    - lungime apropiată de pozitive
    """
    pos_set = set(positives)
    seg_set = set(segment_words)

    pos_lens = [len(w) for w in positives]
    if not pos_lens:
        return []

    min_len = max(1, min(pos_lens) - LENGTH_TOL)
    max_len = max(pos_lens) + LENGTH_TOL

    pool = [
        w for w in global_words
        if w not in seg_set
        and w not in pos_set
        and min_len <= len(w) <= max_len
    ]

    if len(pool) < n:
        pool = [
            w for w in global_words
            if w not in seg_set and w not in pos_set
        ]

    if len(pool) <= n:
        return pool
    return random.sample(pool, n)

def main():
    random.seed(RANDOM_SEED)

    if not os.path.exists(ALIGN_CSV):
        raise FileNotFoundError(f"Missing alignment CSV: {ALIGN_CSV}")

    df = pd.read_csv(ALIGN_CSV)
    required = {"utt_id", "word", "start", "end"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV needs {required}. Found: {df.columns.tolist()}")

    df["word_clean"] = df["word"].apply(clean_word)

    if "score" in df.columns:
        df = df[(df["score"].isna()) | (df["score"] >= MIN_ALIGN_SCORE)]

    df = df[df["word_clean"].apply(valid_word)].copy()

    # Grupăm cuvintele reale per segment
    all_words_by_utt = {}
    for uid, g in df.groupby("utt_id"):
        words = unique_keep_order(g["word_clean"].tolist())
        if len(words) >= 2:
            all_words_by_utt[uid] = words

    utt_ids = sorted(all_words_by_utt.keys())
    utt_ids = [u for u in utt_ids if os.path.exists(os.path.join(NPZ_DIR, u + ".npz"))]

    if MAX_SEGMENTS:
        utt_ids = utt_ids[:MAX_SEGMENTS]

    print("[INFO] usable segments:", len(utt_ids))

    global_words = load_vocab()
    synonym_map = load_synonyms()

    print("[INFO] restricted vocab size:", len(global_words))
    print("[INFO] synonym entries:", len(synonym_map))

    spotter = KeywordSpotter(ROOT, CKPT, SPM)

    rows = []

    for uid in tqdm(utt_ids, desc="keyword_spotting_v3"):
        segment_words = all_words_by_utt[uid]
        positives = segment_words[:MAX_POSITIVES]

        if len(positives) < 2:
            continue

        negatives = choose_negatives(
            positives=positives,
            segment_words=segment_words,
            global_words=global_words,
            n=NUM_NEGATIVES
        )

        if len(negatives) < 5:
            continue

        candidates = positives + negatives
        random.shuffle(candidates)

        npz_path = os.path.join(NPZ_DIR, uid + ".npz")

        try:
            scored = spotter.score_words(
                npz_path=npz_path,
                words=candidates,
                top_n=min(TOP_N_SCORE, len(candidates))
            )
        except Exception as e:
            print("[FAIL]", uid, type(e).__name__, str(e))
            continue

        scored = normalize_scores(scored)
        ranked_words = [x["word"] for x in scored]

        # Strict match
        r1 = recall_at_k(ranked_words, positives, 1)
        r5 = recall_at_k(ranked_words, positives, 5)
        r10 = recall_at_k(ranked_words, positives, 10)
        rr = reciprocal_rank(ranked_words, positives)
        first_rank = first_positive_rank(ranked_words, positives)

        # Semantic match cu sinonime
        semantic_positives = expand_semantic_set(positives, synonym_map)
        sr1 = recall_at_k(ranked_words, semantic_positives, 1)
        sr5 = recall_at_k(ranked_words, semantic_positives, 5)
        sr10 = recall_at_k(ranked_words, semantic_positives, 10)
        srr = reciprocal_rank(ranked_words, semantic_positives)
        sfirst_rank = first_positive_rank(ranked_words, semantic_positives)

        # Ranks stricte pentru pozitive
        ranks = []
        for p in positives:
            if p in ranked_words:
                ranks.append(ranked_words.index(p) + 1)
        mean_pos_rank = float(np.mean(ranks)) if ranks else None

        score_map = {x["word"]: x["score_norm"] for x in scored}
        pos_scores = [score_map[w] for w in positives if w in score_map]
        neg_scores = [score_map[w] for w in negatives if w in score_map]

        rows.append({
            "utt_id": uid,
            "positives": " ".join(positives),
            "semantic_positives": " ".join(sorted(semantic_positives)),
            "negatives_sample": " ".join(negatives[:10]),
            "top_words": " ".join(ranked_words[:10]),
            "top1": ranked_words[0] if ranked_words else "",

            "recall1": r1,
            "recall5": r5,
            "recall10": r10,
            "mrr": rr,
            "first_positive_rank": first_rank if first_rank is not None else -1,
            "mean_positive_rank": mean_pos_rank if mean_pos_rank is not None else -1,

            "semantic_recall1": sr1,
            "semantic_recall5": sr5,
            "semantic_recall10": sr10,
            "semantic_mrr": srr,
            "semantic_first_rank": sfirst_rank if sfirst_rank is not None else -1,

            "avg_positive_score": float(np.mean(pos_scores)) if pos_scores else None,
            "avg_negative_score": float(np.mean(neg_scores)) if neg_scores else None,
            "margin_pos_minus_neg": (float(np.mean(pos_scores)) - float(np.mean(neg_scores))) if (pos_scores and neg_scores) else None,

            "num_pos": len(positives),
            "num_neg": len(negatives),
            "num_candidates": len(candidates),
            "score_norm": SCORE_NORM,
            "min_word_len": MIN_WORD_LEN,
            "min_align_score": MIN_ALIGN_SCORE,
            "length_tol": LENGTH_TOL
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8")

    print("\n[DONE] saved:", OUT_CSV)
    print("segments evaluated:", len(out))

    if len(out):
        print("STRICT Recall@1:", out["recall1"].mean())
        print("STRICT Recall@5:", out["recall5"].mean())
        print("STRICT Recall@10:", out["recall10"].mean())
        print("STRICT MRR:", out["mrr"].mean())
        print("STRICT Mean positive rank:", out["mean_positive_rank"].replace(-1, np.nan).mean())

        print("\nSEMANTIC Recall@1:", out["semantic_recall1"].mean())
        print("SEMANTIC Recall@5:", out["semantic_recall5"].mean())
        print("SEMANTIC Recall@10:", out["semantic_recall10"].mean())
        print("SEMANTIC MRR:", out["semantic_mrr"].mean())

        print("\nMean margin pos-neg:", out["margin_pos_minus_neg"].mean())

        print("\nExamples:")
        print(out.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
