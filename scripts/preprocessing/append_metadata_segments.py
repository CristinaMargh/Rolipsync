Încearcă AI direct în aplicațiile preferate … Folosește Gemini pentru a genera schițe și a rafina conținut și beneficiază de Gemini Pro cu acces la AI de ultimă generație de la Google
1
100 %
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--old_meta", required=True, help="existing metadata_train_final_segments.csv")
    ap.add_argument("--new_meta", required=True, help="metadata_new_segments_vad.csv with filled text")
    ap.add_argument("--out_meta", required=True)
    args = ap.parse_args()

    old = pd.read_csv(args.old_meta, encoding="utf-8")
    new = pd.read_csv(args.new_meta, encoding="utf-8")

    # map to your schema: utt_id, raw_video, raw_audio, text
    # new segments live under segments/
    new2 = pd.DataFrame({
        "utt_id": new["utt_id"].astype(str),
        "raw_video": new["utt_id"].astype(str) + ".mp4",
        "raw_audio": new["utt_id"].astype(str) + ".wav",
        "text": new["text"].fillna("").astype(str)
    })

    # sanity: drop empty text
    new2 = new2[new2["text"].str.strip().astype(bool)]

    out = pd.concat([old, new2], ignore_index=True)
    out.to_csv(args.out_meta, index=False, encoding="utf-8")
    print("[DONE] wrote", args.out_meta, "rows:", len(out), "(added:", len(new2), ")")

if __name__ == "__main__":
    main()
