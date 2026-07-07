Încearcă AI direct în aplicațiile preferate … Folosește Gemini pentru a genera schițe și a rafina conținut și beneficiază de Gemini Pro cu acces la AI de ultimă generație de la Google
1
100 %
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse, subprocess
import pandas as pd

def cut_one(mp4_in, wav_in, t0, t1, mp4_out, wav_out):
    os.makedirs(os.path.dirname(mp4_out), exist_ok=True)
    os.makedirs(os.path.dirname(wav_out), exist_ok=True)

    # Video segment (re-encode for accurate cuts)
    cmd_v = [
        "ffmpeg","-y",
        "-ss", str(t0), "-to", str(t1),
        "-i", mp4_in,
        "-c:v","libx264","-preset","veryfast","-crf","23",
        "-an",
        mp4_out
    ]
    subprocess.check_call(cmd_v, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Audio segment (wav 16k mono)
    cmd_a = [
        "ffmpeg","-y",
        "-ss", str(t0), "-to", str(t1),
        "-i", wav_in,
        "-ac","1","-ar","16000","-f","wav",
        wav_out
    ]
    subprocess.check_call(cmd_a, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_dir", required=True, help="e.g. <root>/segments")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, encoding="utf-8")

    raw_videos = os.path.join(args.root, "raw_videos")
    raw_audio  = os.path.join(args.root, "raw_audio")

    out_v = os.path.join(args.out_dir, "raw_videos")
    out_a = os.path.join(args.out_dir, "raw_audio")
    os.makedirs(out_v, exist_ok=True)
    os.makedirs(out_a, exist_ok=True)

    need_cols = {"utt_id","src_video","src_audio","start","end"}
    if not need_cols.issubset(set(df.columns)):
        raise ValueError(f"CSV must contain columns {sorted(list(need_cols))}. Found: {df.columns.tolist()}")

    for r in df.itertuples(index=False):
        utt = str(r.utt_id)
        src_vid = os.path.join(raw_videos, str(r.src_video))
        src_wav = os.path.join(raw_audio,  str(r.src_audio))
        t0, t1 = float(r.start), float(r.end)

        if not os.path.exists(src_vid):
            print("[SKIP] missing video:", src_vid); continue
        if not os.path.exists(src_wav):
            print("[SKIP] missing audio:", src_wav); continue

        mp4_out = os.path.join(out_v, f"{utt}.mp4")
        wav_out = os.path.join(out_a, f"{utt}.wav")

        cut_one(src_vid, src_wav, t0, t1, mp4_out, wav_out)

    print("[DONE] cut", len(df), "segments into", args.out_dir)

if __name__ == "__main__":
    main()
