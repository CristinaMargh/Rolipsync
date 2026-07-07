Încearcă AI direct în aplicațiile preferate … Folosește Gemini pentru a genera schițe și a rafina conținut și beneficiază de Gemini Pro cu acces la AI de ultimă generație de la Google
1
100 %
import argparse
import importlib.util
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm


def import_train_module(train_py):
    spec = importlib.util.spec_from_file_location("train_ctc_strong_mod", train_py)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_ctc_strong_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_bbox(s):
    vals = [int(x) for x in s.split(",")]
    if len(vals) != 4:
        raise ValueError("bbox must be x1,y1,x2,y2")
    return vals


def norm_word(w):
    w = str(w).lower().strip()
    w = w.replace("ş", "ș").replace("ţ", "ț")
    w = re.sub(r"[^\wăâîșț]+", "", w, flags=re.UNICODE)
    return w


def crop_video_window(video_path, start, end, bbox, target_h, target_w, max_frames, fps):
    x1, y1, x2, y2 = bbox
    cap = cv2.VideoCapture(video_path)

    fs = max(0, int(math.floor(start * fps)))
    fe = max(fs + 1, int(math.ceil(end * fps)))

    cap.set(cv2.CAP_PROP_POS_FRAMES, fs)

    frames = []
    for _ in range(fs, fe):
        ok, frame = cap.read()
        if not ok:
            break

        H, W = frame.shape[:2]
        xx1 = max(0, min(x1, W - 1))
        xx2 = max(xx1 + 1, min(x2, W))
        yy1 = max(0, min(y1, H - 1))
        yy2 = max(yy1 + 1, min(y2, H))

        crop = frame[yy1:yy2, xx1:xx2]
        if crop.size == 0:
            continue

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_AREA)
        frames.append(gray)

    cap.release()

    if len(frames) == 0:
        return None, 0

    arr = np.stack(frames).astype(np.float32) / 255.0

    raw_T = arr.shape[0]
    if raw_T >= max_frames:
        arr = arr[:max_frames]
        T_eff = max_frames
    else:
        pad = np.zeros((max_frames - raw_T, target_h, target_w), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=0)
        T_eff = raw_T

    x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    video_lens = torch.tensor([T_eff], dtype=torch.long)
    return x, video_lens


def ctc_score_word(model, tok, word, video_tensor, video_len, device, score_mode="raw"):
    ids = tok.encode(word)
    if len(ids) == 0:
        return None, None

    with torch.no_grad():
        logits, out_lens = model(video_tensor.to(device), video_len.to(device))
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
        ctc = torch.nn.CTCLoss(blank=tok.blank_id, reduction="none", zero_infinity=True)

        targets = torch.tensor(ids, dtype=torch.long, device=device)
        target_lens = torch.tensor([len(ids)], dtype=torch.long, device=device)
        loss = ctc(log_probs, targets, out_lens.to(device), target_lens)

    loss_val = float(loss.item())

    if score_mode == "norm_char":
        score = -loss_val / max(1, len(word))
    elif score_mode == "norm_token":
        score = -loss_val / max(1, len(ids))
    else:
        score = -loss_val

    return score, loss_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--segments", required=True)
    ap.add_argument("--train_py", required=True)
    ap.add_argument("--spm", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bbox", required=True)
    ap.add_argument("--roi_name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target_h", type=int, required=True)
    ap.add_argument("--target_w", type=int, required=True)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--window_sec", type=float, default=0.80)
    ap.add_argument("--step_sec", type=float, default=0.25)
    ap.add_argument("--seg_pad", type=float, default=0.0)
    ap.add_argument("--max_frames", type=int, default=200)
    ap.add_argument("--max_segments", type=int, default=30)
    ap.add_argument("--max_words_per_segment", type=int, default=20)
    ap.add_argument("--score_mode", choices=["raw", "norm_char", "norm_token"], default="raw")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    mod = import_train_module(args.train_py)
    tok = mod.SPTokenizer(args.spm)
    model = mod.StrongVideoCTC(tok.vocab_size_with_blank, d_model=256).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()

    print("[LOAD]", args.ckpt)
    print("[LOAD] missing", len(missing), "unexpected", len(unexpected))
    print("[DEVICE]", device)
    print("[ROI]", args.roi_name, args.bbox)

    bbox = parse_bbox(args.bbox)
    seg_df = pd.read_csv(args.segments)

    if args.max_segments > 0:
        seg_df = seg_df.head(args.max_segments)

    rows = []

    for _, seg in tqdm(seg_df.iterrows(), total=len(seg_df)):
        seg_start = float(seg["start"])
        seg_end = float(seg["end"])
        segment_id = str(seg["segment_id"])
        text = str(seg.get("text", ""))

        words = str(seg["candidate_words"]).split()
        words = [norm_word(w) for w in words]
        words = sorted(set([w for w in words if len(w) >= 3]))

        if len(words) > args.max_words_per_segment:
            words = words[:args.max_words_per_segment]

        if not words:
            continue

        search_start = max(0.0, seg_start - args.seg_pad)
        search_end = max(search_start + args.window_sec, seg_end + args.seg_pad)

        windows = []
        t = search_start
        while t + args.window_sec <= search_end + 1e-6:
            windows.append((t, t + args.window_sec))
            t += args.step_sec

        if not windows:
            windows = [(search_start, min(search_end, search_start + args.window_sec))]

        win_data = []
        for ws, we in windows:
            vt, vl = crop_video_window(
                args.video, ws, we, bbox,
                args.target_h, args.target_w,
                args.max_frames, args.fps
            )
            if vt is not None:
                win_data.append((ws, we, vt, vl))

        if not win_data:
            continue

        for word in words:
            scores = []
            for ws, we, vt, vl in win_data:
                score, loss = ctc_score_word(
                    model, tok, word, vt, vl,
                    device=device,
                    score_mode=args.score_mode
                )
                if score is not None:
                    scores.append((score, loss, ws, we))

            if not scores:
                continue

            scores_sorted = sorted(scores, key=lambda x: x[0], reverse=True)
            best_score, best_loss, best_s, best_e = scores_sorted[0]
            second_score = scores_sorted[1][0] if len(scores_sorted) > 1 else None
            margin = None if second_score is None else best_score - second_score

            rows.append({
                "roi": args.roi_name,
                "segment_id": segment_id,
                "segment_start": seg_start,
                "segment_end": seg_end,
                "text": text,
                "word": word,
                "estimated_start": best_s,
                "estimated_end": best_e,
                "estimated_center": (best_s + best_e) / 2,
                "best_score": best_score,
                "best_loss": best_loss,
                "second_best_score": second_score,
                "margin": margin,
                "num_windows": len(win_data),
                "num_candidate_words_in_segment": len(words),
                "score_mode": args.score_mode,
                "window_sec": args.window_sec,
                "step_sec": args.step_sec,
            })

    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    print("Saved:", args.out)
    print("Rows:", len(out))
    if len(out):
        print(out.head().to_string(index=False))


if __name__ == "__main__":
    main()
