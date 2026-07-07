Încearcă AI direct în aplicațiile preferate … Folosește Gemini pentru a genera schițe și a rafina conținut și beneficiază de Gemini Pro cu acces la AI de ultimă generație de la Google
1
100 %
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse, glob
import numpy as np
import cv2
from tqdm import tqdm

import mediapipe as mp

# Mouth landmark indices in MediaPipe FaceMesh
# We'll use a set around lips to compute a bounding box.
MOUTH_IDXS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    13, 312, 311, 310, 415, 308, 291, 375, 321, 405,
    314, 17, 84, 181, 91, 146, 61
]

def bbox_from_landmarks(lms_xy, w, h, scale=1.4):
    xs = [p[0] for p in lms_xy]
    ys = [p[1] for p in lms_xy]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    bw = (x1 - x0) * scale
    bh = (y1 - y0) * scale
    x0 = int(max(0, cx - bw/2))
    x1 = int(min(w-1, cx + bw/2))
    y0 = int(max(0, cy - bh/2))
    y1 = int(min(h-1, cy + bh/2))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1

def process_video(mp4_path, out_npz, target=96, max_frames=None):
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        return False, "cannot_open"

    mp_face = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    frames_out = []
    n = 0
    last_box = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1

        # optional cap frames
        if max_frames is not None and len(frames_out) >= max_frames:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_face.process(rgb)

        box = None
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            lms_xy = []
            for idx in MOUTH_IDXS:
                px = int(lm[idx].x * w)
                py = int(lm[idx].y * h)
                lms_xy.append((px, py))
            box = bbox_from_landmarks(lms_xy, w, h, scale=1.6)
            if box is not None:
                last_box = box

        # fallback: if no detection, reuse last_box
        if box is None:
            box = last_box

        if box is None:
            # if we never detected mouth, skip this frame
            continue

        x0, y0, x1, y1 = box
        roi = frame[y0:y1, x0:x1]
        if roi.size == 0:
            continue

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (target, target), interpolation=cv2.INTER_AREA)
        frames_out.append(gray)

    cap.release()
    mp_face.close()

    if len(frames_out) < 5:
        return False, f"too_few_frames({len(frames_out)})"

    arr = np.stack(frames_out, axis=0).astype(np.uint8)  # (T,96,96)
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez_compressed(out_npz, frames=arr)
    return True, f"T={arr.shape[0]}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--in_dir", default=None, help="segments/raw_videos (default under root)")
    ap.add_argument("--out_dir", default=None, help="mouth_npz (default under root)")
    ap.add_argument("--target", type=int, default=96)
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    in_dir = args.in_dir or os.path.join(args.root, "segments", "raw_videos")
    out_dir = args.out_dir or os.path.join(args.root, "mouth_npz")

    mp4s = sorted(glob.glob(os.path.join(in_dir, "*.mp4")))
    print("[IN]", in_dir, "num_videos:", len(mp4s))
    ok = 0
    fail = 0

    for mp4 in tqdm(mp4s, desc="mouth_npz"):
        utt = os.path.splitext(os.path.basename(mp4))[0]
        out_npz = os.path.join(out_dir, utt + ".npz")
        if (not args.overwrite) and os.path.exists(out_npz):
            continue
        good, msg = process_video(mp4, out_npz, target=args.target, max_frames=args.max_frames)
        if good:
            ok += 1
        else:
            fail += 1
            # keep going, just report
            # print("[FAIL]", utt, msg)

    print(f"[DONE] ok={ok} fail={fail} saved to {out_dir}")

if __name__ == "__main__":
    main()
