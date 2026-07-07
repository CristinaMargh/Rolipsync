#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare short demo segments from a video file.

This script cuts a longer input video into fixed-length mp4 segments.
It is useful for creating demo clips that can later be processed into
mouth ROI .npz files and passed through the VSR model.

Example:
    python scripts/prepare_demo_segments.py \
        --input demo_raw/demo1.mp4 \
        --output_dir demo_segments \
        --segment_len 6 \
        --prefix demo1
"""

import argparse
import subprocess
from pathlib import Path


def run_cmd(cmd):
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def get_duration(video_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    out = subprocess.check_output(cmd).decode("utf-8").strip()
    return float(out)


def cut_segments(input_path: Path, output_dir: Path, segment_len: float, prefix: str, max_segments: int | None):
    output_dir.mkdir(parents=True, exist_ok=True)

    duration = get_duration(input_path)
    print(f"[INFO] Input video: {input_path}")
    print(f"[INFO] Duration: {duration:.2f} seconds")
    print(f"[INFO] Segment length: {segment_len:.2f} seconds")

    start = 0.0
    idx = 0

    while start < duration:
        if max_segments is not None and idx >= max_segments:
            break

        out_path = output_dir / f"{prefix}_s{idx:05d}.mp4"

        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(start),
            "-i", str(input_path),
            "-t", str(segment_len),
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "128k",
            str(out_path),
        ]

        run_cmd(cmd)

        start += segment_len
        idx += 1

    print(f"[DONE] Created {idx} segments in: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input video file.")
    parser.add_argument("--output_dir", required=True, help="Directory where segments will be saved.")
    parser.add_argument("--segment_len", type=float, default=6.0, help="Segment length in seconds.")
    parser.add_argument("--prefix", default="demo", help="Prefix for output segment names.")
    parser.add_argument("--max_segments", type=int, default=None, help="Optional maximum number of segments.")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    cut_segments(
        input_path=input_path,
        output_dir=output_dir,
        segment_len=args.segment_len,
        prefix=args.prefix,
        max_segments=args.max_segments,
    )


if __name__ == "__main__":
    main()
