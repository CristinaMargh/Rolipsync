# Ro Lipsync

Ro Lipsync is a Romanian video-only visual speech recognition project.

The system uses mouth ROI video frames to predict or localize words. Audio is used only for preprocessing, transcription, alignment, pseudo-labeling or assisted evaluation. The final inference setup is video-only.

---

## 1. Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

You also need `ffmpeg` installed.

---

## 2. Repository structure

```text
scripts/
  preprocessing/          data preparation and mouth ROI extraction
  direct_transcription/   video-only CTC transcription
  guided_recognition/     keyword spotting and assisted annotation
  evaluation/             metrics and decoding
  experiments/            exploratory training variants

configs/                  example path/config files
examples/                 demo commands
```

Large files are not stored in this repository. Keep videos, `.npz` files, checkpoints and generated outputs outside Git.

---

## 3. Preprocessing

The preprocessing step converts raw videos into visual mouth ROI inputs.

General flow:

```text
raw video
→ short video segments
→ mouth ROI extraction
→ .npz visual input
→ metadata for training / inference
```

Cut a raw video into short clips:

```bash
python scripts/preprocessing/cut_segments_ffmpeg.py \
  --input data/raw/demo.mp4 \
  --output_dir data/demo_segments \
  --segment_len 6
```

Extract mouth ROI frames from the segments:

```bash
python scripts/preprocessing/extract_mouth_npz_mediapipe.py \
  --input_dir data/demo_segments \
  --output_dir data/demo_npz
```

Each `.npz` file contains a sequence of grayscale mouth ROI frames:

```text
(T, 96, 96)
```

where `T` is the number of frames.

---

## 4. Direct video-only transcription

This mode predicts text directly from a mouth ROI `.npz` file.

Example:

```bash
python scripts/direct_transcription/train_ctc_new.py \
  --root . \
  --spm spm/ro_vsr.model \
  --mode infer_npz \
  --ckpt checkpoints/best.pt \
  --npz data/demo_npz/demo_s00000.npz
```

Expected output: Romanian text predicted from the silent mouth video.

Audio is not used during this step.

---

## 5. Guided keyword recognition

Guided recognition uses a known list of candidate words and ranks them using visual CTC scores.

Run the guided keyword experiment:

```bash
python scripts/guided_recognition/eval_keyword_spotting_v3_segment.py
```

Typical output:

```text
keyword_spotting_v3_segment_results.csv
```

Main metrics:

```text
Recall@1
Recall@5
Recall@10
MRR
```

This experiment is not free transcription. The model receives candidate words and ranks them according to visual compatibility with the video.

---

## 6. Assisted annotation for interpreter ROI

This mode estimates visual timestamps for known words on a selected ROI, for example the face of a sign-language interpreter.

It does not assume that the subtitle word order is the final interpretation order. For each transcript segment, the script creates a set of words and searches each word independently.

First, create a word-set version of the candidate file:

```bash
python scripts/guided_recognition/make_word_candidate_set.py \
  --input outputs/WuKYvYJb56Y_word_candidates.csv \
  --output outputs/WuKYvYJb56Y_word_candidates_SET.csv
```

Then estimate visual timestamps on the interpreter ROI:

```bash
python scripts/guided_recognition/estimate_visual_word_timestamps.py \
  --video raw/WuKYvYJb56Y_h264.mp4 \
  --segments outputs/WuKYvYJb56Y_word_candidates_SET.csv \
  --train_py scripts/direct_transcription/train_ctc_strong.py \
  --spm spm/ro_vsr.model \
  --ckpt checkpoints/best.pt \
  --bbox 1450,635,1635,815 \
  --roi_name interpreter \
  --out outputs/interpreter_word_timestamps.csv \
  --target_h 96 \
  --target_w 96 \
  --fps 25 \
  --window_sec 0.80 \
  --step_sec 0.25 \
  --seg_pad 5.0 \
  --max_segments 30 \
  --max_words_per_segment 20 \
  --score_mode raw \
  --device cuda
```

Output columns:

```text
word
estimated_start
estimated_end
estimated_center
best_score
margin
segment_id
text
```

`best_score` is a CTC log-score. Values are usually negative; closer to zero means a better visual match.

`margin` is the difference between the best temporal window and the second-best one. A higher margin means the selected timestamp is more confident.

If several words receive the same timestamp and the margins are very small, the localization is ambiguous.

---

## 7. Notes about data and checkpoints

Do not commit large or private files to Git:

```text
*.mp4
*.wav
*.npz
*.pt
*.pth
*.ckpt
cookies.txt
outputs/
raw/
audio/
```

Recommended external folders:

```text
data/
checkpoints/
outputs/
```

---

## 8. Project status

The project includes:

```text
Romanian VSR dataset preparation
mouth ROI extraction
SentencePiece tokenization
CTC-based video-only transcription
guided keyword recognition
assisted annotation experiments
```

The assisted annotation experiment is a proof-of-concept. It can generate candidate timestamps, but it is not yet accurate enough for fully automatic annotation.
