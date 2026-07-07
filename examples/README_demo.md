# Demo commands

## 1. Cut a video into demo segments

```bash
python scripts/prepare_demo_segments.py \
  --input demo_raw/demo1.mp4 \
  --output_dir demo_segments \
  --segment_len 6 \
  --prefix demo1
```

## 2. Convert candidate words to sets

```bash
python scripts/make_word_candidate_set.py \
  --input sign_guided_experiment/outputs/WuKYvYJb56Y_word_candidates.csv \
  --output sign_guided_experiment/outputs/WuKYvYJb56Y_word_candidates_SET.csv
```

## 3. Estimate visual timestamps on interpreter ROI

```bash
python scripts/estimate_visual_word_timestamps.py \
  --video sign_guided_experiment/raw/WuKYvYJb56Y_h264.mp4 \
  --segments sign_guided_experiment/outputs/WuKYvYJb56Y_word_candidates_SET.csv \
  --train_py train_ctc_strong.py \
  --spm spm/ro_vsr.model \
  --ckpt runs/strong_audio_teacher_whisperx_3000/best.pt \
  --bbox 1450,635,1635,815 \
  --roi_name interpreter \
  --out sign_guided_experiment/outputs/WuKYvYJb56Y_interpreter_word_timestamps_SET_pad5.csv \
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
