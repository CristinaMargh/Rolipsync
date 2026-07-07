# Guided recognition and assisted annotation

This folder contains scripts for guided visual recognition experiments.

The goal is not free transcription. Instead, the model receives a known list of candidate words and uses visual CTC scores to rank or localize them.

## 1. Guided keyword recognition

In the keyword spotting experiment, each video segment has:

- positive words from the transcript/alignment
- negative distractor words from a vocabulary list
- visual input from the mouth ROI `.npz`

The model scores each candidate word and ranks them by CTC compatibility with the video.

Main scripts:

```text
ctc_keyword_spot.py
eval_keyword_spotting_v3_segment.py
