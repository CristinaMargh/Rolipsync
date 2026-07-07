# Preprocessing

This folder contains scripts used to prepare Romanian video data for visual speech recognition.

The goal of preprocessing is to transform raw videos and transcripts into visual training examples.

## Pipeline

```text
raw video
→ short video segments
→ audio / transcript alignment
→ mouth ROI extraction
→ quality filtering
→ .npz visual input
→ metadata for training
