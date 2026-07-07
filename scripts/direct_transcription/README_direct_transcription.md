# Direct video-only transcription

This folder contains scripts for Romanian video-only visual speech recognition.

The model receives a mouth ROI `.npz` file and predicts text using a CTC-based model with SentencePiece tokens.

## Example inference

```bash
python scripts/direct_transcription/train_ctc_new.py \
  --root /content/drive/MyDrive/Licenta/ro_vsr \
  --spm /content/drive/MyDrive/Licenta/ro_vsr/spm/ro_vsr.model \
  --mode infer_npz \
  --ckpt /content/drive/MyDrive/Licenta/ro_vsr/DO_NOT_DELETE_best_053/best_053.pt \
  --npz /content/drive/MyDrive/Licenta/ro_vsr/demo_npz/demo1_s00012.npz
