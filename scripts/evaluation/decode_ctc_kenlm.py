Încearcă AI direct în aplicațiile preferate … Folosește Gemini pentru a genera schițe și a rafina conținut și beneficiază de Gemini Pro cu acces la AI de ultimă generație de la Google
1
100 %
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse, re
import numpy as np
import torch
import sentencepiece as spm
from pyctcdecode import build_ctcdecoder

# import model from your training script
# We assume train_ctc_new.py is importable and contains LipReadingCTC
from train_ctc_new import LipReadingCTC, load_ckpt

def spm_labels_with_blank(sp: spm.SentencePieceProcessor):
    # index 0 is CTC blank
    labels = [""]  # blank
    for i in range(sp.get_piece_size()):
        labels.append(sp.id_to_piece(i))
    return labels

def postprocess_spm_text(s: str) -> str:
    # pyctcdecode returns tokens joined with spaces; we trained LM on pieces separated by spaces.
    # After decoding, convert SentencePiece marker ▁ to whitespace.
    s = s.replace("▁", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--spm", required=True)
    ap.add_argument("--kenlm", required=True, help="KenLM binary .bin")
    ap.add_argument("--npz", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--beam_width", type=int, default=50)
    ap.add_argument("--alpha", type=float, default=0.8, help="LM weight")
    ap.add_argument("--beta", type=float, default=1.0, help="word insertion bonus")
    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print("[DEVICE]", device)

    sp = spm.SentencePieceProcessor()
    sp.load(args.spm)

    labels = spm_labels_with_blank(sp)
    decoder = build_ctcdecoder(labels, kenlm_model_path=args.kenlm)

    vocab_with_blank = sp.get_piece_size() + 1
    model = LipReadingCTC(vocab_size_with_blank=vocab_with_blank).to(device)
    load_ckpt(args.ckpt, model, optim=None, sched=None, scaler=None)
    model.eval()

    z = np.load(args.npz, allow_pickle=False)
    frames = z["frames"].astype(np.float32) / 255.0  # (T,H,W)
    frames = np.expand_dims(frames, axis=1)          # (T,1,H,W)
    video = torch.from_numpy(frames).unsqueeze(0).to(device)  # (1,T,1,H,W)
    in_len = torch.tensor([video.shape[1]], device=device, dtype=torch.long)

    with torch.no_grad():
        logits = model(video)  # (1,T,V)

    # pyctcdecode expects (T,V) numpy float
    logp = logits[0].log_softmax(dim=-1).cpu().numpy()

    text = decoder.decode(logp, beam_width=args.beam_width)
    print("RAW:", text)
    print("PRED:", postprocess_spm_text(text))

if __name__ == "__main__":
    main()
