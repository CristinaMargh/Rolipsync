Încearcă AI direct în aplicațiile preferate … Folosește Gemini pentru a genera schițe și a rafina conținut și beneficiază de Gemini Pro cu acces la AI de ultimă generație de la Google
1
100 %
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, re
import numpy as np
import torch
import sentencepiece as spm
from pyctcdecode import build_ctcdecoder

from train_ctc_new import load_ckpt
from model_conformer_ctc import LipReadingCTCConformer

def spm_labels_with_blank(sp):
    # index 0 is CTC blank
    labels = [""]
    for i in range(sp.get_piece_size()):
        labels.append(sp.id_to_piece(i))
    return labels

def postprocess_spm_text(s: str) -> str:
    s = s.replace("▁", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def greedy_from_logits(logits, in_len, sp):
    pred = logits.argmax(dim=-1)  # (1,T)
    T = int(in_len.item())
    seq = pred[0, :T].tolist()
    cleaned = []
    prev = None
    for t in seq:
        if t == prev:
            continue
        prev = t
        if t == 0:
            continue
        cleaned.append(t)
    sp_ids = [i - 1 for i in cleaned if i > 0]
    return sp.decode(sp_ids) if sp_ids else ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--spm", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--beam_width", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print("[DEVICE]", device)

    sp = spm.SentencePieceProcessor()
    sp.load(args.spm)

    vocab_with_blank = sp.get_piece_size() + 1
    model = LipReadingCTCConformer(vocab_size_with_blank=vocab_with_blank).to(device)
    load_ckpt(args.ckpt, model, optim=None, sched=None, scaler=None)
    model.eval()

    z = np.load(args.npz, allow_pickle=False)
    key = "frames" if "frames" in z.files else z.files[0]
    frames = z[key].astype(np.float32) / 255.0  # (T,H,W)
    frames = np.expand_dims(frames, axis=1)     # (T,1,H,W)
    video = torch.from_numpy(frames).unsqueeze(0).to(device)  # (1,T,1,H,W)
    in_len = torch.tensor([video.shape[1]], device=device, dtype=torch.long)

    with torch.no_grad():
        logits = model(video, in_len)  # (1,T,V)
    logp = logits[0].log_softmax(dim=-1).cpu().numpy()  # (T,V)

    # Greedy
    greedy = greedy_from_logits(logits, in_len[0], sp)
    print("GREEDY_RAW:", greedy)
    print("GREEDY:", postprocess_spm_text(greedy))

    # Beam (no LM)
    labels = spm_labels_with_blank(sp)
    decoder = build_ctcdecoder(labels)  # no kenlm
    beam_txt = decoder.decode(logp, beam_width=args.beam_width)
    print("BEAM_RAW:", beam_txt)
    print("BEAM:", postprocess_spm_text(beam_txt))

if __name__ == "__main__":
    main()
