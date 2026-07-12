Încearcă AI direct în aplicațiile preferate … Folosește Gemini pentru a genera schițe și a rafina conținut și beneficiază de Gemini Pro cu acces la AI de ultimă generație de la Google
1
100 %
import os
import re
import json
import math
import random
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import sentencepiece as spm


# =========================================================
# Utils
# =========================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

_ALLOWED_RE = re.compile(r"[^A-Za-zĂÂÎȘȚăâîșț0-9\s\.,!\?:;\-\'\"]+")

def normalize_text(s: str) -> str:
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("’", "'").replace("`", "'")
    s = s.replace("„", '"').replace("”", '"')
    s = s.replace("ş", "ș").replace("ţ", "ț").replace("Ş", "Ș").replace("Ţ", "Ț")
    s = _ALLOWED_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# =========================================================
# Tokenizer
# =========================================================
class SPTokenizer:
    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(model_path))
        self.sp_size = self.sp.get_piece_size()
        self.blank_id = 0

    @property
    def vocab_size_with_blank(self):
        return self.sp_size + 1

    def encode(self, text: str):
        ids = self.sp.encode(normalize_text(text), out_type=int)
        return [i + 1 for i in ids]

    def decode(self, ids):
        sp_ids = [i - 1 for i in ids if i > 0]
        if not sp_ids:
            return ""
        return re.sub(r"\s+", " ", self.sp.decode(sp_ids)).strip()


# =========================================================
# Dataset
# =========================================================
@dataclass
class Sample:
    utt_id: str
    wav_path: str
    text: str


class AudioCTCDataset(Dataset):
    def __init__(self, samples, tok, sample_rate=16000, max_sec=15.0):
        self.samples = samples
        self.tok = tok
        self.sample_rate = sample_rate
        self.max_len = int(sample_rate * max_sec)

        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=80
        )
        self.db = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        wav, sr = torchaudio.load(s.wav_path)
        wav = wav.mean(dim=0, keepdim=True)

        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        wav = wav[:, :self.max_len]
        mel = self.db(self.melspec(wav)).squeeze(0).transpose(0, 1)  # (T, 80)

        token_ids = self.tok.encode(s.text)

        return {
            "utt_id": s.utt_id,
            "mel": mel,
            "mel_len": mel.shape[0],
            "text": s.text,
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "token_len": len(token_ids),
        }


def collate_fn(batch):
    B = len(batch)
    mel_lens = [b["mel_len"] for b in batch]
    tok_lens = [b["token_len"] for b in batch]
    max_mel = max(mel_lens)
    max_tok = max(tok_lens) if tok_lens else 1

    mels = torch.zeros(B, max_mel, 80, dtype=torch.float32)
    tokens = torch.zeros(B, max_tok, dtype=torch.long)

    utt_ids = []
    texts = []

    for i, b in enumerate(batch):
        T = b["mel_len"]
        L = b["token_len"]
        mels[i, :T] = b["mel"]
        if L > 0:
            tokens[i, :L] = b["token_ids"]
        utt_ids.append(b["utt_id"])
        texts.append(b["text"])

    return {
        "utt_ids": utt_ids,
        "texts": texts,
        "mels": mels,
        "mel_lens": torch.tensor(mel_lens, dtype=torch.long),
        "tokens": tokens,
        "token_lens": torch.tensor(tok_lens, dtype=torch.long),
    }


# =========================================================
# Model
# =========================================================
class AudioTeacherCTC(nn.Module):
    def __init__(self, vocab_size_with_blank, hidden=256, num_layers=2, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(80, hidden)
        self.enc = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.out = nn.Linear(hidden * 2, vocab_size_with_blank)

    def forward(self, x, lengths):
        x = self.in_proj(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.enc(packed)
        out, out_lens = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        logits = self.out(out)
        return logits, out_lens


# =========================================================
# Decode / metrics
# =========================================================
def greedy_decode_ids(logits_btV, blank_id=0):
    ids = logits_btV.argmax(dim=-1).detach().cpu().tolist()
    outs = []
    for seq in ids:
        prev = None
        cur = []
        for x in seq:
            if x != blank_id and x != prev:
                cur.append(x)
            prev = x
        outs.append(cur)
    return outs

def _edit_distance(a, b):
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[m]

def cer(pred, ref):
    pred = str(pred).strip()
    ref = str(ref).strip()
    if len(ref) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return _edit_distance(list(pred), list(ref)) / max(1, len(ref))

def wer(pred, ref):
    pred_w = str(pred).strip().split()
    ref_w = str(ref).strip().split()
    if len(ref_w) == 0:
        return 0.0 if len(pred_w) == 0 else 1.0
    return _edit_distance(pred_w, ref_w) / max(1, len(ref_w))


# =========================================================
# Split
# =========================================================
def make_or_load_split(samples, run_dir, val_ratio, seed, keep_val_fixed):
    os.makedirs(run_dir, exist_ok=True)
    split_path = os.path.join(run_dir, "split.json")

    by_id = {s.utt_id: s for s in samples}
    all_ids = sorted(by_id.keys())

    if keep_val_fixed and os.path.exists(split_path):
        obj = json.load(open(split_path, "r", encoding="utf-8"))
        val_ids = set(obj.get("val_utt_ids", []))
        val = [by_id[i] for i in all_ids if i in val_ids]
        tr = [by_id[i] for i in all_ids if i not in val_ids]
        if len(val) > 0:
            print(f"[SPLIT] loaded fixed split: train={len(tr)} val={len(val)}")
            return tr, val

    rnd = random.Random(seed)
    ids = all_ids[:]
    rnd.shuffle(ids)
    n_val = max(1, int(len(ids) * val_ratio))
    n_val = min(n_val, len(ids) - 1)

    val_ids = set(ids[:n_val])
    tr_ids = ids[n_val:]

    tr = [by_id[i] for i in tr_ids]
    val = [by_id[i] for i in ids if i in val_ids]

    obj = {"seed": seed, "val_ratio": val_ratio, "val_utt_ids": sorted(list(val_ids))}
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

    print(f"[SPLIT] created split: train={len(tr)} val={len(val)} saved to {split_path}")
    return tr, val


# =========================================================
# Main
# =========================================================
def build_samples(meta_csv, wav_dir=None):
    df = pd.read_csv(meta_csv)
    df["utt_id"] = df["utt_id"].astype(str)
    df["text"] = df["text"].astype(str).map(normalize_text)

    samples = []
    missing = 0

    has_raw_audio_path = "raw_audio_path" in df.columns

    for _, r in df.iterrows():
        utt = r["utt_id"]
        text = r["text"]

        if has_raw_audio_path and pd.notna(r["raw_audio_path"]):
            wav_path = str(r["raw_audio_path"]).strip()
        else:
            wav_path = os.path.join(wav_dir, f"{utt}.wav")

        if not os.path.exists(wav_path):
            missing += 1
            continue

        samples.append(Sample(utt_id=utt, wav_path=wav_path, text=text))

    print(f"[DATA] rows in metadata: {len(df)}")
    print(f"[DATA] usable audio samples: {len(samples)}")
    print(f"[DATA] missing wav: {missing}")
    return samples


def evaluate(model, dl, tok, device):
    model.eval()
    cers, wers = [], []

    with torch.no_grad():
        for batch in tqdm(dl, desc="eval", leave=False):
            mels = batch["mels"].to(device)
            mel_lens = batch["mel_lens"].to(device)

            logits, out_lens = model(mels, mel_lens)
            pred_ids = greedy_decode_ids(logits, blank_id=tok.blank_id)

            for ids, ref in zip(pred_ids, batch["texts"]):
                pred = tok.decode(ids)
                cers.append(cer(pred, ref))
                wers.append(wer(pred, ref))

    return float(np.mean(cers)), float(np.mean(wers))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--wav_dir", required=True)
    ap.add_argument("--spm", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--keep_val_fixed", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.run_dir, exist_ok=True)

    tok = SPTokenizer(args.spm)
    samples = build_samples(args.meta, args.wav_dir)
    tr_samples, val_samples = make_or_load_split(
        samples, args.run_dir, args.val_ratio, args.seed, args.keep_val_fixed
    )

    train_ds = AudioCTCDataset(tr_samples, tok)
    val_ds = AudioCTCDataset(val_samples, tok)

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[DEVICE]", device)

    model = AudioTeacherCTC(tok.vocab_size_with_blank).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    ctc = nn.CTCLoss(blank=tok.blank_id, zero_infinity=True)

    history = []
    best_cer = 1e9

    for epoch in range(args.epochs):
        model.train()
        losses = []

        for batch in tqdm(train_dl, desc=f"train e{epoch+1}/{args.epochs}", leave=False):
            mels = batch["mels"].to(device)
            mel_lens = batch["mel_lens"].to(device)
            tokens = batch["tokens"].to(device)
            token_lens = batch["token_lens"].to(device)

            logits, out_lens = model(mels, mel_lens)
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)

            loss = ctc(log_probs, tokens, out_lens, token_lens)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            losses.append(loss.item())

        val_cer, val_wer = evaluate(model, val_dl, tok, device)
        train_loss = float(np.mean(losses)) if losses else None

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_cer": val_cer,
            "val_wer": val_wer,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(os.path.join(args.run_dir, "history.csv"), index=False)

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "cfg": vars(args),
            "best_metric": best_cer,
        }
        torch.save(ckpt, os.path.join(args.run_dir, "last.pt"))

        if val_cer < best_cer:
            best_cer = val_cer
            ckpt["best_metric"] = best_cer
            torch.save(ckpt, os.path.join(args.run_dir, "best.pt"))

        print(f"[EPOCH {epoch}] train_loss={train_loss:.4f} val_CER={val_cer:.4f} val_WER={val_wer:.4f} best_CER={best_cer:.4f}")

    with open(os.path.join(args.run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    print("[DONE] run_dir:", args.run_dir)

if __name__ == "__main__":
    main()
