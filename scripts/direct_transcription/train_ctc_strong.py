Încearcă AI direct în aplicațiile preferate … Folosește Gemini pentru a genera schițe și a rafina conținut și beneficiază de Gemini Pro cu acces la AI de ultimă generație de la Google
1
100 %
import os
import re
import json
import random
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    npz_path: str
    text: str

class VideoDataset(Dataset):
    def __init__(self, samples, tok, max_frames=200):
        self.samples = samples
        self.tok = tok
        self.max_frames = max_frames

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        arr = np.load(s.npz_path)["frames"].astype(np.float32) / 255.0  # (T,H,W)
        x = torch.from_numpy(arr)
        raw_T = x.shape[0]

        if raw_T >= self.max_frames:
            x = x[:self.max_frames]
            T_eff = self.max_frames
        else:
            pad = torch.zeros((self.max_frames - raw_T, x.shape[1], x.shape[2]), dtype=x.dtype)
            x = torch.cat([x, pad], dim=0)
            T_eff = raw_T

        x = x.unsqueeze(0)  # (1,T,H,W)
        token_ids = self.tok.encode(s.text)

        return {
            "utt_id": s.utt_id,
            "video": x,
            "video_len": T_eff,
            "text": s.text,
            "tokens": torch.tensor(token_ids, dtype=torch.long),
            "token_len": len(token_ids),
        }

def collate_fn(batch):
    B = len(batch)
    max_v = max(b["video_len"] for b in batch)
    max_t = max(b["token_len"] for b in batch) if batch else 1
    H = batch[0]["video"].shape[2]
    W = batch[0]["video"].shape[3]

    videos = torch.zeros(B, 1, max_v, H, W, dtype=torch.float32)
    tokens = torch.zeros(B, max_t, dtype=torch.long)

    video_lens, token_lens = [], []
    utt_ids, texts = [], []

    for i, b in enumerate(batch):
        T = b["video_len"]
        videos[i, :, :T] = b["video"][:, :T]
        if b["token_len"] > 0:
            tokens[i, :b["token_len"]] = b["tokens"]

        video_lens.append(T)
        token_lens.append(b["token_len"])
        utt_ids.append(b["utt_id"])
        texts.append(b["text"])

    return {
        "utt_ids": utt_ids,
        "texts": texts,
        "videos": videos,
        "video_lens": torch.tensor(video_lens, dtype=torch.long),
        "tokens": tokens,
        "token_lens": torch.tensor(token_lens, dtype=torch.long),
    }


# =========================================================
# Model
# =========================================================
class Residual2D(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(c)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)

class StrongVideoCTC(nn.Module):
    def __init__(self, vocab_size_with_blank, d_model=256):
        super().__init__()

        # 3D frontend
        self.front3d = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(5,5,5), stride=(1,2,2), padding=(2,2,2)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)),

            nn.Conv3d(32, 64, kernel_size=(3,3,3), stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        # 2D frame tower after collapsing time
        self.frame2d = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            Residual2D(128),

            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            Residual2D(256),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.frame_proj = nn.Linear(256, d_model)

        self.temporal = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )

        self.dropout = nn.Dropout(0.1)
        self.ctc_head = nn.Linear(d_model * 2, vocab_size_with_blank)

    def forward(self, videos_bcthw, video_lens):
        # input: (B,1,T,H,W)
        x = self.front3d(videos_bcthw)            # (B,C,T,H,W)
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous() # (B,T,C,H,W)
        x = x.view(B * T, C, H, W)

        x = self.frame2d(x)
        x = self.pool(x).flatten(1)
        x = self.frame_proj(x)
        x = x.view(B, T, -1)

        packed = nn.utils.rnn.pack_padded_sequence(
            x, video_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.temporal(packed)
        x, out_lens = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        x = self.dropout(x)
        logits = self.ctc_head(x)
        return logits, out_lens


# =========================================================
# Split
# =========================================================
def build_samples(meta_csv, npz_dir):
    df = pd.read_csv(meta_csv)
    df["utt_id"] = df["utt_id"].astype(str)
    df["text"] = df["text"].astype(str).map(normalize_text)

    samples = []
    missing = 0
    empty = 0

    for _, r in df.iterrows():
        utt = r["utt_id"]
        text = r["text"]

        if not text.strip():
            empty += 1
            continue

        npz_path = os.path.join(npz_dir, f"{utt}.npz")
        if not os.path.exists(npz_path):
            missing += 1
            continue

        try:
            arr = np.load(npz_path)["frames"]
            if arr is None or len(arr.shape) < 3 or arr.shape[0] <= 0:
                missing += 1
                continue
        except Exception:
            missing += 1
            continue

        samples.append(Sample(utt_id=utt, npz_path=npz_path, text=text))

    print(f"[DATA] rows in metadata: {len(df)}")
    print(f"[DATA] usable samples: {len(samples)}")
    print(f"[DATA] missing npz: {missing}")
    print(f"[DATA] empty text: {empty}")
    return samples

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
# Eval
# =========================================================
def evaluate(model, dl, tok, device):
    model.eval()
    cers, wers = [], []

    with torch.no_grad():
        for batch in tqdm(dl, desc="eval", leave=False):
            videos = batch["videos"].to(device)
            video_lens = batch["video_lens"].to(device)

            logits, out_lens = model(videos, video_lens)
            pred_ids = greedy_decode_ids(logits, blank_id=tok.blank_id)

            for ids, ref in zip(pred_ids, batch["texts"]):
                pred = tok.decode(ids)
                cers.append(cer(pred, ref))
                wers.append(wer(pred, ref))

    return float(np.mean(cers)), float(np.mean(wers))


# =========================================================
# Main
# =========================================================
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--spm", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--keep_val_fixed", action="store_true")
    ap.add_argument("--max_frames", type=int, default=200)
    ap.add_argument("--init_ckpt", default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    tok = SPTokenizer(args.spm)

    samples = build_samples(args.meta, args.npz_dir)
    tr_samples, val_samples = make_or_load_split(
        samples, args.run_dir, args.val_ratio, args.seed, args.keep_val_fixed
    )

    train_ds = VideoDataset(tr_samples, tok, max_frames=args.max_frames)
    val_ds = VideoDataset(val_samples, tok, max_frames=args.max_frames)

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[DEVICE]", device)

    model = StrongVideoCTC(tok.vocab_size_with_blank, d_model=256).to(device)

    if args.init_ckpt and os.path.exists(args.init_ckpt):
        ckpt = torch.load(args.init_ckpt, map_location="cpu")
        state = ckpt["model"] if "model" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        print("[INIT] loaded:", args.init_ckpt)
        print("[INIT] missing keys:", len(missing))
        print("[INIT] unexpected keys:", len(unexpected))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    ctc = nn.CTCLoss(blank=tok.blank_id, zero_infinity=True)

    best_cer = 1e9
    history = []

    for epoch in range(args.epochs):
        model.train()
        losses = []

        for batch in tqdm(train_dl, desc=f"train e{epoch+1}/{args.epochs}", leave=False):
            videos = batch["videos"].to(device)
            video_lens = batch["video_lens"].to(device)
            tokens = batch["tokens"].to(device)
            token_lens = batch["token_lens"].to(device)

            logits, out_lens = model(videos, video_lens)
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)

            loss = ctc(log_probs, tokens, out_lens, token_lens)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            losses.append(loss.item())

        val_cer, val_wer = evaluate(model, val_dl, tok, device)

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else None,
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

        print(
            f"[EPOCH {epoch}] "
            f"train_loss={row['train_loss']:.4f} "
            f"val_CER={val_cer:.4f} "
            f"val_WER={val_wer:.4f} "
            f"best_CER={best_cer:.4f}"
        )

    with open(os.path.join(args.run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    print("[DONE] run_dir:", args.run_dir)

if __name__ == "__main__":
    main()
