Încearcă AI direct în aplicațiile preferate … Folosește Gemini pentru a genera schițe și a rafina conținut și beneficiază de Gemini Pro cu acces la AI de ultimă generație de la Google
1
100 %
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, random, argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from model_conformer_ctc import LipReadingCTCConformer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import sentencepiece as spm


# -------------------------
# Utils: reproducibility
# -------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# Edit distance (CER/WER)
# -------------------------
def _edit_distance(a: List[Any], b: List[Any]) -> int:
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


def cer(pred: str, ref: str) -> float:
    pred = (pred or "").strip()
    ref  = (ref or "").strip()
    if len(ref) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return _edit_distance(list(pred), list(ref)) / max(1, len(ref))


def wer(pred: str, ref: str) -> float:
    pred_w = (pred or "").strip().split()
    ref_w  = (ref or "").strip().split()
    if len(ref_w) == 0:
        return 0.0 if len(pred_w) == 0 else 1.0
    return _edit_distance(pred_w, ref_w) / max(1, len(ref_w))


# -------------------------
# Dataset
# -------------------------
@dataclass
class Sample:
    utt_id: str
    npz_path: str
    text: str


class RomanianVSRDataset(Dataset):
    """
    Expects:
      - metadata.csv with columns: utt_id, text
      - mouth_npz/<utt_id>.npz with key 'frames' (T, H, W) uint8
    """
    def __init__(
        self,
        samples: List[Sample],
        sp: spm.SentencePieceProcessor,
        max_video_frames: int = 200,
        target_size: int = 96,
        normalize: bool = True,
    ):
        self.samples = samples
        self.sp = sp
        self.max_video_frames = max_video_frames
        self.target_size = target_size
        self.normalize = normalize

    def __len__(self):
        return len(self.samples)

    def _load_frames(self, npz_path: str) -> np.ndarray:
        data = np.load(npz_path, allow_pickle=False)
        key = "frames" if "frames" in data.files else data.files[0]
        frames = data[key]
        if frames.ndim != 3:
            raise ValueError(f"Expected (T,H,W), got {frames.shape} in {npz_path}")

        T, H, W = frames.shape
        if (H != self.target_size) or (W != self.target_size):
            import cv2
            resized = [cv2.resize(frames[i], (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)
                       for i in range(T)]
            frames = np.stack(resized, axis=0)

        # uniform subsample to max_video_frames
        if self.max_video_frames is not None and frames.shape[0] > self.max_video_frames:
            T = frames.shape[0]
            idx = np.linspace(0, T - 1, self.max_video_frames).astype(np.int32)
            frames = frames[idx]

        return frames

    def _encode_text(self, text: str) -> np.ndarray:
        # Reserve 0 for CTC blank; shift spm ids by +1
        ids = self.sp.encode(text, out_type=int)
        ids = [i + 1 for i in ids]
        return np.asarray(ids, dtype=np.int64)

    def __getitem__(self, idx):
        s = self.samples[idx]
        frames = self._load_frames(s.npz_path)  # (T,H,W) uint8
        frames = frames.astype(np.float32) / 255.0 if self.normalize else frames.astype(np.float32)

        frames = np.expand_dims(frames, axis=1)  # (T,1,H,W)
        targets = self._encode_text(s.text)      # (L,)

        return {
            "utt_id": s.utt_id,
            "video": torch.from_numpy(frames),     # (T,1,H,W)
            "targets": torch.from_numpy(targets),  # (L,)
            "text": s.text,
        }


def collate_fn(batch: List[Dict[str, Any]]):
    utt_ids = [b["utt_id"] for b in batch]
    texts   = [b["text"] for b in batch]
    videos  = [b["video"] for b in batch]     # (T,1,H,W)
    targets = [b["targets"] for b in batch]   # (L,)

    T_list = torch.tensor([v.shape[0] for v in videos], dtype=torch.long)
    L_list = torch.tensor([t.numel() for t in targets], dtype=torch.long)

    max_T = int(T_list.max().item())
    C, H, W = videos[0].shape[1], videos[0].shape[2], videos[0].shape[3]

    padded = torch.zeros((len(videos), max_T, C, H, W), dtype=videos[0].dtype)
    for i, v in enumerate(videos):
        padded[i, : v.shape[0]] = v

    cat_targets = torch.cat(targets, dim=0) if len(targets) else torch.empty((0,), dtype=torch.long)

    return {
        "utt_id": utt_ids,
        "video": padded,          # (B,T,1,H,W)
        "input_lengths": T_list,  # (B,)
        "targets": cat_targets,   # (sumL,)
        "target_lengths": L_list, # (B,)
        "text": texts,
    }


# -------------------------
# Model
# -------------------------
class VideoFrontend(nn.Module):
    """
    Simple 3D CNN frontend:
      input: (B, 1, T, H, W)
      output: (B, T, F)
    """
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=(3,5,5), stride=(1,2,2), padding=(1,2,2))
        self.bn1   = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1))
        self.bn2   = nn.BatchNorm3d(64)
        self.conv3 = nn.Conv3d(64, 96, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1))
        self.bn3   = nn.BatchNorm3d(96)

        # input 96x96 -> after /2 /2 /2 => 12x12
        self.proj = nn.Linear(96 * 12 * 12, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        B, C, T, Hp, Wp = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B,T,C,Hp,Wp)
        x = x.view(B, T, C * Hp * Wp)
        x = self.proj(x)                           # (B,T,F)
        return x


class LipReadingCTC(nn.Module):
    def __init__(self, vocab_size_with_blank: int, d_model: int = 256, rnn_hidden: int = 256, rnn_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.frontend = VideoFrontend(out_dim=d_model)
        self.rnn = nn.GRU(
            input_size=d_model,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0.0
        )
        self.classifier = nn.Linear(2 * rnn_hidden, vocab_size_with_blank)
        with torch.no_grad():
            self.classifier.bias.fill_(0.0)
            self.classifier.bias[0] = -2.0

    def forward(self, video_btchw: torch.Tensor) -> torch.Tensor:
        x = video_btchw.permute(0, 2, 1, 3, 4).contiguous()  # (B,1,T,H,W)
        x = self.frontend(x)                                 # (B,T,F)
        x, _ = self.rnn(x)                                   # (B,T,2H)
        logits = self.classifier(x)                           # (B,T,V)
        return logits


# -------------------------
# Decode
# -------------------------
def greedy_decode(logits: torch.Tensor, input_lengths: torch.Tensor, sp: spm.SentencePieceProcessor) -> List[str]:
    probs = logits.log_softmax(dim=-1)
    pred = probs.argmax(dim=-1)  # (B,T)

    out_texts = []
    for b in range(pred.size(0)):
        T = int(input_lengths[b].item())
        seq = pred[b, :T].tolist()

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
        out_texts.append(sp.decode(sp_ids) if len(sp_ids) else "")
    return out_texts


# -------------------------
# Checkpointing (model+optim+sched+scaler)
# -------------------------
def save_ckpt(path: str, model: nn.Module, optim: torch.optim.Optimizer, sched, scaler, epoch: int, best_metric: float, cfg: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "sched": sched.state_dict() if sched is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_metric": best_metric,
        "cfg": cfg,
    }, path)


def load_ckpt(path: str, model: nn.Module, optim: Optional[torch.optim.Optimizer] = None, sched=None, scaler=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    if optim is not None and "optim" in ckpt and ckpt["optim"] is not None:
        optim.load_state_dict(ckpt["optim"])
    if sched is not None and ckpt.get("sched") is not None:
        sched.load_state_dict(ckpt["sched"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt


# -------------------------
# Data building + stable split
# -------------------------
def build_samples(meta_csv: str, npz_dir: str) -> List[Sample]:
    df = pd.read_csv(meta_csv, encoding="utf-8")
    if "utt_id" not in df.columns or "text" not in df.columns:
        raise ValueError(f"metadata must contain columns utt_id and text. Found: {df.columns.tolist()}")

    samples = []
    missing = 0
    empty = 0
    for _, r in df.iterrows():
        utt = str(r["utt_id"]).strip()
        txt = "" if pd.isna(r["text"]) else str(r["text"]).strip()
        if len(txt) == 0:
            empty += 1
            continue

        npz_path = os.path.join(npz_dir, f"{utt}.npz")
        if not os.path.exists(npz_path):
            missing += 1
            continue

        samples.append(Sample(utt_id=utt, npz_path=npz_path, text=txt))

    print(f"[DATA] rows in metadata:     {len(df)}")
    print(f"[DATA] usable samples:      {len(samples)}")
    print(f"[DATA] skipped empty text:  {empty}")
    print(f"[DATA] skipped missing npz: {missing}")
    return samples


def make_or_load_split(samples: List[Sample], run_dir: str, val_ratio: float, seed: int, keep_val_fixed: bool) -> Tuple[List[Sample], List[Sample]]:
    """
    If keep_val_fixed=True:
      - If run_dir/split.json exists: reuse val_utt_ids, put new items into train automatically.
      - Else create a new split and save it.
    """
    os.makedirs(run_dir, exist_ok=True)
    split_path = os.path.join(run_dir, "split.json")

    by_id = {s.utt_id: s for s in samples}
    all_ids = sorted(by_id.keys())

    if keep_val_fixed and os.path.exists(split_path):
        obj = json.load(open(split_path, "r", encoding="utf-8"))
        val_ids = set(obj.get("val_utt_ids", []))
        # Keep val that still exists; anything new goes to train
        val = [by_id[i] for i in all_ids if i in val_ids]
        tr  = [by_id[i] for i in all_ids if i not in val_ids]
        if len(val) == 0:
            # fallback: re-split if val became empty
            keep_val_fixed = False
        else:
            print(f"[SPLIT] loaded fixed split: train={len(tr)} val={len(val)} (new samples go to train)")
            return tr, val

    # Create split
    rnd = random.Random(seed)
    ids = all_ids[:]
    rnd.shuffle(ids)
    if val_ratio <= 0.0:
      # no validation split
      tr_ids = ids
      val_ids = set()
    else:
      n_val = int(len(ids) * val_ratio)
      n_val = max(1, n_val)  # at least 1 in val if val_ratio > 0
      n_val = min(n_val, len(ids) - 1)  # keep at least 1 in train

      val_ids = set(ids[:n_val])
      tr_ids  = ids[n_val:]


    tr = [by_id[i] for i in tr_ids]
    val = [by_id[i] for i in ids if i in val_ids]

    # Save split
    json.dump({"seed": seed, "val_ratio": val_ratio, "val_utt_ids": sorted(list(val_ids))},
              open(split_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[SPLIT] created split: train={len(tr)} val={len(val)} saved to {split_path}")
    return tr, val


# -------------------------
# Eval
# -------------------------
def run_eval(model, loader, device, sp) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_cer = 0.0
    total_wer = 0.0
    n = 0

    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", leave=False):
            video = batch["video"].to(device)
            in_len = batch["input_lengths"].to(device)
            targets = batch["targets"].to(device)
            tgt_len = batch["target_lengths"].to(device)

            logits = model(video)                            # (B,T,V)
            logp = logits.log_softmax(dim=-1).permute(1,0,2)  # (T,B,V)

            loss = ctc(logp, targets, in_len, tgt_len)
            total_loss += float(loss.item())

            pred_texts = greedy_decode(logits, in_len, sp)
            ref_texts = batch["text"]
            for p, r in zip(pred_texts, ref_texts):
                total_cer += cer(p, r)
                total_wer += wer(p, r)
                n += 1

    if n == 0:
        return {"loss": total_loss / max(1, len(loader)), "cer": 1.0, "wer": 1.0}

    return {
        "loss": total_loss / max(1, len(loader)),
        "cer": total_cer / n,
        "wer": total_wer / n
    }

from tqdm import tqdm

def filter_ctc_impossible(samples: List[Sample], sp: spm.SentencePieceProcessor,
                          max_frames: int, ratio: float = 0.90) -> List[Sample]:
    """
    Drop samples where token_len is too large vs effective frames (after max_frames subsampling).
    CTC needs roughly L <= T (safer: L <= ratio*T).
    """
    kept = []
    dropped = 0
    bad_examples = []  # keep a few for debug

    pbar = tqdm(samples, desc="[CTC_FILTER] scanning", total=len(samples))
    for i, s in enumerate(pbar, start=1):
        try:
            tok_len = len(sp.encode(s.text, out_type=int))

            with np.load(s.npz_path, allow_pickle=False) as data:
                key = "frames" if "frames" in data.files else data.files[0]
                T = int(data[key].shape[0])
            T_eff = min(T, max_frames) if max_frames is not None else T

            ok = (tok_len <= int(ratio * T_eff)) and (tok_len > 0) and (T_eff > 0)
            if ok:
                kept.append(s)
            else:
                dropped += 1
                if len(bad_examples) < 3:
                    bad_examples.append((s.utt_id, tok_len, T_eff, s.npz_path))
        except Exception as e:
            dropped += 1
            if len(bad_examples) < 3:
                bad_examples.append((s.utt_id, "ERR", "ERR", s.npz_path))

        # update live stats (lightweight)
        if i % 50 == 0 or i == len(samples):
            pbar.set_postfix(kept=len(kept), dropped=dropped)

    print(f"[CTC_FILTER] kept={len(kept)} dropped={dropped} (rule: tok_len <= {ratio} * T_eff)")
    if bad_examples:
        print("[CTC_FILTER] examples dropped (utt_id, tok_len, T_eff, path):")
        for ex in bad_examples:
            print("  ", ex)
    return kept



# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root dataset folder, e.g. /content/drive/MyDrive/Licenta/ro_vsr")
    ap.add_argument("--meta", default=None, help="metadata csv path (default: <root>/metadata_train_final_segments.csv)")
    ap.add_argument("--npz_dir", default=None, help="mouth_npz folder (default: <root>/mouth_npz)")
    ap.add_argument("--spm", default=None, help="SentencePiece model path (default: <root>/spm/ro_vsr.model)")

    ap.add_argument("--run_dir", default=None, help="Where to save logs+ckpts (default: <root>/runs/ctc_run)")
    ap.add_argument("--epochs", type=int, default=25, help="TOTAL epochs (if resuming, set to desired total)")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--max_frames", type=int, default=200)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true", help="Use mixed precision on GPU")
    ap.add_argument("--resume", default=None, help="Path to checkpoint .pt to resume (usually <run_dir>/last.pt)")
    ap.add_argument("--init_ckpt", default=None, help="Load ONLY model weights from checkpoint (no optimizer/scheduler).")
    ap.add_argument("--reset_optim", action="store_true", help="When resuming, ignore optimizer/scheduler/scaler states.")
    ap.add_argument("--keep_val_fixed", action="store_true", help="Keep the same val split across future runs (recommended)")
    ap.add_argument("--init_frontend", default=None, help="Path to pretrained frontend .pt (state_dict)")
    ap.add_argument("--ctc_ratio", type=float, default=0.95, help="keep if tok_len <= ratio*T_eff")
    ap.add_argument("--freeze_frontend_epochs", type=int, default=0,
                help="Freeze the visual frontend for first N epochs (fine-tune stability).")
    # inference
    ap.add_argument("--mode", choices=["train","infer_npz"], default="train")
    ap.add_argument("--ckpt", default=None, help="Checkpoint for inference")
    ap.add_argument("--npz", default=None, help="Single npz path for inference")

    args = ap.parse_args()
    set_seed(args.seed)

    root = args.root
    meta = args.meta or os.path.join(root, "metadata_train_final_segments.csv")
    npz_dir = args.npz_dir or os.path.join(root, "mouth_npz")
    spm_path = args.spm or os.path.join(root, "spm", "ro_vsr.model")
    run_dir = args.run_dir or os.path.join(root, "runs", "ctc_run")

    if args.mode == "infer_npz":
        if not args.ckpt or not args.npz:
            raise ValueError("--mode infer_npz requires --ckpt and --npz")
        if not os.path.exists(spm_path):
            raise FileNotFoundError(f"Missing spm model: {spm_path}")

        sp = spm.SentencePieceProcessor()
        sp.load(spm_path)
        vocab_with_blank = sp.get_piece_size() + 1

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = LipReadingCTCConformer(
            vocab_size_with_blank=vocab_with_blank,
            d_model=256,
            n_layers=6,
            d_ff=1024,
            n_heads=4,
            conv_kernel=31,
            dropout=0.1,
            blank_bias=-2.0
        ).to(device)
        load_ckpt(args.ckpt, model, optim=None, sched=None, scaler=None)

        data = np.load(args.npz, allow_pickle=False)
        frames = data["frames"].astype(np.float32) / 255.0
        frames = np.expand_dims(frames, axis=1)  # (T,1,H,W)
        video = torch.from_numpy(frames).unsqueeze(0).to(device)
        in_len = torch.tensor([video.shape[1]], device=device, dtype=torch.long)

        model.eval()
        with torch.no_grad():
            logits = model(video)
            pred = greedy_decode(logits, in_len, sp)[0]
        print("PRED:", pred)
        return

    # TRAIN
    for p in [meta, npz_dir, spm_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing: {p}")

    sp = spm.SentencePieceProcessor()
    sp.load(spm_path)

    samples = build_samples(meta, npz_dir)
    samples = filter_ctc_impossible(samples, sp, max_frames=args.max_frames, ratio=args.ctc_ratio)

    if len(samples) < 1:
      raise RuntimeError("No usable samples. Check metadata/npz/text.")


    train_s, val_s = make_or_load_split(
        samples=samples,
        run_dir=run_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        keep_val_fixed=args.keep_val_fixed
    )

    print(f"[SPLIT] train={len(train_s)} val={len(val_s)}")
    do_eval = len(val_s) > 0


    train_ds = RomanianVSRDataset(train_s, sp, max_video_frames=args.max_frames)
    val_ds   = RomanianVSRDataset(val_s, sp, max_video_frames=args.max_frames)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=collate_fn, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=max(1, args.batch), shuffle=False,
        num_workers=max(0, args.num_workers // 2),
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=collate_fn, drop_last=False
    )

    vocab_with_blank = sp.get_piece_size() + 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[DEVICE]", device)

    model = LipReadingCTCConformer(
        vocab_size_with_blank=vocab_with_blank,
        d_model=256,
        n_layers=6,
        d_ff=1024,
        n_heads=4,
        conv_kernel=31,
        dropout=0.1,
        blank_bias=-2.0
    ).to(device)
    if args.init_frontend:
        sd = torch.load(args.init_frontend, map_location="cpu")
        model.frontend.load_state_dict(sd, strict=True)
        print("[INIT] loaded pretrained frontend from", args.init_frontend)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    scaler = torch.cuda.amp.GradScaler(enabled=(args.fp16 and device == "cuda"))

    # scheduler: cosine over TOTAL epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(1, args.epochs))

    start_epoch = 0
    best_cer = 1e9

    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    # if args.resume:
    #     ck = load_ckpt(args.resume, model, optim, sched, scaler)
    #     start_epoch = int(ck.get("epoch", 0)) + 1
    #     best_cer = float(ck.get("best_metric", best_cer))
    #     print(f"[RESUME] start_epoch={start_epoch} best_cer={best_cer:.4f}")
    # (A) init_ckpt: încarcă DOAR greutățile modelului (start_epoch rămâne 0)
    if args.init_ckpt:
        ck = torch.load(args.init_ckpt, map_location="cpu")
        model.load_state_dict(ck["model"], strict=True)
        # păstrăm best_cer din checkpoint ca referință, dar continuăm fine-tune de la epoca 0
        best_cer = float(ck.get("best_metric", best_cer))
        print(f"[INIT_CKPT] loaded model weights from {args.init_ckpt} (best_cer={best_cer:.4f})")
        start_epoch = 0

    # (B) resume: continuă trainingul (dar poți reseta optimizerul)
    elif args.resume:
        if args.reset_optim:
            ck = torch.load(args.resume, map_location="cpu")
            model.load_state_dict(ck["model"], strict=True)
            best_cer = float(ck.get("best_metric", best_cer))
            # IMPORTANT: nu încărcăm optim/sched/scaler
            start_epoch = int(ck.get("epoch", 0)) + 1
            print(f"[RESUME_MODEL_ONLY] loaded model only from {args.resume} (start_epoch={start_epoch}, best_cer={best_cer:.4f})")
        else:
            ck = load_ckpt(args.resume, model, optim, sched, scaler)
            start_epoch = int(ck.get("epoch", 0)) + 1
            best_cer = float(ck.get("best_metric", best_cer))
            print(f"[RESUME_FULL] start_epoch={start_epoch} best_cer={best_cer:.4f}")

    history_path = os.path.join(run_dir, "history.csv")
    history = []
    if os.path.exists(history_path) and start_epoch > 0:
        try:
            history = pd.read_csv(history_path).to_dict("records")
        except Exception:
            history = []

    for epoch in range(start_epoch, args.epochs):
        freeze = (epoch - start_epoch) < args.freeze_frontend_epochs
        for p in model.frontend.parameters():
            p.requires_grad = not freeze
        if freeze:
            print(f"[FREEZE] frontend frozen (epoch {epoch})")
        model.train()
        pbar = tqdm(train_loader, desc=f"train e{epoch+1}/{args.epochs}", leave=False)
        running = 0.0

        for batch in pbar:
            video = batch["video"].to(device)
            in_len = batch["input_lengths"].to(device)
            targets = batch["targets"].to(device)
            tgt_len = batch["target_lengths"].to(device)

            optim.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(args.fp16 and device == "cuda")):
                logits = model(video)
                logp = logits.log_softmax(dim=-1).permute(1,0,2)
                loss = ctc(logp, targets, in_len, tgt_len)
                # debug: cât de mult prezice blank (0)
                if (epoch % 10 == 0) and (pbar.n == 0):
                    with torch.no_grad():
                        pred = logits.argmax(dim=-1)  # (B,T)
                        blank_frac = (pred == 0).float().mean().item()
                    print(f"[DBG] blank_frac={blank_frac:.3f} T={int(in_len[0])} L={int(tgt_len[0])} ref='{batch['text'][0][:80]}'")


            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optim)
            scaler.update()

            running += float(loss.item())
            pbar.set_postfix(loss=f"{running/max(1,(pbar.n+1)):.4f}", lr=f"{optim.param_groups[0]['lr']:.2e}")

        sched.step()

        # eval
        tr_loss = running / max(1, len(train_loader))

        if do_eval:
            val_metrics = run_eval(model, val_loader, device, sp)
            val_loss = val_metrics["loss"]
            val_cer  = val_metrics["cer"]
            val_wer  = val_metrics["wer"]
        else:
            val_loss = float("nan")
            val_cer  = float("nan")
            val_wer  = float("nan")

        row = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": val_loss,
            "val_cer": val_cer,
            "val_wer": val_wer,
            "lr": optim.param_groups[0]["lr"]
        }
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False, encoding="utf-8")

        # save checkpoints
        last_path = os.path.join(run_dir, "last.pt")
        save_ckpt(last_path, model, optim, sched, scaler, epoch, best_cer, vars(args))

        if do_eval and (val_cer < best_cer):
            best_cer = val_cer
            best_path = os.path.join(run_dir, "best.pt")
            save_ckpt(best_path, model, optim, sched, scaler, epoch, best_cer, vars(args))

        print(f"[EPOCH {epoch}] train_loss={tr_loss:.4f} val_loss={val_loss} "
            f"val_CER={val_cer} val_WER={val_wer} best_CER={best_cer:.4f}")


    print("[DONE] run_dir:", run_dir)
    print("best checkpoint:", os.path.join(run_dir, "best.pt"))
    print("last checkpoint:", os.path.join(run_dir, "last.pt"))


if __name__ == "__main__":
    main()
