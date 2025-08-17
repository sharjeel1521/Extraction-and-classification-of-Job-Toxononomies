#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-GPU MNRL training for intfloat/e5-base-v2
  · positive-only pairs with E5 prefixes
  · bf16 / fp16 / fp32 via 🤗 Accelerate
  · gradient accumulation
  · checkpoints every N steps + every epoch
  · OPTIONAL light dev recall@K probe (CPU/Faiss) – disabled when --eval_every 0
"""
from __future__ import annotations
import argparse, json, math, os, random, time, gc
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm

from accelerate import Accelerator
from sentence_transformers import SentenceTransformer, InputExample, losses
from transformers import get_cosine_schedule_with_warmup

# ────────────── defaults ────────────── #
SAVE_EVERY_DEFAULT = 10_000
EVAL_EVERY_DEFAULT = 10_000          # 0 ⇒ never evaluate
DEV_K_DEFAULT      = 20
DEV_LIMIT_DEFAULT  = 20_000          # 0 ⇒ use all positive dev pairs

# ────────────── helpers ────────────── #
def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def unwrap(model, accel: Accelerator | None = None):
    return accel.unwrap_model(model) if accel and hasattr(accel, "unwrap_model") \
           else model.module if hasattr(model, "module") else model

# ────────────── dataset ────────────── #
class PicklePositiveDataset(Dataset):
    """pkl → positive pairs with E5 prefixes"""
    def __init__(self, pkl: str):
        import pickle; raw: List[InputExample] = pickle.load(open(pkl, "rb"))
        self.data = [
            InputExample(texts=[f"query: {ex.texts[0].strip()}",
                                f"passage: {ex.texts[1].strip()}"],
                         label=1.0)
            for ex in raw if float(ex.label) == 1.0
        ]
    def __len__(self):          return len(self.data)
    def __getitem__(self, i):   return self.data[i]

# ────────────── optional light probe ────────────── #
@torch.inference_mode()
def light_recall(model: SentenceTransformer,
                 dev_pickle: str,
                 k: int,
                 limit: int,
                 bs: int = 256) -> float:
    if k == 0 or limit == 0:
        return 0.0
    import pickle, random, faiss  # faiss CPU build is fine
    raw = pickle.load(open(dev_pickle, "rb"))
    pos = [(ex.texts[0].strip(), ex.texts[1].strip())
           for ex in raw if float(ex.label) == 1.0]
    if not pos:
        return 0.0
    if 0 < limit < len(pos):
        pos = random.sample(pos, limit)
    qs, ps = zip(*pos)
    qs = [f"query: {q}"   for q in qs]
    ps = [f"passage: {p}" for p in ps]

    q = np.asarray(model.encode(qs, batch_size=bs,
                                convert_to_tensor=False,
                                normalize_embeddings=True), np.float32)
    p = np.asarray(model.encode(ps, batch_size=bs,
                                convert_to_tensor=False,
                                normalize_embeddings=True), np.float32)
    dim = p.shape[1]
    index = faiss.IndexFlatIP(dim); index.add(p)
    _, I = index.search(q, k)
    return float((I == np.arange(len(q))[:, None]).any(1).mean())

# ────────────── CLI ────────────── #
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_pickle", required=True)
    ap.add_argument("--dev_pickle",   required=True)
    ap.add_argument("--output_dir",   required=True)
    # training hyper-params
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch",  type=int, default=32)
    ap.add_argument("--accum",  type=int, default=4)
    ap.add_argument("--lr",     type=float, default=5e-6)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    # model
    ap.add_argument("--model_name", default="intfloat/e5-base-v2")
    ap.add_argument("--max_seq_length", type=int, default=128)
    ap.add_argument("--mixed_precision", choices=["no","fp16","bf16"], default="no")
    ap.add_argument("--seed", type=int, default=42)
    # checkpoints / eval
    ap.add_argument("--save_every", type=int, default=SAVE_EVERY_DEFAULT)
    ap.add_argument("--eval_every", type=int, default=EVAL_EVERY_DEFAULT)
    ap.add_argument("--dev_k",      type=int, default=DEV_K_DEFAULT)
    ap.add_argument("--dev_limit",  type=int, default=DEV_LIMIT_DEFAULT)
    return ap.parse_args()

# ────────────── main ────────────── #
def main():
    args = get_args(); set_seed(args.seed)

    out = Path(args.output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    (out/"args.json").write_text(json.dumps(vars(args), indent=2))

    acc = Accelerator(mixed_precision=args.mixed_precision)
    if acc.is_main_process:
        print(f"world_size={acc.num_processes}  device={acc.device}  "
              f"precision={acc.mixed_precision}", flush=True)

    base = SentenceTransformer(args.model_name, device=acc.device)
    base.max_seq_length = args.max_seq_length
    loss_fn = losses.MultipleNegativesRankingLoss(base)

    dl = DataLoader(PicklePositiveDataset(args.train_pickle),
                    batch_size=args.batch, shuffle=True, drop_last=True,
                    collate_fn=base.smart_batching_collate,
                    num_workers=4, pin_memory=True, persistent_workers=True)

    opt = AdamW(base.parameters(),
                lr=args.lr * math.sqrt(acc.num_processes))
    steps_tot = (len(dl)//args.accum) * args.epochs
    sch = get_cosine_schedule_with_warmup(
        opt, int(steps_tot*args.warmup_ratio), steps_tot)

    model,opt,sch,dl = acc.prepare(base,opt,sch,dl)

    gstep,best=0,-1.; t0=time.time()
    for ep in range(1, args.epochs+1):
        model.train(); run=0.
        pbar=tqdm(dl, disable=not acc.is_main_process, desc=f"epoch {ep}")
        for step,b in enumerate(pbar,1):
            feats,lbl=b
            feats=[{k:v.to(acc.device) for k,v in f.items()} for f in feats]
            lbl=lbl.to(acc.device)
            with acc.autocast(): loss=loss_fn(feats,lbl)/args.accum
            acc.backward(loss); run+=loss.item()
            if step%args.accum==0:
                opt.step(); sch.step(); opt.zero_grad(); gstep+=1
                if acc.is_main_process:
                    pbar.set_postfix(loss=f"{run:.4f}",
                                     lr=f"{sch.get_last_lr()[0]:.2e}",
                                     step=gstep); run=0.
                # checkpoint
                if gstep%args.save_every==0 and acc.is_main_process:
                    tag=out/f"step_{gstep}"; tag.mkdir(parents=True,exist_ok=True)
                    unwrap(model,acc).save(str(tag/"model"))
                    torch.save({"step":gstep}, tag/"state.pt")
                    print(f"[✓] checkpoint → {tag}", flush=True)
                # optional probe
                if args.eval_every>0 and gstep%args.eval_every==0 and acc.is_main_process:
                    print(f"[probe] computing R@{args.dev_k} …", flush=True)
                    rec=light_recall(unwrap(model,acc), args.dev_pickle,
                                     args.dev_k, args.dev_limit)
                    best=max(best,rec)
                    print(f"[probe] R@{args.dev_k}={rec:.4f} (best {best:.4f})",flush=True)
        # epoch checkpoint
        if acc.is_main_process:
            tag=out/f"epoch_{ep}"; tag.mkdir(parents=True,exist_ok=True)
            unwrap(model,acc).save(str(tag/"model"))
            torch.save({"epoch":ep}, tag/"state.pt")
            print(f"[✓] checkpoint → {tag}  (best R {best:.4f})")
        gc.collect(); torch.cuda.empty_cache()

    acc.wait_for_everyone()
    if acc.is_main_process:
        unwrap(model,acc).save(str(out/"final_model"))
        print(f"[✓] done in {(time.time()-t0)/60:.1f} min  "
              f"(best light R {best:.4f})", flush=True)

if __name__ == "__main__":
    main()
