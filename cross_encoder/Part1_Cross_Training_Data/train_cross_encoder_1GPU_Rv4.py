#!/usr/bin/env python3
# train_cross_encoder_1GPU_noeval.py
# ------------------------------------------------------------
# Fine‑tune a cross‑encoder on a single GPU – *no* dev eval
# ------------------------------------------------------------
from __future__ import annotations
import argparse, json, logging, os, pickle, random, time
from pathlib import Path
from typing import Dict, List

# silence WandB / TensorFlow
os.environ["WANDB_DISABLED"]      = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sentence_transformers import CrossEncoder, LoggingHandler
from sentence_transformers.readers import InputExample
from tqdm.auto import tqdm

# ------------------------- CLI -------------------------
def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train_pkl", required=True,
                   help="Part1_Cross_Training_Data/output/train.pkl")
    p.add_argument("--output_dir", default="Part3_Cross_May1")
    p.add_argument("--epochs",      type=int,   default=4)
    p.add_argument("--batch",       type=int,   default=64)
    p.add_argument("--lr",          type=float, default=2e-5)
    p.add_argument("--save_steps",  type=int,   default=10_000,
                   help="save HF checkpoint every N optimizer steps")
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()

# ----------------------- helpers -----------------------
def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_examples(pkl: str) -> List[InputExample]:
    rows: List[Dict] = pickle.load(open(pkl, "rb"))
    return [InputExample(texts=[r["sentence1"], r["sentence2"]],
                         label=float(r["label"]))
            for r in rows]

# ------------------------- main ------------------------
def main() -> None:
    args = get_args()
    seed_everything(args.seed)

    logging.basicConfig(format="%(asctime)s  %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    log = logging.getLogger(__name__)

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    # ---------- data ----------
    log.info("📂 Loading training pickle …")
    train_ex = load_examples(args.train_pkl)
    log.info("✓ train: %s samples", f"{len(train_ex):,}")

    # ---------- model ----------
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2",
                         num_labels=1, max_length=256,
                         default_activation_function=None,
                         device="cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = model.tokenizer
    device    = model.device

    # bump batch‑size if VRAM allows
    if torch.cuda.is_available():
        free_gib = (torch.cuda.get_device_properties(0).total_memory
                    - torch.cuda.memory_allocated()) / 1024**3
        if free_gib > 15 and args.batch < 128:
            log.info("🚀 %.1f GiB free – raising batch to 128", free_gib)
            args.batch = 128

    # collate → tensors stay on CPU, moved to GPU in the loop
    def collate(batch):
        s1, s2, lab = zip(*[(b.texts[0], b.texts[1], b.label) for b in batch])
        enc = tokenizer(list(s1), list(s2),
                        padding="max_length", truncation=True,
                        max_length=256, return_tensors="pt")
        enc["labels"] = torch.tensor(lab, dtype=torch.float)
        return enc

    dl_train = DataLoader(train_ex,
                          batch_size=args.batch,
                          shuffle=True,
                          num_workers=4,
                          pin_memory=True,
                          collate_fn=collate)

    opt    = AdamW(model.model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    total_steps  = len(dl_train) * args.epochs
    global_step  = 0
    run_loss     = 0.0
    t0           = time.time()

    # -------- resume support --------
    epoch_dirs = sorted(d for d in out.iterdir() if d.name.startswith("epoch"))
    start_epoch = len(epoch_dirs) + 1
    if start_epoch > 1:
        last = epoch_dirs[-1]
        log.info("↪️  Resuming from %s", last)
        resume_ckpt = Path("Part3_Cross_May1/step00110000")
        model.model.load_state_dict(torch.load(resume_ckpt / "model.safetensors"))
        """
        from safetensors.torch import load_file

        model.model.load_state_dict(load_file(str(resume_ckpt / "model.safetensors")))
        """

    # -------------- training loop --------------
    for epoch in range(start_epoch, args.epochs + 1):
        model.model.train()
        bar = tqdm(dl_train, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")
        for batch in bar:
            global_step += 1
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model.model(**{k: v for k, v in batch.items()
                                        if k != "labels"}).logits.squeeze(-1)
                loss   = torch.nn.functional.binary_cross_entropy_with_logits(
                           logits, batch["labels"])

            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad()

            run_loss += loss.item()
            if global_step % 100 == 0:
                avg = run_loss / 100
                run_loss = 0.0
                elapsed = time.time() - t0
                speed   = global_step / elapsed
                eta     = (total_steps - global_step) / speed
                bar.set_postfix(loss=f"{avg:.4f}",
                                elapsed=time.strftime('%H:%M:%S', time.gmtime(elapsed)),
                                ETA=time.strftime('%H:%M:%S', time.gmtime(eta)),
                                steps_s=f"{speed:.1f}")

            if global_step % args.save_steps == 0:
                ck = out / f"step{global_step:08d}"
                ck.mkdir(exist_ok=True)
                model.save(ck)
                log.info("💾 Saved checkpoint → %s", ck)

        # save at end of each epoch
        ep_dir = out / f"epoch{epoch}"
        ep_dir.mkdir(exist_ok=True)
        model.save(ep_dir)
        log.info("💾 Saved epoch checkpoint → %s", ep_dir)

    # final weights
    torch.save(model.model.state_dict(), out / "final.pt")
    log.info("🏁 Training finished – final weights at %s", out / 'final.pt')


if __name__ == "__main__":
    main()
