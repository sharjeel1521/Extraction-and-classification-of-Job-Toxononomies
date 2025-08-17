#!/usr/bin/env python3
# ------------------------------------------------------------------
# Build Cross-Encoder training/dev datasets (full + small)
# ------------------------------------------------------------------

#HOW TO RUN
#python Part1_Cross_Training.py \
#  --jobs_pq   Cross_Encoder_4k_attr_jobs_to_label.pq \
#  --pairs_csv Cross_Encoder_o3m.csv \
#  --defs_tsv  defs.tsv



from __future__ import annotations
import argparse, logging, pickle, random
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm.auto import tqdm

logging.basicConfig(
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO)
log = logging.getLogger(__name__)


# ────────────────────────────── CLI ──────────────────────────────
def get_args():
    p = argparse.ArgumentParser(
        description="Build train.pkl/dev.pkl and train_small.pkl/dev_small.pkl")
    p.add_argument("--jobs_pq",    required=True,
                   help="4k_attr_jobs_to_label.pq (job_id, title, description …)")
    p.add_argument("--pairs_csv",  required=True,
                   help="o3m.csv (id, suid, label)")
    p.add_argument("--defs_tsv",   required=True,
                   help="defs.tsv (suid, label, definition)")
    p.add_argument("--out_dir",    default="Part1_Cross_Training_Data",
                   help="target folder for *.pkl files")

    p.add_argument("--neg_per_pos", type=int, default=3,
                   help="keep at most N negatives per positive for each job_id")
    p.add_argument("--dev_ratio",   type=float, default=0.20,
                   help="fraction of job_ids reserved for dev")

    # “small” subset sizes
    p.add_argument("--small_train", type=int, default=5_000)
    p.add_argument("--small_dev",   type=int, default=500)

    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ────────────────────────── helpers ──────────────────────────────
def seed_everything(seed: int):
    import numpy as np, random
    random.seed(seed);  np.random.seed(seed)


def save_pickle(obj, path: Path):
    path.write_bytes(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
    log.info("📦  wrote %s  (%s examples)", path.name, f"{len(obj):,}")


# ─────────────────────────── main ────────────────────────────────
def main():
    args = get_args()
    seed_everything(args.seed)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) jobs  -----------------------------------------------------------------
    jobs = pd.read_parquet(args.jobs_pq,
                           columns=["job_id", "title", "description"])
    jobs["text"] = jobs["title"].fillna("") + ": " + jobs["description"].fillna("")
    id2text: Dict[int, str] = dict(zip(jobs.job_id, jobs.text))
    log.info("✓ loaded %s job texts", f"{len(id2text):,}")

    # 2) attribute definitions -------------------------------------------------
    defs = pd.read_csv(args.defs_tsv, sep="\t")
    defs["text"] = defs["label"].fillna("") + ": " + defs["definition"].fillna("")
    suid2text: Dict[str, str] = dict(zip(defs.suid, defs.text))
    log.info("✓ loaded %s attribute definitions", f"{len(suid2text):,}")

    # 3) supervision pairs -----------------------------------------------------
    pairs = pd.read_csv(args.pairs_csv, usecols=["id", "suid", "label"])
    before = len(pairs)
    pairs = pairs[pairs.id.isin(id2text) & pairs.suid.isin(suid2text)]
    log.info("✓ loaded %s supervision rows (filtered from %s)",
             f"{len(pairs):,}", f"{before:,}")

    # 4) split by *job*  -------------------------------------------------------
    job_ids = pairs.id.unique().tolist()
    random.shuffle(job_ids)
    cut = int(len(job_ids)*(1 - args.dev_ratio))
    train_ids, dev_ids = set(job_ids[:cut]), set(job_ids[cut:])

    def sample_rows(g):
        pos = g[g.label == 1]
        neg = g[g.label == 0]
        if args.neg_per_pos and len(neg) > len(pos)*args.neg_per_pos:
            neg = neg.sample(len(pos)*args.neg_per_pos, random_state=args.seed)
        return pd.concat([pos, neg])

    def build_bucket(target_ids: set[int]) -> List[Dict]:
        bucket: List[Dict] = []
        for jid, grp in tqdm(pairs.groupby("id"), total=len(job_ids)):
            if jid not in target_ids:        # fast skip
                continue
            jt = id2text[jid]
            for _, row in sample_rows(grp).iterrows():
                bucket.append({
                    "sentence1": jt,
                    "sentence2": suid2text[row.suid],
                    "label": float(row.label)
                })
        random.shuffle(bucket)
        return bucket

    log.info("→ creating full TRAIN …")
    train = build_bucket(train_ids)
    log.info("→ creating full DEV …")
    dev   = build_bucket(dev_ids)

    # 5) “small” subsets -------------------------------------------------------
    small_train = random.sample(train, min(args.small_train, len(train)))
    small_dev   = random.sample(dev,   min(args.small_dev,   len(dev)))

    # 6) save all four pickles -------------------------------------------------
    save_pickle(train,        out_dir / "train.pkl")
    save_pickle(dev,          out_dir / "dev.pkl")
    save_pickle(small_train,  out_dir / "train_small.pkl")
    save_pickle(small_dev,    out_dir / "dev_small.pkl")

    log.info("🏁  done – data cached in %s", out_dir)


if __name__ == "__main__":
    main()
