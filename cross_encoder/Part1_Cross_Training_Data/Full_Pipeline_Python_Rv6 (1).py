#!/usr/bin/env python3
"""
Precision-focused pipeline that implements:
1. Targeted candidate selection from bi-encoder
2. Multi-pass filtering to boost precision while maintaining recall
3. Job-specific precision optimizations
"""
import argparse, os, json, random
from pathlib import Path
from typing import Dict, List, Tuple, Set
import numpy as np
import pandas as pd
import torch
import faiss
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder
import matplotlib
matplotlib.use('Agg')  # For headless servers
import matplotlib.pyplot as plt

# ─── FIXED PATHS ─────────────────────────────────────────────────
BASE = "/home/jovyan/butterfly/src/notebooks/BI_ENCODER_TRAINING"
BI_MODEL = f"{BASE}/ToSubmit_Part2_O1_MNRL_Prefix5_Multi_GPU_Final_Rv1/step_10000/model"
SENTS = f"{BASE}/golden_dataset_sentences.csv"
CE_MODEL = "/home/jovyan/butterfly/src/notebooks/Part2_Cross_May5/epoch4"
#""epoch4"
DEFS = "/home/jovyan/butterfly/src/notebooks/defs.tsv"
GROUND = "/home/jovyan/butterfly/src/notebooks/golden_dataset_labels.csv"

# ─── HELPERS ───────────────────────────────────────────────────────
def normalize(txt: str) -> str:
    return txt.lower().strip() if isinstance(txt, str) else ""

def load_ground(path: str) -> Dict[str, List[str]]:
    df = pd.read_csv(path, dtype={"job_id": str})
    df["display_name"] = df["display_name"].map(normalize)
    pos = df[df["label"] == "POSITIVE"]
    return pos.groupby("job_id")["display_name"].unique().apply(list).to_dict()

def metrics(pred: Dict[str, set], gt: Dict[str, List[str]]) -> Tuple[float, float, float]:
    p = r = f1 = 0.0
    job_metrics = []
    
    for job, true_lbls in gt.items():
        pr = pred.get(job, set())
        tp = len(pr & set(true_lbls))
        fp = len(pr - set(true_lbls))
        fn = len(set(true_lbls) - pr)
        prec = tp / (tp + fp) if tp + fp else 0.0
        reca = tp / (tp + fn) if tp + fn else 0.0
        f1_i = 2 * prec * reca / (prec + reca) if prec + reca else 0.0
        p += prec
        r += reca
        f1 += f1_i
        job_metrics.append((job, prec, reca, f1_i, tp, fp, fn))
    
    n = len(gt)
    return p / n, r / n, f1 / n, job_metrics

def batched_encode(texts: List[str], model, bs: int = 128) -> np.ndarray:
    out = []
    for i in tqdm(range(0, len(texts), bs), desc="encode", unit="batch"):
        out.append(
            model.encode(
                texts[i : i + bs],
                batch_size=len(texts[i : i + bs]),
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        )
    return np.vstack(out).astype("float32")

def plot_histogram(scores, title, filepath, threshold=None):
    """Create a simple histogram of scores"""
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=50, alpha=0.7)
    plt.title(title)
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    
    if threshold is not None:
        plt.axvline(x=threshold, color='r', linestyle='--', 
                    label=f'Threshold = {threshold:.3f}')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    print(f"Saved histogram to {filepath}")

# ─── MAIN ─────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-sample", type=int, help="randomly sample N jobs")
    p.add_argument("--sample", help="comma‑separated job_ids to keep")
    p.add_argument("--K", type=int, default=2000, help="FAISS pool size")
    p.add_argument("--bi-topn", type=int, default=1600, help="Bi-encoder top-N")
    p.add_argument("--max-labels", type=int, default=None, 
                   help="Maximum labels per job (None=unlimited)")
    p.add_argument("--thr", type=float, default=0.15, 
                   help="Base threshold for cross-encoder")
    p.add_argument("--recall-target", type=float, default=0.9, 
                   help="Target recall to maintain")
    p.add_argument("--precision-mode", choices=['strict', 'balanced', 'recall'], 
                   default='balanced', help="Precision optimization mode")
    p.add_argument("--out", default="precision_metrics.json")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # 1) load & sample data ------------------------------------------------
    ground = load_ground(GROUND)
    sents = pd.read_csv(SENTS, dtype={"Job_ID": str})
    sents.rename(columns={"Job_ID": "job_id"}, inplace=True)

    if args.sample:
        keep = set(args.sample.split(","))
    elif args.n_sample:
        keep = set(random.sample(list(ground.keys()), args.n_sample))
    else:
        keep = set(ground.keys())

    sents = sents[sents.job_id.isin(keep)]
    ground = {j: ground[j] for j in keep if j in ground}
    print(f"Jobs: {len(ground)}  Sentences: {len(sents)}")

    queries = [f"query: {q}" for q in sents["Sentences"]]
    defs_df = pd.read_csv(DEFS, sep="\t").fillna("")
    passages = [f"passage: {r.label}. {r.definition}" for r in defs_df.itertuples()]
    
    # Map label indices to actual labels
    idx_to_label = {i: normalize(row.label) for i, row in enumerate(defs_df.itertuples())}

    # 2) load models --------------------------------------------------------
    bi = SentenceTransformer(BI_MODEL, device=device)
    bi.max_seq_length = 256
    ce = CrossEncoder(CE_MODEL, device=device)
    ce.max_length = 256

    # 3) bi-encoder + FAISS ------------------------------------------------
    emb_pass = batched_encode(passages, bi, 128)
    idx = faiss.IndexFlatIP(emb_pass.shape[1])
    idx.add(emb_pass)

    emb_q = batched_encode(queries, bi, 128)
    D, I = idx.search(emb_q, args.K)
    
    # 4) Implement a ranking-based approach to get high recall -------------
    print(f"\nSTAGE 1: Using bi-encoder with ranking approach")
    
    # Group sentences by job
    job_sentences = {}
    for i, row in enumerate(sents.itertuples()):
        job_sentences.setdefault(row.job_id, []).append(i)
    
    # Create job-level predictions using bi-encoder ranking
    job_predictions_bi = {}
    
    for job_id, sentence_indices in job_sentences.items():
        job_cands = set()
        
        for sent_idx in sentence_indices:
            # For each sentence, get top-N candidates
            for j in range(min(args.bi_topn, len(I[sent_idx]))):
                idx_pos = I[sent_idx][j]
                if idx_pos >= 0:
                    label = idx_to_label.get(idx_pos)
                    if label:
                        job_cands.add(label)
        
        job_predictions_bi[job_id] = job_cands
    
    # Evaluate bi-encoder results
    bi_precision, bi_recall, bi_f1, _ = metrics(job_predictions_bi, ground)
    print(f"Bi-encoder Results (top {args.bi_topn}):")
    print(f"  Precision: {bi_precision:.4f}")
    print(f"  Recall: {bi_recall:.4f}")
    print(f"  F1 Score: {bi_f1:.4f}")
    
    if bi_recall < args.recall_target:
        print(f"Warning: Bi-encoder recall {bi_recall:.4f} is below target {args.recall_target:.4f}")
        print("Consider increasing --bi-topn to get higher recall")
    
    # 5) Get ground truth info for cross-encoder optimization --------------
    # For each job, identify which labels are relevant and non-relevant
    job_relevant_labels = {}
    job_non_relevant_labels = {}
    
    for job_id, bi_labels in job_predictions_bi.items():
        gt_labels = set(normalize(l) for l in ground.get(job_id, []))
        relevant = bi_labels & gt_labels
        non_relevant = bi_labels - gt_labels
        
        job_relevant_labels[job_id] = relevant
        job_non_relevant_labels[job_id] = non_relevant
    
    # 6) Cross-encoder scoring with smart batching ------------------------
    print(f"\nSTAGE 2: Cross-encoder scoring with optimized processing")
    
    # For each job, prepare query-passage pairs with removed prefixes
    job_label_scores = {}
    
    for job_id, sentence_indices in tqdm(job_sentences.items(), desc="Processing jobs"):
        job_candidates = job_predictions_bi[job_id]
        
        # Find best matching passage index for each label
        label_passage_idx = {}
        for i, row in enumerate(defs_df.itertuples()):
            label = normalize(row.label)
            if label in job_candidates:
                label_passage_idx[label] = i
        
        # For each sentence in this job, create pairs with candidates
        all_pairs = []
        all_labels = []
        
        for sent_idx in sentence_indices:
            query = sents.Sentences.iloc[sent_idx]
            
            # Find overlap between this sentence's top results and job candidates
            sent_labels = set()
            for j in range(min(args.bi_topn, len(I[sent_idx]))):
                idx_pos = I[sent_idx][j]
                if idx_pos >= 0:
                    label = idx_to_label.get(idx_pos)
                    if label and label in job_candidates:
                        sent_labels.add(label)
            
            # Create pairs for this sentence
            for label in sent_labels:
                passage_idx = label_passage_idx.get(label)
                if passage_idx is not None:
                    # Create pair with REMOVED PREFIXES
                    pair = (
                        query,  # No prefix
                        passages[passage_idx].replace("passage: ", "")  # Remove prefix
                    )
                    all_pairs.append(pair)
                    all_labels.append(label)
        
        # Score all pairs in batches
        all_scores = []
        batch_size = 64
        
        for i in range(0, len(all_pairs), batch_size):
            batch = all_pairs[i:i+batch_size]
            with torch.inference_mode():
                scores = ce.predict(batch, convert_to_numpy=True)
            all_scores.extend(scores)
        
        # Convert to probabilities
        all_probs = 1 / (1 + np.exp(-np.array(all_scores)))
        
        # Group scores by label (take max score for each label)
        label_scores = {}
        for label, prob in zip(all_labels, all_probs):
            if label not in label_scores or prob > label_scores[label]:
                label_scores[label] = prob
        
        # Store scores
        for label, score in label_scores.items():
            job_label_scores[(job_id, label)] = score
    
    # 7) Apply precision-focused filtering --------------------------------
    print(f"\nSTAGE 3: Precision-focused filtering")
    
    # For each job, analyze score distribution and apply filtering
    job_predictions_final = {}
    
    # Set precision mode parameters
    if args.precision_mode == 'strict':
        # Strict mode: Prioritize precision even at cost of recall
        base_threshold = max(0.4, args.thr)
        z_threshold = 1.5
        top_percent = 10
    elif args.precision_mode == 'balanced':
        # Balanced mode: Try to balance precision and recall
        base_threshold = max(0.2, args.thr)
        z_threshold = 1.0
        top_percent = 15
    else:  # recall mode
        # Recall mode: Prioritize recall while improving precision
        base_threshold = max(0.1, args.thr)
        z_threshold = 0.5
        top_percent = 30
    
    for job_id in job_sentences.keys():
        # Get all scores for this job
        job_scores = []
        job_labels = []
        
        for (j, label), score in job_label_scores.items():
            if j == job_id:
                job_scores.append(score)
                job_labels.append(label)
        
        if not job_scores:
            job_predictions_final[job_id] = set()
            continue
        
        # Calculate statistics for this job
        mean_score = np.mean(job_scores)
        std_score = np.std(job_scores)
        
        # Get percentile threshold (e.g. top 10% of scores)
        percentile_threshold = np.percentile(job_scores, 100 - top_percent) if job_scores else 0
        
        # Multi-criteria selection
        selected_labels = set()
        
        # For analysis
        gt_labels = set(normalize(l) for l in ground.get(job_id, []))
        analysis_scores = {'selected': [], 'rejected': [], 'relevant': [], 'non_relevant': []}
        
        for label, score in zip(job_labels, job_scores):
            # Calculate z-score (standard deviations from mean)
            z_score = (score - mean_score) / std_score if std_score > 0 else 0
            
            # Multi-criteria selection:
            selected = False
            
            # Criteria 1: Score above absolute threshold
            if score >= base_threshold:
                selected = True
            
            # Criteria 2: Score is an outlier (significantly above mean)
            elif z_score >= z_threshold:
                selected = True
            
            # Criteria 3: Score is in top percentile
            elif score >= percentile_threshold:
                selected = True
            
            # Apply selection
            if selected:
                selected_labels.add(label)
                analysis_scores['selected'].append(score)
            else:
                analysis_scores['rejected'].append(score)
            
            # Track relevant/non-relevant scores for analysis
            if label in gt_labels:
                analysis_scores['relevant'].append(score)
            else:
                analysis_scores['non_relevant'].append(score)
        
        # Limit the number of labels if specified
        if args.max_labels and len(selected_labels) > args.max_labels:
            # Sort by score and keep top N
            label_scores = [(label, job_label_scores.get((job_id, label), 0)) 
                           for label in selected_labels]
            label_scores.sort(key=lambda x: x[1], reverse=True)
            selected_labels = set(label for label, _ in label_scores[:args.max_labels])
        
        # Store final predictions
        job_predictions_final[job_id] = selected_labels
        
        # Create analysis plot for this job
        if analysis_scores['selected'] and analysis_scores['rejected']:
            plot_histogram(
                analysis_scores['selected'] + analysis_scores['rejected'],
                f"Job {job_id} Score Distribution",
                f"job_{job_id}_scores.png",
                base_threshold
            )
    
    # 8) Evaluate final results ---------------------------------------------
    final_precision, final_recall, final_f1, job_metrics = metrics(job_predictions_final, ground)
    
    print(f"\nFinal Results with {args.precision_mode} mode:")
    print(f"  Precision: {final_precision:.4f}")
    print(f"  Recall: {final_recall:.4f}")
    print(f"  F1 Score: {final_f1:.4f}")
    
    # Per-job metrics
    print("\nPer-job metrics:")
    for job_id, precision, recall, f1, tp, fp, fn in job_metrics:
        print(f"  Job {job_id}: P={precision:.2f}, R={recall:.2f}, F1={f1:.2f}, TP={tp}, FP={fp}, FN={fn}")
    
    # 9) Save results -------------------------------------------------------
    output = {
        "jobs": len(ground),
        "sentences": len(sents),
        "bi_encoder": {
            "topn": args.bi_topn,
            "precision": bi_precision,
            "recall": bi_recall,
            "f1": bi_f1
        },
        "cross_encoder": {
            "threshold": args.thr,
            "precision_mode": args.precision_mode,
            "precision": final_precision,
            "recall": final_recall,
            "f1": final_f1
        },
        "parameters": vars(args),
        "job_metrics": [{
            "job_id": jm[0],
            "precision": jm[1],
            "recall": jm[2],
            "f1": jm[3],
            "tp": jm[4],
            "fp": jm[5],
            "fn": jm[6]
        } for jm in job_metrics]
    }
    
    with open(args.out, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.out}")

if __name__ == "__main__":
    main()