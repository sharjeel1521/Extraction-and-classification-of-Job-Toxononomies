#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train_biencoder_RvPython2.2_Claude_Part1_Full_Faster_GPU_Multithreading_Rv1_Grok.py

Optimized script to:
1) Perform spaCy sentence segmentation on a GPU (3-4 hours for full dataset)
2) Build positive and negative examples efficiently using multiprocessing
3) Save spaCy results for crash recovery
4) Organize outputs in a timestamped folder
5) Allow user-defined negative sample sizes
6) Handle NaN values in preflabel to prevent errors

Requires:
  pip install spacy spacy-transformers sentence-transformers pandas tqdm
  python -m spacy download en_core_web_trf
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings
import re
import pickle
import spacy
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import InputExample
import random
from datetime import datetime
import logging
import json
import argparse
from typing import List, Dict, Tuple, Optional, Set
import multiprocessing as mp
from functools import partial
import psutil
import bisect

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("data_preparation_multithreaded.log")]
)
logger = logging.getLogger(__name__)

# Load SpaCy Model for GPU usage
def load_nlp_model(model_name: str = "en_core_web_trf"):
    logger.info("Requesting GPU usage via spaCy...")
    spacy.require_gpu()
    logger.info(f"Loading spaCy transformer model: {model_name}")
    nlp = spacy.load(model_name)
    for pipe_name in list(nlp.pipe_names):
        if pipe_name not in ("transformer", "senter", "sentencizer"):
            nlp.remove_pipe(pipe_name)
    if "senter" not in nlp.pipe_names and "sentencizer" not in nlp.pipe_names:
        logger.info("Adding sentencizer for sentence boundary detection.")
        nlp.add_pipe("sentencizer")
    logger.info(f"Final pipeline components: {nlp.pipe_names}")
    return nlp

# Sentence Splitting in Large Batches
def batch_sentence_splitting(job_desc_dict: Dict[str, str], nlp, batch_size: int = 1500):
    job_ids = list(job_desc_dict.keys())
    texts = [job_desc_dict[jid] if isinstance(job_desc_dict[jid], str) else "" for jid in job_ids]
    logger.info(f"Running spaCy sentence segmentation on {len(job_ids)} job descriptions (batch_size={batch_size})...")
    docs = nlp.pipe(texts, batch_size=batch_size)
    jobid2sentences = {}
    for jid, doc in tqdm(zip(job_ids, docs), total=len(job_ids), desc="spaCy nlp.pipe"):
        if not doc.text or not doc.text.strip():
            jobid2sentences[jid] = []
            continue
        sents_for_this_doc = []
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if len(sent_text) < 5:
                continue
            sents_for_this_doc.append((sent_text, sent.start_char, sent.end_char))
        jobid2sentences[jid] = sents_for_this_doc
    return jobid2sentences

# Fallback Snippet for Out-of-Bound Indices
def get_context_snippet(original_text: str, begin_idx: int, end_idx: int, window: int = 100) -> str:
    i = begin_idx
    while i > 0 and original_text[i - 1] not in '.!?\n':
        i -= 1
    start = max(0, i - window)
    j = end_idx
    while j < len(original_text) - 1 and original_text[j] not in '.!?\n':
        j += 1
    finish = min(len(original_text), j + window)
    snippet = original_text[start:finish].strip()
    return snippet

# Clean Sentence Text
def clean_sentence(sentence: str) -> str:
    sentence = re.sub(r'\s+', ' ', sentence).strip()
    sentence = re.sub(r'^[\*\-\•\+]\s*', '', sentence)
    sentence = re.sub(r'^\s*[0-9]+\.\s*', '', sentence)
    return sentence

# Find Sentence Using Binary Search
def find_sentence(sentences, begin_idx, end_idx):
    if not sentences:
        return ""
    # Ensure sentences are sorted by start_char
    sentences = sorted(sentences, key=lambda x: x[1])
    starts = [s[1] for s in sentences]
    ends = [s[2] for s in sentences]
    idx = bisect.bisect_right(starts, begin_idx) - 1
    if 0 <= idx < len(sentences) and starts[idx] <= begin_idx < ends[idx] and end_idx <= ends[idx]:
        return sentences[idx][0]
    else:
        return ""

# Worker Function for Multiprocessing
def process_job_id(
    job_id: str,
    job_desc_dict: Dict[str, str],
    job_id_to_labels: Dict[str, List[Dict]],
    preflabel_to_label_def_text: Dict[str, str],
    jobid2sentences: Dict[str, List[Tuple[str, int, int]]],
    all_labels: List[str],
    num_negatives: int,
    is_train: bool
) -> Tuple[List[Dict], List[InputExample]]:
    original_text = job_desc_dict.get(job_id, "")
    if not original_text.strip():
        return [], []
    sents_with_offsets = jobid2sentences.get(job_id, [])
    if not sents_with_offsets:
        return [], []
    matching_rows = job_id_to_labels.get(job_id, [])
    if not matching_rows:
        return [], []
    job_rows = []
    local_examples = []
    for row in matching_rows:
        try:
            begin_idx = int(row["begin"])
            end_idx = int(row["end"])
            preflabel = row["preflabel"]
            if begin_idx < 0 or end_idx > len(original_text) or begin_idx >= end_idx:
                continue
            index_text = original_text[begin_idx:end_idx]
            found_sentence = find_sentence(sents_with_offsets, begin_idx, end_idx)
            if not found_sentence:
                found_sentence = get_context_snippet(original_text, begin_idx, end_idx)
            found_sentence = clean_sentence(found_sentence)
            if len(found_sentence) < 10:
                continue
            label_def_text = preflabel_to_label_def_text[preflabel]
            job_rows.append({
                "Job ID": job_id,
                "Job Sentence": found_sentence,
                "Index Text": index_text,
                "Preflabel": preflabel,
                "Begin": begin_idx,
                "End": end_idx
            })
            positive_example = InputExample(texts=[found_sentence, label_def_text], label=1.0)
            local_examples.append(positive_example)
            for _ in range(num_negatives):
                neg_label = random.choice(all_labels)
                while neg_label == label_def_text:
                    neg_label = random.choice(all_labels)
                negative_example = InputExample(texts=[found_sentence, neg_label], label=0.0)
                local_examples.append(negative_example)
        except Exception as e:
            logger.error(f"Error on job_id={job_id}: {e}")
            continue
    return job_rows, local_examples

# Process Job IDs in Batches
def process_job_ids_batch(
    job_ids_batch,
    job_desc_dict,
    job_id_to_labels,
    preflabel_to_label_def_text,
    jobid2sentences,
    all_labels,
    num_negatives,
    is_train
):
    batch_rows = []
    batch_examples = []
    for job_id in job_ids_batch:
        rows, examples = process_job_id(
            job_id, job_desc_dict, job_id_to_labels, preflabel_to_label_def_text,
            jobid2sentences, all_labels, num_negatives, is_train
        )
        batch_rows.extend(rows)
        batch_examples.extend(examples)
    return batch_rows, batch_examples

# Create Batches
def create_batches(items, batch_size):
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

# Main Function
def main(
    csv_labeled_path: str = "all_labeled_data_dedpued_June2024.csv",
    pickle_jobdesc_path: str = "fl.pickle",
    defs_path: str = "defs.tsv",
    output_dir: str = "./Part1_Samples_GPU_Multithreading",
    output_prefix: str = None,
    train_job_ids_ratio: float = 0.8,
    num_workers: int = 85,
    limit_num_jobids: int = None,
    negative_examples_per_positive: int = 1,
    spacy_model: str = "en_core_web_trf",
    batch_size: int = 1500,
    chunk_size: int = 10
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not output_prefix:
        output_prefix = timestamp
    output_subdir = os.path.join(output_dir, f"{output_prefix}_Output_Part_1")
    os.makedirs(output_subdir, exist_ok=True)

    output_csv = os.path.join(output_subdir, "Bi_Encoder_Labeled_Data.csv")
    train_pickle = os.path.join(output_subdir, "train_samples.pkl")
    dev_pickle = os.path.join(output_subdir, "dev_samples.pkl")
    meta_json = os.path.join(output_subdir, "metadata.json")
    sentences_file = os.path.join(output_subdir, "jobid2sentences.pkl")

    logger.info(f"System info: {mp.cpu_count()} CPUs, {psutil.virtual_memory().total / (1024**3):.2f} GB RAM")
    logger.info(f"Reading labeled CSV: {csv_labeled_path}")
    df_labeled = pd.read_csv(csv_labeled_path)
    df_labeled["job_id"] = df_labeled["job_id"].astype(str)

    # Clean NaN values and invalid preflabels
    df_labeled = df_labeled[df_labeled["preflabel"].notna() & (df_labeled["preflabel"] != 'nan')]
    logger.info(f"After cleaning NaN preflabels, {len(df_labeled)} rows remain")

    # Preprocess labeled data
    job_id_to_labels = df_labeled.groupby('job_id').apply(lambda x: x.to_dict(orient='records'), include_groups=False).to_dict()

    logger.info(f"Reading job descriptions: {pickle_jobdesc_path}")
    with open(pickle_jobdesc_path, "rb") as f:
        job_desc_dict = pickle.load(f)
    job_desc_dict = {str(k): str(v) for k, v in job_desc_dict.items() if isinstance(v, (str, bytes))}

    logger.info(f"Reading definitions: {defs_path}")
    df_defs = pd.read_csv(defs_path, sep="\t")
    preflabel2definition = {}
    if "label" in df_defs.columns and "definition" in df_defs.columns:
        for _, row in df_defs.iterrows():
            skill_name = str(row["label"]).strip()
            definition = str(row["definition"]).strip()
            preflabel2definition[skill_name] = definition
    else:
        logger.warning("Defs TSV missing columns [label, definition]. Will skip definitions.")

    all_preflabels = set(df_labeled['preflabel'])
    preflabel_to_label_def_text = {}
    for preflabel in all_preflabels:
        definition = preflabel2definition.get(preflabel, "")
        label_def_text = preflabel if not definition else f"{preflabel}: {definition}"
        preflabel_to_label_def_text[preflabel] = label_def_text

    all_job_ids = sorted(job_id_to_labels.keys())
    logger.info(f"Found {len(all_job_ids)} unique job_ids in labeled CSV.")

    if limit_num_jobids and limit_num_jobids < len(all_job_ids):
        logger.info(f"Limiting job_ids to first {limit_num_jobids}.")
        all_job_ids = all_job_ids[:limit_num_jobids]
        job_desc_dict = {job_id: job_desc_dict[job_id] for job_id in all_job_ids if job_id in job_desc_dict}
        job_id_to_labels = {job_id: job_id_to_labels[job_id] for job_id in all_job_ids if job_id in job_id_to_labels}

    random.seed(42)
    random.shuffle(all_job_ids)
    split_idx = int(len(all_job_ids) * train_job_ids_ratio)
    train_job_ids = all_job_ids[:split_idx]
    dev_job_ids = all_job_ids[split_idx:]
    logger.info(f"Training job_ids: {len(train_job_ids)}, Dev job_ids: {len(dev_job_ids)}")

    # Load or compute sentence segmentation
    if os.path.exists(sentences_file):
        logger.info(f"Loading existing sentence segmentation from {sentences_file}")
        with open(sentences_file, "rb") as f:
            jobid2sentences = pickle.load(f)
    else:
        nlp = load_nlp_model(model_name=spacy_model)
        jobid2sentences = batch_sentence_splitting(job_desc_dict, nlp, batch_size=batch_size)
        with open(sentences_file, "wb") as f:
            pickle.dump(jobid2sentences, f)

    all_labels = list(all_preflabels)

    logger.info(f"Using {num_workers} workers with chunk_size={chunk_size}...")
    train_batches = create_batches(train_job_ids, chunk_size)
    dev_batches = create_batches(dev_job_ids, chunk_size)
    logger.info(f"Created {len(train_batches)} train batches and {len(dev_batches)} dev batches")

    pool = mp.Pool(processes=num_workers)
    logger.info("Processing training job IDs in batches...")
    train_results = list(tqdm(
        pool.imap_unordered(
            partial(process_job_ids_batch, job_desc_dict=job_desc_dict, job_id_to_labels=job_id_to_labels,
                    preflabel_to_label_def_text=preflabel_to_label_def_text, jobid2sentences=jobid2sentences,
                    all_labels=all_labels, num_negatives=negative_examples_per_positive, is_train=True),
            train_batches
        ),
        total=len(train_batches),
        desc="Processing train batches"
    ))

    logger.info("Processing dev job IDs in batches...")
    dev_results = list(tqdm(
        pool.imap_unordered(
            partial(process_job_ids_batch, job_desc_dict=job_desc_dict, job_id_to_labels=job_id_to_labels,
                    preflabel_to_label_def_text=preflabel_to_label_def_text, jobid2sentences=jobid2sentences,
                    all_labels=all_labels, num_negatives=negative_examples_per_positive, is_train=False),
            dev_batches
        ),
        total=len(dev_batches),
        desc="Processing dev batches"
    ))

    pool.close()
    pool.join()

    csv_rows = []
    train_examples = []
    dev_examples = []
    for batch_rows, batch_examples in train_results:
        csv_rows.extend(batch_rows)
        train_examples.extend(batch_examples)
    for batch_rows, batch_examples in dev_results:
        csv_rows.extend(batch_rows)
        dev_examples.extend(batch_examples)

    random.shuffle(train_examples)
    random.shuffle(dev_examples)

    logger.info(f"Saving CSV with {len(csv_rows)} rows -> {output_csv}")
    df_out = pd.DataFrame(csv_rows)
    df_out.to_csv(output_csv, index=False)

    logger.info(f"Saving train examples ({len(train_examples)}) -> {train_pickle}")
    with open(train_pickle, "wb") as f:
        pickle.dump(train_examples, f)

    logger.info(f"Saving dev examples ({len(dev_examples)}) -> {dev_pickle}")
    with open(dev_pickle, "wb") as f:
        pickle.dump(dev_examples, f)

    metadata = {
        "timestamp": timestamp,
        "num_job_ids": len(all_job_ids),
        "num_train_examples": len(train_examples),
        "num_dev_examples": len(dev_examples),
        "negative_examples_per_positive": negative_examples_per_positive,
        "spacy_model": spacy_model,
        "train_job_ids_ratio": train_job_ids_ratio,
        "batch_size": batch_size,
        "chunk_size": chunk_size,
        "num_workers": num_workers,
        "files": {
            "csv": output_csv,
            "train_pickle": train_pickle,
            "dev_pickle": dev_pickle,
            "sentences_file": sentences_file
        }
    }
    with open(meta_json, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Metadata saved -> {meta_json}")
    logger.info("Finished all steps successfully!")
    return output_csv, train_pickle, dev_pickle, meta_json

# CLI Entry
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training data for a bi-encoder model with GPU-based spaCy and multithreading.")
    parser.add_argument("--csv_labeled_path", type=str, default="all_labeled_data_dedpued_June2024.csv")
    parser.add_argument("--pickle_jobdesc_path", type=str, default="fl.pickle")
    parser.add_argument("--defs_path", type=str, default="defs.tsv")
    parser.add_argument("--output_dir", type=str, default="./Part1_Samples_GPU_Multithreading")
    parser.add_argument("--output_prefix", type=str, default=None)
    parser.add_argument("--train_job_ids_ratio", type=float, default=0.8)
    parser.add_argument("--num_workers", type=int, default=85)
    parser.add_argument("--limit_num_jobids", type=int, default=None)
    parser.add_argument("--negative_examples_per_positive", type=int, default=1)
    parser.add_argument("--spacy_model", type=str, default="en_core_web_trf")
    parser.add_argument("--batch_size", type=int, default=1500)
    parser.add_argument("--chunk_size", type=int, default=10)

    args = parser.parse_args()

    try:
        import psutil
    except ImportError:
        import subprocess
        subprocess.check_call(["pip", "install", "psutil"])
        import psutil

    csv_file, train_file, dev_file, meta_file = main(
        csv_labeled_path=args.csv_labeled_path,
        pickle_jobdesc_path=args.pickle_jobdesc_path,
        defs_path=args.defs_path,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        train_job_ids_ratio=args.train_job_ids_ratio,
        num_workers=args.num_workers,
        limit_num_jobids=args.limit_num_jobids,
        negative_examples_per_positive=args.negative_examples_per_positive,
        spacy_model=args.spacy_model,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size
    )

    logger.info("\n[DONE] Output files created:")
    logger.info(f"  -> CSV: {csv_file}")
    logger.info(f"  -> Train Examples: {train_file}")
    logger.info(f"  -> Dev Examples: {dev_file}")
    logger.info(f"  -> Metadata: {meta_file}")

    logger.info("\n[DEBUG] Displaying first 3 rows from the CSV file:")
    df_check = pd.read_csv(csv_file)
    logger.info(df_check.head(3).to_string())

    logger.info("[DEBUG] Displaying first 3 train examples:")
    with open(train_file, "rb") as f:
        train_data = pickle.load(f)
        for idx, item in enumerate(train_data[:3], start=1):
            logger.info(f"  Train Sample {idx}: {item}")

    logger.info("[DEBUG] Displaying first 3 dev examples:")
    with open(dev_file, "rb") as f:
        dev_data = pickle.load(f)
        for idx, item in enumerate(dev_data[:3], start=1):
            logger.info(f"  Dev Sample {idx}: {item}")