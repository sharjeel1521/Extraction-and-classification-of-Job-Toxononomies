# 🧠 Job–Attribute Matching System (Bi-Encoder + Cross-Encoder)

A scalable NLP system for mapping **job descriptions ↔ attribute definitions** using **Bi-Encoder retrieval** and **Cross-Encoder re-ranking**.  
The pipeline handles **large-scale labeled datasets**, supports **multi-GPU training**, and is optimized for **restart-friendly, memory-safe training**.
Designed and implemented a scalable Job–Attribute Matching System leveraging Bi-Encoder retrieval for speed and Cross-Encoder re-ranking for precision.

Bi-Encoder pipeline: Built training data generator with spaCy-based sentence splitting, positive/negative pair sampling, and restart-safe checkpoints; trained using E5 embeddings with MultipleNegativesRankingLoss across multi-GPU infrastructure.

Cross-Encoder pipeline: Fine-tuned ms-marco-MiniLM with extended 512-token input, binary classification, and mixed-precision training to capture nuanced job–attribute relationships.

Retrieval Evaluation: Integrated FAISS for scalable similarity search, implemented threshold-based and top-N ranking evaluations, delivering measurable improvements in precision, recall, and F1 against golden datasets.

Outcome: Delivered a modular, restart-friendly, multi-GPU pipeline capable of high-throughput training and inference, enabling accurate large-scale job–attribute classification with explainability and reproducibility.
---

## ⚡ Bi-Encoder Pipeline

### 🔹 Part 1: Creating Training Data
**Path:**  
`Part1_Samples_GPU_Multithreading/train_biencoder_RvPython2.2_Part1_Full_Faster_GPU_Multithreading_Rv1.py`  
`Part1_Samples_GPU_Multithreading/train_biencoder_RvPython2.2_Part1_Full_Faster_GPU_Multithreading_Rv2.py`

**Logic:**
- Uses **spaCy (`en_core_web_trf`)** to split job descriptions into sentences (with offsets).
- Cleans text: removes bullets, numbering, whitespace.
- Handles exceptions (e.g., `Sr.` / `Dr.` not split).
- Builds **positive pairs**: `(sentence, correct label definition)`.
- Adds **negative pairs**: `(sentence, random wrong labels)`.
- Jobs are processed in **batches with multithreading**, checkpointed to pickle files to ensure restart safety.

**Outputs:**
- ✅ Full Training Dataset:  
  - `train_samples_final.pkl`  
  - `dev_samples_final.pkl`  
- ✅ Sample Dataset (debug):  
  - `train_samples_5k.pkl`  
  - `dev_samples_1k.pkl`

---

### 🔹 Part 2: Training the Bi-Encoder
**Path:**  
`ToSubmit_part2_train_e5_mnrl_Rv3/part2_train_e5_mnrl_Rv3.py`

**Logic:**
- Model: `intfloat/e5-base-v2`
- Adds **E5 prefixes**: `"query: ..."`, `"passage: ..."`
- Uses **MultipleNegativesRankingLoss** for contrastive training.  
  (Earlier runs with CosineLoss → unstable, loss stuck near 0.)
- Positive only pairs (`label=1.0`) used.
- HuggingFace **Accelerate** for multi-GPU, supports fp16/bf16.
- **Gradient accumulation** → simulate large batch sizes.
- **Checkpointing & Faiss recall@K eval** mid-training.

**Output:**
- `ToSubmit_part2_train_e5_mnrl_Rv3/Part2_O1_MNRL_Prefix5_Multi_GPU_Final`

---

### 🔹 Part 3: Evaluation / Inference
**Path:**  
`Part3_BI_Encoder_Evaluation.ipynb`

**Logic:**
- Loads trained model checkpoint.
- Embeds **job descriptions** (as queries) + **attribute definitions** (as passages).
- Stores attributes in a **FAISS index**.
- Evaluates retrieval:
  - **Threshold-based** (sim ≥ 0.9 → positive).
  - **Ranking-based** (Top-N check).
- Metrics: **Precision, Recall, F1** across thresholds and top-N.

---

## ⚡ Cross-Encoder Pipeline

### 🔹 Part 1: Creating Training Data
**Path:**  
`Part1_Cross_Training_Data/Part1_Cross_Training.py`

**Logic:**
- Uses **job descriptions + attribute definitions + labeled pairs**.
- Forms **positive (job, attribute)** and **negative (job, wrong attribute)** pairs.
- Saves 4 datasets:
  - Train / Dev (full)
  - Train / Dev (small debug)

**Outputs:**
- Full:  
  - `Part1_Cross_Training_Data/train.pkl`  
  - `Part1_Cross_Training_Data/dev.pkl`  
- Small Debug:  
  - `train_small.pkl`, `dev_small.pkl`

---

### 🔹 Part 2: Training the Cross-Encoder
**Paths:**  
- Sentence-level: `train_cross_encoder_1GPU_Rv3.py`  
- Job-level: `train_cross_encoder_1GPU_Rv4`

**Logic:**
- Fine-tunes **cross-encoder (ms-marco-MiniLM)**.
- Input: `(job text, attribute definition)` → binary label.
- Max sequence length = **512 tokens** (extended for large inputs).
- Loss: **BCEWithLogitsLoss**.
- Training with **torch.cuda.amp** (mixed precision).
- Saves checkpoints every N steps + at each epoch.

**Output:**
- `Part1_Cross_Training_Data/Part2_Cross_May5`

---

### 🔹 Part 3: Full Inference
**Path:**  
`Part1_Cross_Training_Data/Full_Pipeline_Python_Rv6.py`

Runs **end-to-end evaluation** on Golden Dataset.

---

## 📂 Key Files

- **New Dataset:** `butterfly/src/notebooks/New_Dataset_May9`  
- **Definitions:** `Defs.tsv`  
- **Labels:** `all_labeled_data_dedpued_June2024.csv`, `Golden_dataset_labels.csv`, `Rosetta-labels-us.csv`  
- **Checkpoints:** `Cross_Encoder_4k_attr_jobs_to_label.pq`, `Fl.pickle`  

---

## ✅ Summary of Strengths
- **Bi-Encoder** → scalable retrieval with FAISS.  
- **Cross-Encoder** → precise re-ranking with extended token support (512+).  
- **Restart-friendly, streaming training** with pickle checkpoints.  
- **Multi-GPU + Mixed Precision** → efficient training at scale.  

---
