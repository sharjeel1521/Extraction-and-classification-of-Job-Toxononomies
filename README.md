# 🧠 Job–Attribute Matching System (Bi-Encoder + Cross-Encoder)

At scale, matching job descriptions to their corresponding skill or attribute definitions presents a dual challenge: speed and precision. Our team designed a real-time NLP system that leverages Bi-Encoder retrieval for rapid candidate selection and Cross-Encoder re-ranking for high-fidelity mapping.

Bi-Encoder Pipeline:
We engineered a restart-safe training data generator that uses spaCy-based sentence splitting and positive/negative pair sampling. Training leveraged E5 embeddings with MultipleNegativesRankingLoss on a multi-GPU cluster, enabling rapid embedding generation while ensuring restart-friendly, memory-efficient operations.

Cross-Encoder Pipeline:
To refine candidate selections, we fine-tuned ms-marco-MiniLM with an extended 512-token input for binary classification. Mixed-precision training allowed us to capture nuanced job–attribute relationships without sacrificing latency. This layer ensures that the top candidates from the Bi-Encoder are accurately re-ranked.

Real-Time Deployment:
New job descriptions entering the system are streamed through a Kafka pipeline, connected to a feature store for pre-computed embeddings. The Bi-Encoder generates embeddings in real-time using multi-GPU acceleration, retrieves top candidates via FAISS similarity search, and the Cross-Encoder re-ranks them before final mapping.

Monitoring & Feedback:
A logging and drift monitoring layer continuously captures model feedback, detects shifts in input distributions, and triggers canary deployments for shadow traffic testing. This ensures that the system maintains high accuracy while scaling to thousands of new job postings daily.

Impact & Evaluation:
Using threshold-based and top-N evaluations on golden datasets, the system achieved measurable improvements in precision, recall, and F1, providing fast, reliable, and interpretable job–attribute mappings. This end-to-end architecture combines speed, accuracy, and robustness, enabling real-time decision-making for job matching at enterprise scale.
Outcome: Delivered a modular, restart-friendly, multi-GPU pipeline capable of high-throughput training and inference, enabling accurate large-scale job–attribute classification with explainability and reproducibility.


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




```mermaid

flowchart TD

%% Real-Time Job-Attribute Matching with Production Features


%% Real-Time Job-Attribute Matching with Production Features

subgraph Ingestion[Real-Time Job Ingestion]
    J1[New Job Description] --> K1[Kafka Topic / Stream]
    K1 --> FStore[Feature Store Lookup / Update]
end

subgraph BI[Bi-Encoder Pipeline]
    K1 --> B1[Sentence Splitting - spaCy]
    B1 --> B2[Positive/Negative Pair Sampling]
    B2 --> B3[Restart-Safe Checkpoints / Train Data]
    B3 --> B4[Multi-GPU Training Layer]
    B4 --> B5[Trained Bi-Encoder Model]
end

subgraph CE[Cross-Encoder Pipeline]
    C1[Job Description + Candidate Attributes] --> C2[Pair Generation]
    C2 --> C3[Train/Dev Data - 512 tokens]
    C3 --> D1[Cross-Encoder Fine-Tuning / Mixed Precision]
    D1 --> D2[Trained Cross-Encoder Model]
end

subgraph RealTime_Inference[Real-Time Inference]
    K1 --> B5
    B5 --> E1[FAISS Retrieval for Candidate Attributes]
    D2 --> E1
    E1 --> E2[Cross-Encoder Re-Ranking]
    E2 --> F[Final Job ↔ Attribute Mapping]
end

subgraph Production_Features[Production Deployment & Monitoring]
    F --> Logger[Logger & Feedback Capture]
    Logger --> Drift[Drift Monitoring]
    Drift --> FStore
    F --> Canary[Canary / Shadow Traffic Testing]
end
