<div align="center">
<img src="https://raw.githubusercontent.com/rpaut03l/NCA-GENL/main/nca_genl.svg" width="1300" alt="NCA-GENL — NVIDIA Generative AI & LLMs Associate"/>
</div>

# NVIDIA Generative AI & LLMs Associate (NCA-GENL) — 2-Day Exam Prep

> **Exam:** 50–60 questions | 60 minutes | Multiple choice
> **Goal:** Pass in 2 days with focused, high-yield study sprints.

---

## Strategy Overview

| Day | Focus Areas | Weight |
|-----|------------|--------|
| **Day 1** | ML & Neural Network Foundations + Transformer / NLP Architecture | **50%** of exam |
| **Day 2** | NVIDIA Deployment Stack + Trustworthy AI | **50%** of exam |

**Study method:** Read → Mnemonic → Active Recall (practice questions after each section).

---

# DAY 1 — Foundations & Transformers

---

## Section 1: Machine Learning & Neural Network Foundations (30%)

### 1.1 Types of Machine Learning

| Type | Description | Example |
|------|------------|---------|
| **Supervised** | Labeled data → learns mapping | Spam detection |
| **Unsupervised** | No labels → finds patterns | Customer clustering |
| **Semi-supervised** | Small labeled + large unlabeled | Medical imaging |
| **Self-supervised** | Creates own labels from data | GPT pre-training (next-token prediction) |
| **Reinforcement** | Agent → action → reward signal | RLHF for LLMs |

> **Mnemonic — "SUSS-R"**
> **S**upervised, **U**nsupervised, **S**emi-supervised, **S**elf-supervised, **R**einforcement
> *"SUSS out the Right learning type"*

### 1.2 Neural Network Building Blocks

| Component | Purpose |
|-----------|---------|
| **Neuron** | Weighted sum + bias → activation function |
| **Activation Functions** | Introduce non-linearity (ReLU, Sigmoid, Tanh, Softmax) |
| **Loss Function** | Measures how wrong the model is (Cross-Entropy, MSE) |
| **Optimizer** | Updates weights to minimize loss (SGD, Adam) |
| **Backpropagation** | Chain rule to compute gradients layer-by-layer |

> **Mnemonic — "NALOB"**
> **N**euron → **A**ctivation → **L**oss → **O**ptimizer → **B**ackprop
> *"NALOB: Neurons Activate, Loss Optimizes Backward"*

### 1.3 Key Activation Functions

| Function | Range | Use Case | Gotcha |
|----------|-------|----------|--------|
| **ReLU** | [0, ∞) | Hidden layers (default) | Dying ReLU (outputs 0 forever) |
| **Sigmoid** | (0, 1) | Binary classification output | Vanishing gradients |
| **Tanh** | (-1, 1) | Hidden layers (legacy) | Vanishing gradients |
| **Softmax** | (0, 1), sums to 1 | Multi-class output | Used in attention scores too |

> **Mnemonic — "Real Students Take Softmax"**
> **R**eLU → **S**igmoid → **T**anh → **S**oftmax

### 1.4 Overfitting vs. Underfitting

| Problem | Symptom | Solutions |
|---------|---------|-----------|
| **Overfitting** | High train accuracy, low test accuracy | Dropout, regularization (L1/L2), more data, early stopping |
| **Underfitting** | Low accuracy everywhere | Bigger model, more features, train longer |

> **Mnemonic — "DROP IT"** (Overfitting cures)
> **D**ropout, **R**egularization, **O**ut-of-sample validation, **P**rune less, **I**ncrease data, **T**erminate early

### 1.5 Key Architectures

| Architecture | Best For | Key Feature |
|-------------|----------|-------------|
| **CNN** | Images, spatial data | Convolutional filters, pooling |
| **RNN** | Sequential data | Hidden state carries memory |
| **LSTM** | Long sequences | Gates (forget, input, output) solve vanishing gradients |
| **Transformer** | NLP, modern LLMs | Self-attention, parallelizable |

> **Mnemonic — "CRLT"**
> *"CaRs Love Transformers"* — CNN → RNN → LSTM → Transformer

### 1.6 Training Concepts

| Concept | Definition |
|---------|-----------|
| **Epoch** | One full pass through the training data |
| **Batch Size** | Number of samples per gradient update |
| **Learning Rate** | Step size for weight updates (too high = diverge, too low = slow) |
| **Gradient Descent** | Move in the direction that reduces loss |
| **Vanishing Gradient** | Gradients shrink to ~0 in deep networks → early layers don't learn |

> **Mnemonic — "Every Batch Learns by Gradient Victory"**
> **E**poch, **B**atch, **L**earning rate, **G**radient descent, **V**anishing gradient

---

## Section 2: NLP & LLM Architecture (20%)

### 2.1 The Transformer Architecture

The Transformer is the foundation of all modern LLMs (GPT, BERT, T5, LLaMA).

**Core Idea:** Replace recurrence with **self-attention** — process all tokens in parallel.

```
Input → [Embedding + Positional Encoding] → [Encoder Stack] → [Decoder Stack] → Output
```

| Component | What It Does |
|-----------|-------------|
| **Input Embedding** | Converts tokens → dense vectors |
| **Positional Encoding** | Injects word order info (sine/cosine or learned) |
| **Multi-Head Attention** | Multiple attention heads capture different relationships |
| **Feed-Forward Network** | Two linear layers + ReLU per position |
| **Layer Normalization** | Stabilizes training |
| **Residual Connections** | Skip connections to help gradient flow |

> **Mnemonic — "Every Position Matters, Feeding Layers Right"**
> **E**mbedding, **P**ositional encoding, **M**ulti-head attention, **F**eed-forward, **L**ayer norm, **R**esidual connections

### 2.2 Self-Attention Mechanism (CRITICAL for exam)

**Formula:** `Attention(Q, K, V) = softmax(QK^T / √d_k) × V`

| Symbol | Meaning |
|--------|---------|
| **Q** (Query) | "What am I looking for?" |
| **K** (Key) | "What do I contain?" |
| **V** (Value) | "What information do I provide?" |
| **√d_k** | Scaling factor to prevent extreme softmax values |

> **Mnemonic — "Queens Keep Valuable Secrets"**
> **Q**uery asks, **K**ey matches, **V**alue delivers, **S**cale stabilizes

**Why multi-head?** Each head learns a *different* type of relationship (syntax, semantics, co-reference). Outputs are concatenated and projected.

### 2.3 Encoder vs. Decoder Models

| Type | Architecture | Training Objective | Example | Best For |
|------|-------------|-------------------|---------|----------|
| **Encoder-only** | Bidirectional attention | Masked Language Model (MLM) | BERT | Classification, NER, QA |
| **Decoder-only** | Causal (left-to-right) attention | Next-token prediction | GPT, LLaMA | Text generation |
| **Encoder-Decoder** | Full Transformer | Seq-to-seq | T5, BART | Translation, summarization |

> **Mnemonic — "BEG-T"**
> **B**ERT = Encoder, **G**PT = Decoder, **T**5 = Both
> *"BEG T5 to do both"*

### 2.4 Tokenization

| Method | How It Works |
|--------|-------------|
| **Word-level** | Each word = 1 token (huge vocabulary) |
| **BPE (Byte-Pair Encoding)** | Merges frequent character pairs iteratively (GPT uses this) |
| **WordPiece** | Similar to BPE, used by BERT |
| **SentencePiece** | Language-agnostic, works on raw text |

> **Mnemonic — "Words Break into Word-Sentence Pieces"**
> **W**ord → **B**PE → **W**ordPiece → **S**entencePiece

### 2.5 Fine-Tuning & Adaptation

| Method | Description | When to Use |
|--------|------------|-------------|
| **Full Fine-Tuning** | Update all weights | Small model, lots of data |
| **LoRA (Low-Rank Adaptation)** | Train small rank-decomposition matrices | Large model, limited compute |
| **P-Tuning / Prompt Tuning** | Learn soft prompt embeddings | Very efficient, minimal changes |
| **RLHF** | RL with human preference reward model | Align model behavior (ChatGPT) |
| **RAG** | Retrieve external docs → augment prompt | Need current/private knowledge |

> **Mnemonic — "Fine-tune LLMs Properly, Really Accurately, Guaranteed"**
> **F**ull fine-tuning, **L**oRA, **P**rompt tuning, **R**LHF, **A**ugmented retrieval (RAG), **G**uardrails

### 2.6 Key LLM Concepts

| Concept | Definition |
|---------|-----------|
| **Temperature** | Controls randomness (0 = deterministic, 1+ = creative) |
| **Top-k** | Sample from top k most likely tokens |
| **Top-p (nucleus)** | Sample from smallest set whose cumulative probability ≥ p |
| **Context Window** | Max tokens the model can process at once |
| **Hallucination** | Model generates plausible but factually wrong content |
| **Prompt Engineering** | Designing inputs to get desired outputs (zero-shot, few-shot, CoT) |

> **Mnemonic — "Transformers Think Through Context, Handling Prompts"**
> **T**emperature, **T**op-k/p, **C**ontext window, **H**allucination, **P**rompt engineering

---

# DAY 2 — NVIDIA Stack & Trustworthy AI

---

## Section 3: NVIDIA Deployment & Tools (40%) — HIGHEST WEIGHT

### 3.1 The NVIDIA AI Stack at a Glance

```
┌─────────────────────────────────────────────┐
│            APPLICATION LAYER                │
│    NeMo (Build & Fine-tune LLMs)            │
├─────────────────────────────────────────────┤
│            INFERENCE LAYER                  │
│    Triton Inference Server (Serve models)    │
│    TensorRT (Optimize for GPU inference)     │
├─────────────────────────────────────────────┤
│            DATA LAYER                       │
│    RAPIDS: cuDF, cuML, cuGraph              │
├─────────────────────────────────────────────┤
│            HARDWARE LAYER                   │
│    NVIDIA GPUs + CUDA                       │
└─────────────────────────────────────────────┘
```

> **Mnemonic — "Never Trust Raw Hardware"**
> **N**eMo (build) → **T**riton + TensorRT (serve) → **R**APIDS (data) → **H**ardware (GPU/CUDA)

### 3.2 NVIDIA NeMo

**Purpose:** End-to-end framework for building, training, and fine-tuning LLMs and speech/vision models.

| Feature | Detail |
|---------|--------|
| **NeMo Framework** | Training & fine-tuning large models at scale |
| **NeMo Guardrails** | Add safety rails to LLM applications (topic control, fact-checking) |
| **Supported models** | GPT, LLaMA, Falcon, Mixtral, and more |
| **Fine-tuning methods** | Full, LoRA, P-Tuning, SFT, RLHF |
| **Data parallelism** | Distribute training across multiple GPUs |
| **Model parallelism** | Split a single model across GPUs (tensor & pipeline parallelism) |

> **Mnemonic — "NeMo: Nice Models, Guardrailed & Parallel"**
> Build **N**ice **M**odels with **G**uardrails using **P**arallelism

### 3.3 Triton Inference Server

**Purpose:** Serve ML models at scale in production. Multi-framework, multi-model.

| Feature | Detail |
|---------|--------|
| **Multi-framework** | PyTorch, TensorFlow, TensorRT, ONNX, custom Python |
| **Dynamic batching** | Groups requests to maximize GPU utilization |
| **Model ensemble** | Chain multiple models in a pipeline |
| **Concurrent model execution** | Run different models simultaneously on same GPU |
| **Model repository** | Organized directory structure for model management |
| **HTTP/gRPC endpoints** | Standard APIs for inference requests |
| **Metrics** | Prometheus metrics for monitoring |

> **Mnemonic — "Triton Does Many Concurrent Ensembles"**
> **T**riton: **D**ynamic batching, **M**ulti-framework, **C**oncurrent execution, **E**nsembles

**Model Repository Structure:**
```
model_repository/
├── model_name/
│   ├── config.pbtxt        # Model configuration
│   └── 1/                  # Version 1
│       └── model.plan      # TensorRT engine (or .pt, .onnx, etc.)
```

### 3.4 TensorRT

**Purpose:** Optimize deep learning models for **maximum inference speed** on NVIDIA GPUs.

| Optimization | What It Does |
|-------------|-------------|
| **Layer Fusion** | Combines multiple layers into one kernel |
| **Precision Calibration** | FP32 → FP16 / INT8 (faster, less memory) |
| **Kernel Auto-Tuning** | Selects best GPU kernel for each operation |
| **Dynamic Tensor Memory** | Reuses memory buffers efficiently |
| **Multi-Stream Execution** | Parallel execution of independent ops |

> **Mnemonic — "TensorRT: Layers Fuse Precisely, Kernels Dominate, Memory Streams"**
> Or simply: **"FLPKM"** — **F**usion, **L**ower precision, **P**rofiling, **K**ernel tuning, **M**emory optimization

**Workflow:**
```
Trained Model (.onnx/.pt) → TensorRT Optimizer → Optimized Engine (.plan) → Deploy on Triton
```

**Exam tip:** TensorRT-LLM is specifically designed for LLM inference optimization. Know the difference:
- **TensorRT** = general deep learning inference optimizer
- **TensorRT-LLM** = specialized for LLMs with KV-cache, in-flight batching, tensor parallelism

### 3.5 RAPIDS (GPU-Accelerated Data Science)

| Library | GPU Replacement For | Purpose |
|---------|-------------------|---------|
| **cuDF** | Pandas | GPU-accelerated DataFrames |
| **cuML** | Scikit-learn | GPU-accelerated ML algorithms |
| **cuGraph** | NetworkX | GPU-accelerated graph analytics |
| **Dask-cuDF** | Dask + Pandas | Multi-GPU, multi-node DataFrames |

> **Mnemonic — "RAPIDS CUres Slow Data"**
> **cu**DF, **cu**ML, **cu**Graph — everything starts with **"cu"** = CUDA Unleashed

**Key exam point:** RAPIDS provides a **near-identical API** to Pandas/Scikit-learn — minimal code changes to go from CPU → GPU.

```python
# CPU (Pandas)
import pandas as pd
df = pd.read_csv("data.csv")

# GPU (cuDF) — same API!
import cudf
df = cudf.read_csv("data.csv")
```

### 3.6 NVIDIA AI Enterprise & NGC

| Component | Purpose |
|-----------|---------|
| **NGC (NVIDIA GPU Cloud)** | Catalog of pre-trained models, containers, Helm charts |
| **NVIDIA AI Enterprise** | Enterprise software suite for AI (support, security, management) |
| **NVIDIA AI Foundations** | Cloud services for custom generative AI model building |
| **CUDA** | Programming model for GPU parallel computing |

> **Mnemonic — "NGC: NVIDIA's Grand Catalog"**

### 3.7 Multi-GPU Training Strategies

| Strategy | How It Works |
|----------|-------------|
| **Data Parallelism** | Same model on each GPU, different data batches |
| **Tensor Parallelism** | Split individual layers across GPUs |
| **Pipeline Parallelism** | Split model layers into stages across GPUs |
| **ZeRO (DeepSpeed)** | Partition optimizer states, gradients, and parameters |

> **Mnemonic — "Data Travels in Pipeline Zones"**
> **D**ata parallelism, **T**ensor parallelism, **P**ipeline parallelism, **Z**eRO

---

## Section 4: Trustworthy AI & Ethics (10%)

### 4.1 Core Principles

| Principle | Description |
|-----------|------------|
| **Fairness** | No bias against protected groups |
| **Transparency** | Explainable decisions |
| **Privacy** | Data protection, differential privacy, federated learning |
| **Safety** | Guardrails against harmful outputs |
| **Accountability** | Clear ownership of AI decisions |
| **Robustness** | Resistant to adversarial attacks |

> **Mnemonic — "FTP-SAR"**
> **F**airness, **T**ransparency, **P**rivacy, **S**afety, **A**ccountability, **R**obustness
> *"FTP your SAR (Safety Assurance Report)"*

### 4.2 Bias & Mitigation

| Bias Type | Where It Occurs | Mitigation |
|-----------|----------------|-----------|
| **Data Bias** | Training data over/under-represents groups | Balanced datasets, data augmentation |
| **Algorithmic Bias** | Model learns biased patterns | Fairness constraints, bias audits |
| **Evaluation Bias** | Metrics don't capture fairness | Use disaggregated metrics per group |

### 4.3 NeMo Guardrails (Exam Favorite)

**What:** A toolkit for adding programmable constraints to LLM applications.

| Feature | Purpose |
|---------|---------|
| **Topical Guardrails** | Keep conversations on-topic |
| **Safety Guardrails** | Block harmful/toxic content |
| **Security Guardrails** | Prevent prompt injection, jailbreaks |
| **Fact-Checking Rails** | Verify outputs against knowledge base |
| **Colang** | NVIDIA's modeling language for defining conversational guardrails |

> **Mnemonic — "Guard TSSF with Colang"**
> **T**opical, **S**afety, **S**ecurity, **F**act-checking — written in **Colang**

---

## Quick-Reference Cheat Sheet

### Tool → Purpose (One-liner)

| Tool | Remember As |
|------|-----------|
| **NeMo** | "**Build** and fine-tune" |
| **Triton** | "**Serve** models at scale" |
| **TensorRT** | "**Optimize** for fast inference" |
| **RAPIDS/cuDF** | "**GPU DataFrames** (drop-in for Pandas)" |
| **cuML** | "**GPU Scikit-learn**" |
| **NGC** | "**Model/container catalog**" |
| **NeMo Guardrails** | "**Safety rails** for LLM apps" |
| **Colang** | "**Language** for writing guardrails" |
| **RLHF** | "**Align** model with human preferences" |
| **LoRA** | "**Efficient** fine-tuning with small matrices" |
| **RAG** | "**Retrieve** external knowledge to reduce hallucination" |

### Master Mnemonic: The NVIDIA AI Pipeline

> **"Build → Optimize → Serve → Accelerate Data"**
> **NeMo → TensorRT → Triton → RAPIDS**
> *"Bots Only Serve Awesome Data"*

---

## Exam Tips

1. **Time management:** ~1 minute per question. Don't overthink — flag and move on.
2. **NVIDIA tools make up 40%** — know what each tool does and when to use it.
3. **Transformer attention formula** will likely appear: `softmax(QK^T / √d_k) × V`
4. **cuDF vs. Pandas** — remember: same API, GPU-accelerated, that's the selling point.
5. **Triton dynamic batching** — the #1 feature for production inference scaling.
6. **NeMo Guardrails + Colang** — expect 2-3 questions on this.
7. **TensorRT optimizations** — layer fusion and precision reduction (FP16/INT8) are the key points.
8. **RLHF flow:** Pre-train → SFT → Reward Model → PPO fine-tuning.
9. **Encoder vs. Decoder:** BERT = understanding, GPT = generation. If the question is about classification → encoder. Generation → decoder.
10. **RAG vs. Fine-tuning:** RAG for current/external knowledge, fine-tuning for behavior/style changes.

---

## Mock Exam — Practice Questions & Answers

### Q1 — ML Foundations
**Which type of learning does GPT use during pre-training?**
A) Supervised Learning | B) Reinforcement Learning | C) Self-Supervised Learning | D) Unsupervised Learning
> **Answer: C** — GPT uses next-token prediction, creating its own labels from raw text. That's self-supervised.
> *Mnemonic: "SUSS-R" — GPT pre-trains on "Self"*

### Q2 — Transformer Architecture
**In `Attention(Q,K,V) = softmax(QK^T / √d_k) × V`, what is the purpose of dividing by `√d_k`?**
A) Normalize output values | B) Prevent large dot products pushing softmax into tiny-gradient regions | C) Reduce key dimensionality | D) Convert to probabilities
> **Answer: B** — Large dot products → near-one-hot softmax → vanishing gradients. Scaling keeps values stable.
> *Mnemonic: "Queens Keep Valuable Secrets" — without Scale, the Secret (gradient) vanishes*

### Q3 — NVIDIA Tools
**What is the primary purpose of NVIDIA Triton Inference Server?**
A) Fine-tuning LLMs with LoRA | B) Layer fusion and precision reduction | C) Serving ML models at scale with multi-framework support and dynamic batching | D) GPU-accelerated data preprocessing
> **Answer: C** — Triton = production serving. NeMo = fine-tuning, TensorRT = optimization, RAPIDS = data.
> *Mnemonic: "Triton Does Many Concurrent Ensembles"*

### Q4 — Neural Networks
**A model gets 98% train accuracy but 62% test accuracy. What is this, and what helps?**
A) Underfitting — increase complexity | B) Overfitting — apply dropout and regularization | C) Vanishing gradients — use ReLU | D) Overfitting — increase learning rate
> **Answer: B** — High train + low test = overfitting. Dropout & regularization help generalize.
> *Mnemonic: "DROP IT" — Dropout, Regularization, Out-of-sample, Prune less, Increase data, Terminate early*

### Q5 — LLM Architecture
**Which model type for text classification (sentiment analysis)?**
A) Decoder-only (GPT) | B) Encoder-only (BERT) | C) Encoder-Decoder (T5) | D) RNN (LSTM)
> **Answer: B** — BERT uses bidirectional attention = understands full context = ideal for classification.
> *Mnemonic: "BEG T5 to do both" — BERT=Encoder(understands), GPT=Decoder(generates), T5=Both*

### Q6 — NVIDIA Tools
**Which RAPIDS library is a GPU-accelerated drop-in replacement for Pandas?**
A) cuML | B) cuGraph | C) cuDF | D) Dask
> **Answer: C** — cuDF = CUDA DataFrames. Same API as Pandas, runs on GPU.
> *Mnemonic: "RAPIDS CUres Slow Data" — cuDF = cuDA DataFrames*

### Q7 — Fine-Tuning
**Adapt a 70B param LLM with limited GPU resources. Best method?**
A) Full fine-tuning | B) Train from scratch | C) LoRA | D) Increase context window
> **Answer: C** — LoRA freezes original weights, trains small matrices. Low resources = LoRA.
> *Mnemonic: "LoRA = Low Resources Allowed"*

### Q8 — TensorRT
**Which is NOT a TensorRT optimization?**
A) Layer fusion | B) Precision calibration | C) Dynamic batching | D) Kernel auto-tuning
> **Answer: C** — Dynamic batching is Triton (serving), not TensorRT (model optimization).
> *Mnemonic: "TensorRT optimizes the MODEL. Triton optimizes the SERVING."*

### Q9 — Trustworthy AI
**What is NeMo Guardrails, and what language defines its rules?**
A) Training framework; Python | B) Safety constraints toolkit; Colang | C) Inference optimizer; YAML | D) Bias detection; JSON
> **Answer: B** — NeMo Guardrails adds programmable safety rails. Rules written in Colang.
> *Mnemonic: "Guard TSSF with Colang" — Topical, Safety, Security, Fact-checking*

### Q10 — Attention Mechanism
**Multi-head attention: 8 heads, model dim 512. Dimension per head?**
A) 512 | B) 64 | C) 8 | D) 4096
> **Answer: B** — 512 ÷ 8 = 64. Each head gets an equal slice, outputs concatenated back.
> *Mnemonic: "Heads Divide, then Concatenate"*

### Q11 — NVIDIA Tools
**In NVIDIA's AI workflow, what is the correct order from training to production?**
A) Triton → TensorRT → NeMo | B) NeMo → TensorRT → Triton | C) TensorRT → NeMo → Triton | D) NeMo → Triton → TensorRT
> **Answer: B** — Build (NeMo) → Optimize (TensorRT) → Serve (Triton). Always optimize before serving.
> *Mnemonic: "Bots Only Serve Awesome Data" — Build → Optimize → Serve*

### Q12 — Self-Supervised Learning
**BERT's MLM is self-supervised rather than supervised because?**
A) Requires human annotators | B) Model generates its own labels by masking and predicting tokens | C) Uses RL with reward signal | D) Clusters sentences without labels
> **Answer: B** — The original tokens ARE the labels. GPT = next-token, BERT = masked-token. Both self-label.
> *Mnemonic: "Self = Self-labeling"*

### Q13 — Encoder vs Decoder
**Automated customer support chatbot that generates human-like responses. Which architecture?**
A) Encoder-only (BERT) | B) Decoder-only (GPT) | C) Encoder-Decoder (T5) | D) CNN
> **Answer: B** — Chatbot generates new text = Decoder (GPT). BERT = understand, T5 = transform text→text.
> *Mnemonic: "BERT Understands, GPT Generates, T5 Transforms"*

### Q14 — TensorRT vs Triton
**Reduce inference latency by converting FP32 to FP16. Which tool?**
A) Triton | B) NeMo | C) TensorRT | D) RAPIDS cuML
> **Answer: C** — Precision calibration is model optimization = TensorRT's job.
> *Mnemonic: "FLPKM" — Fusion, Lower precision, Profiling, Kernel tuning, Memory optimization*

### Q15 — RLHF
**Correct order of RLHF pipeline steps?**
A) Reward Model → Pre-training → SFT → PPO | B) Pre-training → SFT → Reward Model → PPO | C) SFT → Pre-training → PPO → Reward Model | D) Pre-training → PPO → Reward Model → SFT
> **Answer: B** — Pre-train → SFT → Reward Model → PPO. How ChatGPT was built.
> *Mnemonic: "Please Stop Reviewing Poorly" — Pre-train → SFT → Reward → PPO*

### Q16 — RAG vs Fine-Tuning
**Legal firm needs LLM to answer from private, frequently updated case law. Best approach?**
A) Full fine-tuning | B) LoRA | C) RAG | D) Increase temperature
> **Answer: C** — Private + frequently updated = RAG. Fine-tuning bakes static knowledge into weights.
> *Mnemonic: "RAG = Real-time Access to Grounded knowledge"*

### Q17 — Multi-GPU Training
**175B model can't fit on one GPU. Which strategy splits individual layers across GPUs?**
A) Data Parallelism | B) Tensor Parallelism | C) Pipeline Parallelism | D) Gradient Accumulation
> **Answer: B** — Tensor = split WITHIN a layer. Pipeline = split BETWEEN layers. Data = split the data.
> *Mnemonic: "Tensor = split the Tensor (inside layer)"*

### Q18 — Triton Inference Server
**Which Triton feature groups multiple incoming requests to maximize GPU utilization?**
A) Model ensemble | B) Concurrent model execution | C) Dynamic batching | D) Model versioning
> **Answer: C** — Dynamic batching groups requests on-the-fly. Triton's #1 production feature.
> *Mnemonic: "Triton Does Many Concurrent Ensembles" — D = Dynamic batching*

### Q19 — Tokenization
**GPT models use which tokenization method?**
A) Word-level | B) Character-level | C) BPE | D) WordPiece
> **Answer: C** — GPT = BPE (Byte-Pair Encoding). BERT = WordPiece. T5/LLaMA = SentencePiece.
> *Mnemonic: "Good Pairs Together = Byte Pair Encoding"*

### Q20 — NeMo Guardrails
**Banking chatbot needs to prevent off-topic discussions. Which guardrail?**
A) Safety Guardrails | B) Security Guardrails | C) Topical Guardrails | D) Fact-Checking Rails
> **Answer: C** — Staying on-topic = Topical Guardrails. Safety = block harm, Security = block attacks.
> *Mnemonic: "Guard TSSF with Colang" — Topical, Safety, Security, Fact-checking*

### Q21 — Activation Functions
**Most common activation function in the output layer of multi-class classification?**
A) ReLU | B) Sigmoid | C) Tanh | D) Softmax
> **Answer: D** — Softmax outputs probabilities summing to 1. Sigmoid = binary, ReLU = hidden layers.
> *Mnemonic: "Single class? Sigmoid. Multiple classes? SoftMax."*

### Q22 — NVIDIA Ecosystem
**Where to download pre-trained models, GPU-optimized containers, and Helm charts?**
A) RAPIDS | B) NGC | C) NeMo | D) CUDA Toolkit
> **Answer: B** — NGC = NVIDIA's Grand Catalog. One-stop shop for models, containers, charts.
> *Mnemonic: "NGC: NVIDIA's Grand Catalog"*

### Q23 — Encoder vs Decoder
**Extract disease names from medical records (NER). Which architecture?**
A) Decoder-only (GPT) | B) Encoder-only (BERT) | C) Encoder-Decoder (T5) | D) GAN
> **Answer: B** — NER = labeling/classifying tokens = understanding = BERT (Encoder).
> *Mnemonic: "BERT = Brain that Examines and Reads Text"*

### Q24 — Loss Functions
**Most appropriate loss function for binary classification?**
A) MSE | B) Binary Cross-Entropy | C) Categorical Cross-Entropy | D) MAE
> **Answer: B** — Cross-Entropy for Classification, MSE for Measurement (regression).
> *Mnemonic: "Cross for Class, MSE for Measure"*

### Q25 — TensorRT-LLM
**What distinguishes TensorRT-LLM from standard TensorRT?**
A) TensorRT-LLM is for training | B) Adds KV-cache, in-flight batching, tensor parallelism for LLMs | C) Only supports PyTorch | D) Replaces Triton
> **Answer: B** — TensorRT-LLM = TensorRT + KIT (KV-cache, In-flight batching, Tensor parallelism).
> *Mnemonic: "TensorRT-LLM = TensorRT + KIT"*

### Q26 — Vanishing Gradients
**Which architecture was designed to solve vanishing gradients in sequential models?**
A) CNN | B) Standard RNN | C) LSTM | D) Autoencoder
> **Answer: C** — LSTM uses three gates (Forget, Input, Output) to control gradient flow.
> *Mnemonic: "LSTM = Long-term Saving Through Memory gates" — FIO fixes the flow*

### Q27 — Prompt Engineering
**Developer provides 3 example input-output pairs before asking LLM to perform a task. What technique?**
A) Zero-shot | B) Few-shot | C) Chain-of-Thought | D) RAG
> **Answer: B** — Providing examples = few-shot. CoT = step-by-step reasoning, not examples.
> *Mnemonic: "Counting examples? Shot-counting. Showing reasoning? CoT."*

### Q28 — RAPIDS cuML
**Data scientist needs GPU-accelerated, API-compatible alternative to Scikit-learn's RandomForest?**
A) cuDF | B) cuGraph | C) cuML | D) Dask-cuDF
> **Answer: C** — cuML = GPU Scikit-learn. cuDF = Pandas. cuGraph = NetworkX.
> *Mnemonic: cuDF = DataFrames, cuML = Machine Learning, cuGraph = Graphs*

### Q29 — Positional Encoding
**Why do Transformers need positional encoding?**
A) Reduce dimensionality | B) Add word order since self-attention has no inherent position sense | C) Normalize attention weights | D) Prevent overfitting
> **Answer: B** — Self-attention treats input as a set. Positional encoding injects order.
> *Mnemonic: "No Position = No Order = No Meaning"*

### Q30 — Triton Model Repository
**Correct Triton model repository structure?**
A) `models/model_name/model.pt` | B) `model_repository/model_name/config.pbtxt` + `.../1/model.plan` | C) `triton/models/config.yaml` + `.../weights.bin` | D) `repository/model_name/latest/model.onnx`
> **Answer: B** — config.pbtxt (not yaml) + numbered version dirs (not "latest").
> *Mnemonic: "pbtxt, not yaml. Numbers, not names."*

### Q31 — Encoder vs Decoder
**Translate English to French. Which model?**
A) Encoder-only (BERT) | B) Decoder-only (GPT) | C) Encoder-Decoder (T5) | D) LSTM
> **Answer: C** — Translation = REWRITE (one text into another) = Encoder-Decoder (T5).
> *Mnemonic: "READ → BERT, WRITE → GPT, REWRITE → T5"*

### Q32 — Learning Rate
**What happens if learning rate is set too high?**
A) Converges slowly but accurately | B) Overshoots optimal weights and may diverge | C) Underfits | D) Gradients vanish
> **Answer: B** — Too high = big jumps = overshoot minimum = loss diverges.
> *Mnemonic: "High = Hyper jumps, Low = Lazy crawls"*

### Q33 — NeMo Framework
**Which is NOT a NeMo capability?**
A) Training LLMs from scratch | B) Fine-tuning with LoRA/P-Tuning | C) Real-time serving with dynamic batching | D) Data and tensor parallelism
> **Answer: C** — Serving = Triton. NeMo builds and trains, never serves.
> *Mnemonic: "NeMo Never Serves"*

### Q34 — Hallucination
**Most effective approach to reduce hallucinations for company-specific data?**
A) Increase temperature | B) Larger model | C) RAG | D) More attention heads
> **Answer: C** — RAG grounds responses in retrieved documents. Higher temp = more hallucination.
> *Mnemonic: "RAG = Real Answers, Grounded"*

### Q35 — Dropout
**During training, dropout sets 50% of neurons to zero. What happens at inference?**
A) Dropout continues | B) Dropout off, all neurons active, weights scaled | C) Only surviving neurons used | D) Dropout rate doubled
> **Answer: B** — Dropout is training-only. At inference, full capacity with scaled weights.
> *Mnemonic: "DROP at Train, FULL at Test"*

### Q36 — Model Parallelism
**Layers 1-20 on GPU1, 21-40 on GPU2, 41-60 on GPU3. What type of parallelism?**
A) Data | B) Tensor | C) Pipeline | D) ZeRO
> **Answer: C** — Groups of layers split across GPUs = Pipeline. Single layer split = Tensor.
> *Mnemonic: "Pipeline = split between layers (stages in a pipe)"*

### Q37 — Context Window
**50,000 token document into a 32,000 token context window. What happens?**
A) Model auto-summarizes | B) Input truncated or request fails | C) Multi-pass automatically | D) Extra tokens stored in memory
> **Answer: B** — Context window is a hard limit. No auto-summarization or memory.
> *Mnemonic: "Context Window = Hard Wall"*

### Q38 — Triton Ensemble
**Triton deployment needs preprocessing → inference → postprocessing in one request. Which feature?**
A) Dynamic batching | B) Model ensemble | C) Concurrent execution | D) Model versioning
> **Answer: B** — Model ensemble chains models into a single pipeline.
> *Mnemonic: "Ensemble = End-to-end pipeline"*

### Q39 — Federated Learning
**5 hospitals want to train AI without sharing patient data. Which approach?**
A) Transfer learning | B) Data parallelism | C) Federated learning | D) Data augmentation
> **Answer: C** — Federated learning trains locally, shares only model updates (gradients), never raw data.
> *Mnemonic: "Data stays home, only gradients travel"*

### Q40 — Encoder vs Decoder (Matching)
**Match: 1) Summarize article, 2) Generate Python code, 3) Detect spam**
A) 1: Enc-Dec, 2: Decoder, 3: Encoder | B) 1: Decoder, 2: Encoder, 3: Enc-Dec | C) 1: Encoder, 2: Decoder, 3: Enc-Dec | D) 1: Enc-Dec, 2: Encoder, 3: Decoder
> **Answer: A** — Summarize = REWRITE (T5), Generate code = WRITE (GPT), Detect spam = READ (BERT).
> *Mnemonic: "READ → BERT, WRITE → GPT, REWRITE → T5"*

---

### Progress Tracker — 40/50 Questions

| Domain | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| ML Foundations | 7 | 10 | 70% |
| NLP/Transformers | 8 | 13 | 62% |
| NVIDIA Tools | 10 | 12 | 83% |
| Trustworthy AI | 3 | 3 | 100% |
| Fine-Tuning/RLHF | 3 | 3 | 100% |
| **Overall** | **28** | **40** | **70%** |

---

