## TransHLA2.0 — Tutorial and README

A modular ESM-based pipeline for peptide–HLA binding prediction, offering:
- TransHLA2.0-PRE: standardization and preprocessing
- TransHLA2.0-BIND: minimal Hugging Face-compatible binding classifier
- TransHLA2.0-IM: research model with cross-attention/CNN branches
- Training scripts to train your own models

Status:
- Published on Hugging Face: TransHLA2.0-BIND (SkywalkerLu/TransHLA2.0-BIND)
- Local repo scripts: models.py, utils.py, train_val.py, infer.py
- Planned: PRE and IM checkpoints to be uploaded (examples below anticipate their usage)

## 1. Environment and Installation

Requirements:
- Python ≥ 3.8 (≥ 3.9 recommended for training scripts)
- torch ≥ 2.0
- transformers ≥ 4.40
- peft (only if you use LoRA/PEFT adapters)
- Optional (training scripts): pandas, numpy, scikit-learn, tqdm, matplotlib, seaborn

Install core libs:
- pip install torch transformers peft

Install full training stack (from repo root):
- pip install -r requirements.txt

Tip: install torch matching your CUDA first, e.g. CUDA 12.1:
- pip install torch --index-url https://download.pytorch.org/whl/cu121

## 2. Data Conventions (PRE)

TransHLA2.0 uses standardized inputs:
- peptide: amino-acid string (uppercased)
- pseudosequence: HLA pseudo-sequence (length typically 34–46; default padding target = 36)
- label: binary (0/1) for binding classification

Tokenization/lengths:
- Tokenizer: facebook/esm2_t33_650M_UR50D (for BIND examples)
- Fixed lengths (defaults): peptide = 16, HLA = 36
- Add special tokens from ESM tokenizer, then pad/truncate to fixed lengths
- Pad token id: tokenizer.pad_token_id (fallback to 1 if None)

File format (TSV):
- Required columns: peptide, pseudosequence (or hla_pseudo), label
- Example paths:
  - data/TransHLA_train_version_8_clean.txt
  - data/TransHLA_val_version_8_clean.txt
  - data/TransHLA_test_version_8_clean.txt

Note: PRE utilities will include standardized mapping from HLA allele to pseudosequence and validators for sequence alphabet/length when published.


## 3. Quick Start: TransHLA2.0-BIND (Hugging Face)

A minimal Hugging Face-compatible PyTorch model for peptide–HLA binding classification using ESM. Inference mirrors training:
1) tokenize peptide and HLA pseudosequence with ESM tokenizer
2) pad/truncate to fixed lengths (default peptide=16, HLA=36)
3) forward pass to get logits and features
4) apply softmax to obtain binding probability

Python snippet:

```python
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "SkywalkerLu/TransHLA2.0-BIND"
model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device).eval()

tok = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

peptide = "GILGFVFTL"  # 9-mer example
hla_pseudoseq = "YYSEYRNIYAQTDESNLYLSYDYYTWAERAYEWY"  # demo pseudosequence

PEP_LEN = 16
HLA_LEN = 36
PAD_ID = tok.pad_token_id if tok.pad_token_id is not None else 1

def pad_to_len(ids_list, target_len, pad_id):
    return (ids_list + [pad_id] * (target_len - len(ids_list))) if len(ids_list) < target_len else ids_list[:target_len]

pep_ids = tok(peptide, add_special_tokens=True)["input_ids"]
hla_ids = tok(hla_pseudoseq, add_special_tokens=True)["input_ids"]

pep_ids = pad_to_len(pep_ids, PEP_LEN, PAD_ID)
hla_ids = pad_to_len(hla_ids, HLA_LEN, PAD_ID)

pep_tensor = torch.tensor([pep_ids], dtype=torch.long, device=device)
hla_tensor = torch.tensor([hla_ids], dtype=torch.long, device=device)

with torch.no_grad():
    logits, features = model(pep_tensor, hla_tensor)  # logits: [1, 2]
    prob_bind = F.softmax(logits, dim=1)[0, 1].item()
    pred = int(prob_bind >= 0.5)

print({"peptide": peptide, "bind_prob": round(prob_bind, 6), "label": pred})

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "SkywalkerLu/TransHLA2.0-BIND"
model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device).eval()
tok = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

PEP_LEN = 16
HLA_LEN = 36
PAD_ID = tok.pad_token_id if tok.pad_token_id is not None else 1

def pad_to_len(ids_list, target_len, pad_id):
    return (ids_list + [pad_id] * (target_len - len(ids_list))) if len(ids_list) < target_len else ids_list[:target_len]

batch = [
    {"peptide": "GILGFVFTL", "hla_pseudo": "YYSEYRNIYAQTDESNLYLSYDYYTWAERAYEWY", "label": 1},
    {"peptide": "SIINFEKL", "hla_pseudo": "YYSEYRNIYAQTDESNLYLSYDYYTWAERAYEWY", "label": 0},
]

pep_ids_batch, hla_ids_batch = [], []
for item in batch:
    pep_ids = tok(item["peptide"], add_special_tokens=True)["input_ids"]
    hla_ids = tok(item["hla_pseudo"], add_special_tokens=True)["input_ids"]
    pep_ids_batch.append(pad_to_len(pep_ids, PEP_LEN, PAD_ID))
    hla_ids_batch.append(pad_to_len(hla_ids, HLA_LEN, PAD_ID))

pep_tensor = torch.tensor(pep_ids_batch, dtype=torch.long, device=device)  # [B, PEP_LEN]
hla_tensor = torch.tensor(hla_ids_batch, dtype=torch.long, device=device)  # [B, HLA_LEN]

with torch.no_grad():
    logits, _ = model(pep_tensor, hla_tensor)   # [B, 2]
    probs = F.softmax(logits, dim=1)[:, 1]

labels = (probs >= 0.5).long().tolist()

for i, item in enumerate(batch):
    print({"peptide": item["peptide"], "bind_prob": float(probs[i].item()), "label": labels[i]})


```
###Notes:

The model returns (logits, features). Apply softmax only at inference time to obtain probabilities.
Keep fixed PEP_LEN and HLA_LEN consistent with training.

```markdown
## 4. TransHLA2.0-PRE (Data Prep Utilities)

Purpose:
- Map HLA allele names to pseudosequences
- Validate inputs (AA alphabet, length checks)
- Standardize tokenization and fixed-length padding
- Export ready-to-train TSV files

Status:
- To be uploaded. Until then, follow Section 2 (Data Conventions) and replicate the tokenization/padding utilities shown in BIND Quick Start.

Expected usage (preview):

```python
from transhla2_pre import load_hla_map, to_pseudo, tokenize_pad

hla_map = load_hla_map("path/to/hla_map.tsv")  # allele -> pseudo
pseudo = to_pseudo("HLA-A*02:01", hla_map)
pep_ids, hla_ids = tokenize_pad(peptide="GILGFVFTL",
                                hla_pseudo=pseudo,
                                tokenizer_name="facebook/esm2_t33_650M_UR50D",
                                pep_len=16, hla_len=36)


```markdown
## 5. TransHLA2.0-IM (Research Model)

What it is:
- Dual LoRA-ESM encoders (peptide and HLA)
- Per-stream Transformer encoders
- Stacked bi-directional cross-attention
- Optional CNN branches
- MLP classifier head
- Ablations: Lora_ESM, NoCNN, NoTransformer, NoCrossAttention

Status:
- Training/inference scripts included locally (models.py, train_val.py, infer.py)
- Public HF checkpoint planned. API will mirror BIND with HF AutoModel and trust_remote_code=True.

Intended inference:

```python
from transformers import AutoModel, AutoTokenizer
import torch, torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "SkywalkerLu/TransHLA2.0-IM"  # replace once published
model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device).eval()
tok = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

# tokenize + pad as in Section 3
# logits, feats = model(pep_tensor, hla_tensor); prob = softmax(logits, dim=1)[:,1]

```markdown
## 6. Local Project Structure

project/
├─ models.py # Model definitions (TransHLA2_0_IM + ablations)
├─ utils.py # Data loading, tokenization, samplers, loss, metrics
├─ train_val.py # Training + validation (early stopping, best checkpoint)
├─ infer.py # Inference/testing (metrics, plots, CSV)
├─ requirements.txt # Dependencies
├─ data/
│ ├─ TransHLA_train_version_8_clean.txt
│ ├─ TransHLA_val_version_8_clean.txt
│ └─ TransHLA_test_version_8_clean.txt
├─ checkpoints/ # Saved model weights
└─ output/ # Plots and CSV outputs


On first run, pretrained assets are auto-downloaded:
- Tokenizer: facebook/esm2_t33_650M_UR50D
- Backbone (example for IM ablations): facebook/esm2_t12_35M_UR50D

## 7. Training Your Own Model (Local)

Run training + validation (default: TransHLA2_0_BIND):
```bash
python train_val.py \
  --model_name TransHLA2_0_BIND \
  --epochs 100 \
  --batch_size 32 \
  --lr 1e-5 \
  --save_dir checkpoints \
  --save_prefix TransHLA2_0_BIND_best.pt

```
python train_val.py \
  --model_name TransHLA2_0_IM \
  --epochs 100 \
  --batch_size 32 \
  --lr 1e-5 \
  --save_dir checkpoints \
  --save_prefix TransHLA2_0_IM_best.pt

Use standard shuffle (no weighted sampler):

```
python train_val.py --model_name TransHLA2_0_BIND --balanced --epochs 100 --batch_size 32


Key arguments (train_val.py):

--train_path, --val_path, --test_path: TSV paths
--model_name: {TransHLA2_0_IM, TransHLA2_0_BIND, NoCNN, NoTransformer, NoCrossAttention}
--epochs, --batch_size, --lr, --weight_decay
--patience: early stopping patience on validation AUC (default 5)
--balanced: switch to standard shuffle instead of weighted sampler
--save_dir, --save_prefix
--device: cuda or cpu (auto if omitted)
Tips:

Loss: CrossEntropy with optional entropy regularization and smoothing in utils.get_val_loss
Outputs: Prefer raw logits in training; apply softmax only at evaluation/inference
Fixed lengths: keep pep_len=16, hla_len=36 unless you retrain
Memory: try NoCNN/NoCrossAttention, mixed precision (torch.cuda.amp), gradient accumulation
Class imbalance: weighted sampler is default; use --balanced to disable
Freezing: start by training only the classifier, then unfreeze encoders later


```markdown
## 8. Inference and Evaluation (Local)

Run inference on the test set:
```bash
python infer.py \
  --model_name TransHLA2_0_BIND \
  --checkpoint checkpoints/TransHLA2_0_BIND_best.pt \
  --output_dir output

Run inference with IM:

```bash
python infer.py \
  --model_name TransHLA2_0_IM \
  --checkpoint checkpoints/TransHLA2_0_IM_best.pt \
  --output_dir output



Outputs:

Console: Acc, AUC, MCC, F1, Recall, Precision
CSV: output/results_<test_filename>.csv with Pred_Prob_0, Pred_Prob_1, Pred_Label, True_Label plus original columns
Plots in output/: roc_curve.png, pr_curve.png, confusion_matrix.png


```markdown
## 9. Reproducibility

- Fix random seeds via utils.set_seed
- Record dependency versions (pip freeze) and CLI arguments
- Place checkpoints (e.g., TransHLA2_0_BIND_best.pt) under checkpoints/ and reference via --checkpoint



