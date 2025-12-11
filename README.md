## TransHLA2.0

This is a modular pipeline for HLA–peptide binding prediction built on ESM + LoRA with optional cross‑attention and CNN branches. It provides separated modules for models, utilities, train/validation, and inference. Everything below is presented as one continuous Markdown block using “##” headings, suitable for direct saving as README.md. The default model name and checkpoint prefix are set to TransHLA2_0_BIND.

## Project Structure
project/  
├─ models.py          # Model definitions (TransHLA2_0_IM and ablation models)  
├─ utils.py           # Data loading/tokenization, samplers, loss and evaluation utilities  
├─ train_val.py       # Training + validation script (early stopping + best checkpoint saving)  
├─ infer.py           # Inference + testing script (metrics, plots, CSV export)  
├─ requirements.txt   # Dependency list  
├─ data/  
│  ├─ TransHLA_train_version_8_clean.txt  
│  ├─ TransHLA_val_version_8_clean.txt  
│  └─ TransHLA_test_version_8_clean.txt  
├─ checkpoints/       # Saved model weights  
└─ output/            # Plots and CSV outputs from inference

## Environment
Use Python ≥ 3.9. Install PyTorch first according to your CUDA/CPU setup (e.g., CUDA 12.x: `pip install torch --index-url https://download.pytorch.org/whl/cu121`), then install the rest via `pip install -r requirements.txt`. The requirements.txt should include: torch, transformers, peft, pandas, numpy, scikit-learn, tqdm, matplotlib, seaborn. On first run, pretrained assets are auto-downloaded from Hugging Face: Tokenizer = `facebook/esm2_t33_650M_UR50D`, Backbone = `facebook/esm2_t12_35M_UR50D`.

## Data Format
Input format is TSV (tab-separated, `sep="\t"`), with at least the following columns: `peptide` (epitope string, automatically converted to uppercase), `pseudosequence` (HLA pseudo-sequence; if missing, it may be 0 and will be replaced internally by `'<pad>' * 34`), and `label` (binary 0/1). Default paths: `data/TransHLA_train_version_8_clean.txt`, `data/TransHLA_val_version_8_clean.txt`, `data/TransHLA_test_version_8_clean.txt`. If your paths differ, specify them via script arguments.

## Components Overview
models.py: Includes the main model TransHLA2_0_IM (renamed from This_work), featuring dual LoRA-ESM encoders (epitope/hla), per-stream Transformer encoders, stacked bi-directional cross-attention, CNN branches, and an MLP classifier head. Also provides ablation models Lora_ESM, ThisWork, NoCNNModel, NoTransformerModel, NoCrossAttentionModel, plus `reinit_classifier` to reinitialize the classifier head. utils.py: Includes `load_and_tokenize` (TSV reading, normalization, tokenization, fixed-length padding/truncation), `pad_inner_lists_to_length`, `TCRDataset`, `addbatch`, `unbalanced_addbatch` (weighted sampler for class imbalance), `get_val_loss` (CrossEntropy with entropy regularization and smoothing), `get_loss`, `test_loader_eval` (Acc/AUC/MCC/F1/Recall/Precision and predictions), and `set_seed`. train_val.py: Trains and validates, applying early stopping based on validation AUC and saving the best checkpoint. infer.py: Loads checkpoints, evaluates on the test set, exports CSV results, and saves ROC/PR/Confusion Matrix plots.

## Quick Start
To train + validate (default model name = TransHLA2_0_BIND, using weighted sampler by default):  
`python train_val.py --model_name TransHLA2_0_BIND --epochs 100 --batch_size 32 --lr 1e-5 --save_dir checkpoints --save_prefix TransHLA2_0_BIND_best.pt`  
To train TransHLA2_0_IM model:  
`python train_val.py --model_name TransHLA2_0_IM --epochs 100 --batch_size 32 --lr 1e-5 --save_dir checkpoints --save_prefix TransHLA2_0_IM_best.pt`  
To use standard shuffle (no weighted sampler):  
`python train_val.py --model_name TransHLA2_0_BIND --balanced --epochs 100 --batch_size 32`  
To train other variants, change `--model_name` accordingly (e.g., `TransHLA2_0_IM`, `TransHLA2_0_BIND`, `NoCNN`, `NoTransformer`, `NoCrossAttention`).  
To run inference on the test set:  
`python infer.py --model_name TransHLA2_0_BIND --checkpoint checkpoints/TransHLA2_0_BIND_best.pt --output_dir output`  
To run inference with TransHLA2_0_IM:  
`python infer.py --model_name TransHLA2_0_IM --checkpoint checkpoints/TransHLA2_0_IM_best.pt --output_dir output`  
The console prints Acc, AUC, MCC, F1, Recall, and Precision; the `output/` directory includes `results_<test_filename>.csv`, `roc_curve.png`, `pr_curve.png`, and `confusion_matrix.png`.

## CLI Arguments
For train_val.py: `--train_path` path to train TSV (default `data/TransHLA_train_version_8_clean.txt`), `--val_path` path to val TSV (default `data/TransHLA_val_version_8_clean.txt`), `--test_path` path to test TSV (default `data/TransHLA_test_version_8_clean.txt`), `--model_name` default `TransHLA2_0_BIND` (alternatives: `{TransHLA2_0_IM, TransHLA2_0_BIND, NoCNN, NoTransformer, NoCrossAttention}`), `--epochs` (default 100), `--batch_size` (default 32), `--lr` (default 1e-5), `--weight_decay` (default 2.5e-3), `--patience` (early stop patience on validation AUC, default 5), `--balanced` (use standard shuffle instead of weighted sampler), `--save_dir` (default `checkpoints`), `--save_prefix` default `TransHLA2_0_BIND_best.pt`, `--device` `{cuda, cpu}` (auto if omitted). For infer.py: `--train_path`, `--val_path`, `--test_path` (optional for tokenizer consistency), `--model_name` default `TransHLA2_0_BIND` (alternatives: `{TransHLA2_0_IM, TransHLA2_0_BIND, NoCNN, NoTransformer, NoCrossAttention}`), `--checkpoint` (path to `.pt`), `--batch_size` (default 32), `--device`, `--output_dir` (default `output`).

## Training and Usage Tips
Loss: default is `nn.CrossEntropyLoss` augmented by `utils.get_val_loss`, which adds entropy regularization and the smoothing term `(loss - 0.04).abs() + 0.04`. Model outputs: current implementations return softmax probabilities; for the standard CE workflow with raw logits, remove the final softmax in the model and apply softmax only during evaluation. Tokenization and lengths: default peptide length is 16 and HLA length is 36; adjust via `utils.load_and_tokenize(pep_len=..., hla_len=...)`. Memory and performance: the TransHLA2_0_IM variant is heavier due to dual LoRA-ESM and cross‑attention; if memory is tight, try NoCNN or NoCrossAttention; consider mixed precision (torch.cuda.amp) and gradient accumulation. Class imbalance: weighted sampling is the default; use `--balanced` to switch to standard shuffle. Freezing and fine-tuning: to train only the classifier head initially, set `requires_grad=False` for all parameters except the classifier and call `reinit_classifier`; unfreeze later for full fine‑tuning.

## Outputs
CSV (output/results_<test_filename>.csv) includes `Pred_Prob_0`, `Pred_Prob_1` (class probabilities), `Pred_Label` (argmax), `True_Label` (ground truth), and retains the original test columns. Plots (in output/): `roc_curve.png` (with AUC), `pr_curve.png` (with AUPRC), and `confusion_matrix.png` (with percentage annotations).

## Reproducibility and Checkpoints
You can place weights (e.g., `TransHLA2_0_BIND_best.pt`) in `checkpoints/` and run inference via `python infer.py --model_name TransHLA2_0_BIND --checkpoint checkpoints/TransHLA2_0_BIND_best.pt --output_dir output`. For strict reproducibility, fix random seeds (see `utils.set_seed`) and record dependency versions and CLI arguments.

## FAQ
Slow first run: pretrained models and tokenizer are downloaded from Hugging Face; subsequent runs use cache. CUDA OOM: reduce `--batch_size`, switch to NoCNN/NoCrossAttention, enable mixed precision, or shorten `pep_len`/`hla_len`. AUC not improving or overfitting: increase `weight_decay`, lower `lr`, adjust `patience`, try a different model variant, or add data augmentation. Data parsing errors: ensure TSV format with columns `peptide`, `pseudosequence`, `label`; validate with `pandas.read_csv(path, sep="\t")`.

## Extensibility
Add a new model by implementing it in models.py and registering it in the model builders inside train_val.py and infer.py. Add metrics or visualizations by extending `utils.test_loader_eval` and plotting/export logic in infer.py. For mixed precision and multi‑GPU, integrate `torch.cuda.amp` and distributed training (`torch.distributed`) as needed.

## License
This project is provided for research purposes; include your preferred license (e.g., MIT, Apache-2.0).
