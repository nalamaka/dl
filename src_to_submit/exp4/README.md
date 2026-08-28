# Exp4: Transformer Neural Machine Translation (EN <-> ZH)

This experiment implements a `Transformer seq2seq` model in PyTorch for:

1. English-Chinese bidirectional translation (`zh2en` / `en2zh`)
2. BLEU4 evaluation

Default dataset path:
`E:\hw\deep_learning\data\sample-submission-version`

For the submission version in `src_to_submit/exp4`, the code resolves paths relative
to the project root. In other words, if your project root is:

`E:\hw\deep_learning`

then the expected data directory is:

`E:\hw\deep_learning\data\sample-submission-version`

If you move this submitted folder to another machine, keep the same relative layout:

```text
project_root/
├─ data/
│  └─ sample-submission-version/
└─ src_to_submit/
   └─ exp4/
```

## 1. Environment Setup

Recommended: Python `3.10` or `3.11`.

### Option A: `venv` (PowerShell)

```powershell
cd E:\hw\deep_learning
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r src_to_submit/exp4/requirements.txt
```

### Option B: `conda`

```powershell
cd E:\hw\deep_learning
conda create -n dl-exp4 python=3.11 -y
conda activate dl-exp4
python -m pip install --upgrade pip
pip install -r src_to_submit/exp4/requirements.txt
```

Notes:
- If you have NVIDIA GPU, install a CUDA-compatible PyTorch build first from official PyTorch instructions, then install the rest packages.
- If your terminal blocks script activation, run once:
  `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
- Optional tokenizer upgrade:
  `pip install sentencepiece`

## 2. Data Check

Ensure these files exist:

- `data/sample-submission-version/TM-training-set/chinese.txt`
- `data/sample-submission-version/TM-training-set/english.txt`
- `data/sample-submission-version/Test-set/Niu.test.txt`
- `data/sample-submission-version/Reference-for-evaluation/Niu.test.reference`

## 3. Run Commands

Run from project root `E:\hw\deep_learning`.

All commands below are for the submitted code under `src_to_submit/exp4`.

### Tokenizer Compatibility

- Legacy-compatible default:
  `--src_tokenizer legacy --tgt_tokenizer legacy`
- Optional SentencePiece (new tokenizer) while keeping old artifacts untouched:
  `--src_tokenizer spm --tgt_tokenizer spm --auto_train_spm`
  It auto-trains/loads:
  - `src/exp4/tokenizers/<direction>.src.spm.model`
  - `src/exp4/tokenizers/<direction>.tgt.spm.model`

### Train

```powershell
# Chinese -> English
python src_to_submit/exp4/main.py --mode train --direction zh2en

# English -> Chinese
python src_to_submit/exp4/main.py --mode train --direction en2zh

# save a step-checkpoint every 500 global steps
python src_to_submit/exp4/main.py --mode train --direction zh2en --save_every_steps 500

# print 3 preview samples after each epoch (default now is 0 for safety)
python src_to_submit/exp4/main.py --mode train --direction zh2en --preview_samples 3 --preview_max_decode_len 48

# basic training optimizations (AMP + grad accumulation)
python src_to_submit/exp4/main.py --mode train --direction zh2en --batch_size 8 --grad_accum_steps 4 --use_amp

# SentencePiece tokenizer training (compatible with old legacy checkpoints/vocabs)
python src_to_submit/exp4/main.py --mode train --direction zh2en --src_tokenizer spm --tgt_tokenizer spm --auto_train_spm

# log to jsonl (for visualization)
python src_to_submit/exp4/main.py --mode train --direction zh2en --run_name exp4_run1 --log_every_steps 10
```

Training logs and console output now display validation BLEU in the common `0-100`
scale. Checkpoint selection still uses the same underlying BLEU computation as before.

### Resume training (checkpoint)

```powershell
# auto: prefer *.last.pth, fallback to *.best.pth
python src_to_submit/exp4/main.py --mode train --direction zh2en --resume --resume_policy auto

# force resume from last checkpoint
python src_to_submit/exp4/main.py --mode train --direction zh2en --resume --resume_policy last

# resume from a specific checkpoint path
python src_to_submit/exp4/main.py --mode train --direction zh2en --resume --resume_ckpt src_to_submit/exp4/checkpoints/transformer_zh2en.last.pth
```

During training, checkpoints are saved to:
- `transformer_xxx.last.pth`: latest epoch state (for resume)
- `transformer_xxx.best.pth`: best BLEU4 state (for eval/inference)
- `transformer_xxx.pth`: backward-compatible alias of best
- `transformer_xxx.step.pth`: rolling step-checkpoint (updated every `--save_every_steps N`)

### Evaluate BLEU4 on validation split

```powershell
python src_to_submit/exp4/main.py --mode eval --direction zh2en

# safer low-memory eval
python src_to_submit/exp4/main.py --mode eval --direction zh2en --eval_batch_size 8 --eval_decode_len 64 --max_eval_samples 500 --clear_cuda_cache_every 10
```

Default eval profile is now aligned to quick small-scale testing:
- `--eval_batch_size 4`
- `--max_eval_samples 100`
- `--eval_decode_len 48`
- `--clear_cuda_cache_every 1`
- `--decode_strategy greedy`

### Evaluate BLEU4 on official test set

```powershell
python src_to_submit/exp4/main.py --mode test_eval --direction zh2en

# safer low-memory test eval
python src_to_submit/exp4/main.py --mode test_eval --direction zh2en --test_batch_size 8 --test_decode_len 64 --max_test_samples 300 --clear_cuda_cache_every 10
```

Default test profile is now aligned to quick small-scale testing:
- `--test_batch_size 4`
- `--max_test_samples 1000`
- `--test_decode_len 48`
- `--clear_cuda_cache_every 1`
- `--decode_strategy greedy`

Test/eval console output uses the common `0-100` BLEU scale. For example,
`24.62` means `24.62 BLEU`, not `0.2462`.

To improve translation quality at evaluation time, try beam search:
```powershell
python src_to_submit/exp4/main.py --mode test_eval --direction zh2en --resume_ckpt src_to_submit/exp4/checkpoints/transformer_zh2en.best.pth --decode_strategy beam --beam_size 4 --length_penalty 0.6 --no_repeat_ngram_size 3
```

For full evaluation/report, explicitly increase limits, e.g.:
```powershell
python src_to_submit/exp4/main.py --mode test_eval --direction zh2en --test_batch_size 8 --max_test_samples 1000 --test_decode_len 128 --clear_cuda_cache_every 10
```

By default, `test_eval` now prints about `10` evenly sampled cases with:
- source sentence
- reference translation
- model prediction

You can change or disable this via:
```powershell
# show 10 samples (default)
python src_to_submit/exp4/main.py --mode test_eval --direction zh2en --show_samples 10

# disable sample display
python src_to_submit/exp4/main.py --mode test_eval --direction zh2en --show_samples 0
```

### Fast Official Test Preview

If you only want to inspect a few official test examples without running full BLEU
evaluation, use the separate preview mode:

```powershell
# show about 10 evenly sampled official test cases
python src_to_submit/exp4/main.py --mode test_preview --direction zh2en --resume_ckpt src_to_submit/exp4/checkpoints/transformer_zh2en.best.pth --show_samples 10 --test_batch_size 8 --test_decode_len 128 --decode_strategy beam --beam_size 5 --length_penalty 0.8 --no_repeat_ngram_size 4
```

This mode:
- does not compute full test BLEU
- only decodes sampled official test sentences
- prints `SRC / REF / PRED`

### Small Grid Search (Decode Params)

Use small-sample grid search first, then run full evaluation with the selected params:

```powershell
python src_to_submit/exp4/grid_search_decode.py --resume_ckpt src_to_submit/exp4/checkpoints/transformer_zh2en.best.pth --direction zh2en --max_test_samples 100 --test_batch_size 4 --test_decode_len 48
```

The script will:
- run greedy + beam variants
- parse BLEU4 automatically
- print Top-K configs
- print one recommended full-eval command
- save detailed logs to `src_to_submit/exp4/decode_grid_results.jsonl`

### Training Log Visualization

Train logs are saved as JSONL in `src/exp4/logs` by default.

```powershell
python src_to_submit/exp4/visualize_results.py --log_file src_to_submit/exp4/logs/exp4_run1.jsonl
```

It generates:
- `step_loss.png`
- `lr_curve.png`
- `epoch_loss.png`
- `val_bleu4.png`

### Translate

```powershell
# single sentence
python src_to_submit/exp4/main.py --mode translate --direction zh2en --input_text "sample sentence ."

# from file
python src_to_submit/exp4/main.py --mode translate --direction zh2en --input_file data/sample-submission-version/Test-set/Niu.test.txt --output_file src_to_submit/exp4/pred_zh2en.txt
```

## 4. Quick Smoke Run

Use a small subset to verify environment quickly:

```powershell
python src_to_submit/exp4/main.py --mode train --direction zh2en --max_train_samples 256 --epochs 1 --batch_size 16
python src_to_submit/exp4/main.py --mode eval --direction zh2en --max_train_samples 256 --batch_size 16
```

## 5. Key Files

- `src_to_submit/exp4/config.py`: default paths and hyperparameters
- `src_to_submit/exp4/data_utils.py`: data loading, tokenization, vocabulary, dataloader
- `src_to_submit/exp4/model.py`: Transformer encoder-decoder model
- `src_to_submit/exp4/engine.py`: train/eval/translate/checkpoint logic
- `src_to_submit/exp4/evaluator.py`: BLEU4 implementation
- `src_to_submit/exp4/main.py`: CLI entrypoint

## 6. Submission Notes

Recommended submission content:

- `src_to_submit/exp4/*.py`
- `src_to_submit/exp4/README.md`
- `src_to_submit/exp4/requirements.txt`
- `src_to_submit/exp4/doc/report.md`
- optional: `src_to_submit/exp4/doc/report.pdf`

Usually **do not** submit the dataset itself. The external items needed to run this
submission are:

- dataset directory: `project_root/data/sample-submission-version`
- Python dependencies from `src_to_submit/exp4/requirements.txt`
- optional: trained checkpoints if the instructor requires direct reproduction of the final numbers

## 7. Training Optimizations Included

- `Warmup + Cosine` LR scheduler (default enabled):
  `--scheduler warmup_cosine --warmup_steps 1000 --min_lr_ratio 0.1`
- Decoder embedding/output weight tying (default enabled):
  `--tie_embeddings` (or disable via `--no_tie_embeddings`)
- Gradient accumulation:
  `--grad_accum_steps 4` means effective batch size = `batch_size * 4`
- AMP mixed precision (default enabled on CUDA):
  use `--no_amp` to disable
- Label smoothing (default `0.1`)
- Gradient clipping (default `1.0`)

## 8. Common Issues

- `FileNotFoundError` for vocab/checkpoint:
  run `--mode train` first to generate vocab and model checkpoint.
- `UnicodeEncodeError` in terminal:
  switch terminal encoding to UTF-8, or write outputs to `--output_file`.
- Very low BLEU at early epochs:
  expected for small training steps; increase `epochs` and keep full dataset.
- Eval/test freezes or OOM:
  reduce `--eval_batch_size`/`--test_batch_size`, reduce `--max_decode_len`, and use `--max_eval_samples` or `--max_test_samples`.
  The code now has CUDA OOM auto-retry by splitting batch.
