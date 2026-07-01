# LLM Fine-Tuning & Alignment Pipeline

> Assignment 4 for a Natural Language Processing course (college coursework).

This project compares three strategies for adapting a language model to a
Python code-generation task, moving from full fine-tuning through
parameter-efficient fine-tuning to preference-based alignment.

| Part | Notebook | Technique | Model |
|------|----------|-----------|-------|
| I | `part1_fft_roberta.ipynb` | Full Fine-Tuning (FFT) | `FacebookAI/xlm-roberta-base` (adapted as a causal LM) |
| II | `part2_sft_qlora_qwen.ipynb` | Q-LoRA Supervised Fine-Tuning | `Qwen/Qwen2-1.5B-Instruct` |
| III | `part3_dpo_alignment.ipynb` | Direct Preference Optimization (DPO) | Merged SFT model from Part II |

## Overview

**Part I — Full Fine-Tuning baseline.** All parameters of XLM-RoBERTa are
updated (`is_decoder=True` makes attention causal). Trained on
`flytech/python-codes-25k`. This part shows that low held-out perplexity
under teacher forcing does **not** guarantee coherent free-running
generation — the model degenerates into repetition when generating
autoregressively, motivating a decoder-based model for Part II.

**Part II — Q-LoRA SFT.** Qwen2-1.5B-Instruct is loaded frozen in 4-bit NF4
precision (`BitsAndBytesConfig`), and rank-16 LoRA adapters
(`target_modules="all-linear"`) are trained on top, using
`completion_only_loss` so the loss only backpropagates through the
assistant's response tokens (system prompt and user query are masked). The
trained adapter is merged back into the base weights (`merge_and_unload()`)
to produce a standalone model, which becomes the reference/init model for
Part III. Training touches under 1% of total parameters.

**Part III — DPO alignment.** The merged SFT model from Part II is aligned
on `jondurbin/truthy-dpo-v0.1` using Direct Preference Optimization. A fresh
LoRA adapter is trained per run over a sweep of `β ∈ {0.1, 0.5, 0.8, 1.0}` to
study how the KL-penalty strength trades off alignment strength against
drift from the reference policy. Results are evaluated via reward accuracy
and reward margins, with qualitative before/after generation comparisons.

## Key results

- **FFT (Part I):** Low eval perplexity, but degenerate/incoherent
  autoregressive generation — a concrete illustration that perplexity alone
  is a poor proxy for generation quality, and that adapting a bidirectional
  encoder as a decoder has real limits.
- **Q-LoRA SFT (Part II):** Trains a small fraction of parameters yet
  produces a model that reliably writes functional Python — a large
  capability jump over the Part I baseline at a fraction of the compute/
  memory cost.
- **DPO (Part III):** Every tested β value successfully aligns the model,
  reaching reward accuracy of 1.0 on held-out preference pairs. β controls
  the magnitude of drift from the reference policy — the sweep quantifies
  the capability-vs-control trade-off between a purely compliant SFT model
  and a behaviorally-aligned one.

## Stack

`PyTorch` · Hugging Face `transformers`, `datasets`, `peft`, `trl` ·
`bitsandbytes` (4-bit NF4 quantization) · TensorBoard logging

## Repo structure

```
.
├── part1_fft_roberta.ipynb        # Part I: full fine-tuning baseline
├── part2_sft_qlora_qwen.ipynb     # Part II: Q-LoRA SFT
└── part3_dpo_alignment.ipynb      # Part III: DPO alignment + β sweep
```

## Running

Each notebook is self-contained and expects a CUDA GPU. Set the
`A4_SMOKE=1` environment variable to run a fast smoke test (reduced steps/
samples) before a full run. Part III depends on the merged model produced
by Part II (`results/part2_sft/merged`).
