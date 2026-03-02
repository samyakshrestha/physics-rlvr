# Physics RLVR: Reasoning on MMLU-Pro Physics with GRPO

The goal was to specialize a language model for physics reasoning. Starting from **Qwen3-4B-Base** with no domain-specific post-training, a two-stage fine-tuning pipeline (SFT warmup + GRPO) was applied on MMLU-Pro physics questions, reaching **59.0% accuracy** on a 200-question held-out evaluation, a **+21pp gain** over the untuned base model.

---

## Results

Evaluated on 200 held-out MMLU-Pro physics questions with greedy decoding (`temperature=0`).

| Model | Accuracy | Format Compliance | Δ from Base |
|-------|----------|-------------------|-------------|
| Base (Qwen3-4B-Base, 5-shot) | 38.0% | N/A | - |
| SFT (0-shot XML) | 56.0% | 99.0% | +18.0pp |
| GRPO (0-shot XML) | **59.0%** | 98.0% | **+21.0pp** |

SFT delivers the headline gain (+18pp) by teaching the model the output schema and exposing it to physics-domain reasoning traces. GRPO adds a further +3pp by reinforcing correct answers via policy gradient.

---

## Pipeline

Three stages, four notebooks:

```
00_data_preparation  |  01_sft_warmup  |  02_grpo_training  |  03_evaluation
```

### Stage 1: Data Preparation (`00_data_preparation.ipynb`)

Source: `TIGER-Lab/MMLU-Pro`, `category="physics"`, ~1,299 questions, MIT license.

Split (seed=42, stratified by answer letter):

| Split | Size | Purpose |
|-------|------|---------|
| `sft_train.jsonl` | 80 | SFT warmup |
| `grpo_train.jsonl` | 1,000 | GRPO training |
| `eval.jsonl` | 200 | Held-out evaluation (never seen during training) |

Each row contains `question`, `options` (list of strings), `answer` (letter A-J), and `cot_content` (free chain-of-thought trace from the dataset, no teacher distillation required).

### Stage 2: SFT Warmup (`01_sft_warmup.ipynb`)

Supervised fine-tuning on 80 examples. The goal is to teach the model the XML output schema and warm up its physics reasoning before RL.

- **Model**: Qwen3-4B-Base loaded in full precision (no quantization)
- **Data**: `cot_content` field used as the reasoning trace, formatted into the XML schema
- **Loss masking**: `train_on_responses_only()` computes gradients on assistant tokens only
- **LoRA**: rank=32, alpha=64 (standard 2x ratio), targeting all attention and MLP projection layers
- **Output**: merged 16-bit weights + LoRA adapters saved separately

### Stage 3: GRPO Training (`02_grpo_training.ipynb`)

Policy gradient training on 1,000 physics MCQs using `GRPOTrainer` (TRL 0.22.2). Starts from the SFT LoRA checkpoint.

For each training prompt, the model generates G=8 rollouts. Per-rollout advantage is group-normalized:

```
advantage_i = (r_i - mean(r_1...r_G)) / std(r_1...r_G)
```

Groups where all G rollouts are correct or all wrong produce zero gradient (zero-advantage collapse). Only mixed-outcome groups generate a learning signal. This is a known limitation of binary rewards at small G.

---

## Output Schema

The same XML schema is enforced across SFT, GRPO, and evaluation:

```
<START_WORKING_OUT>
Step-by-step reasoning.
</END_WORKING_OUT>
<SOLUTION>
A
</SOLUTION>
```

`validate_schema()` checks for all four tags in correct order. `extract_solution()` finds the first valid A-J letter inside `<SOLUTION>...</SOLUTION>`, handling minor format variants (`(E)`, `E.`, etc.) produced during early training.

---

## Reward Design

Three additive reward functions, summed per rollout:

| Function | Range | Signal |
|----------|-------|--------|
| `format_reward_func` | 0.0 to 0.02 | +0.02 if all four XML tags present and ordered correctly |
| `reasoning_reward_func` | -0.1 to 0.10 | Linear 0 to 0.1 for 0-400 chars; plateau 400-800; -0.03/100 chars beyond 800, floor -0.1 |
| `correctness_reward_func` | 0.0 or 1.0 | 1.0 if predicted letter matches ground truth, else 0.0 |

**Correctness is binary.** MCQ has no partial credit. The GRPO group advantage provides the fractional gradient signal: a group of 8 rollouts with mixed outcomes produces nonzero advantage even from binary per-rollout rewards.

Correctness dominates the total reward. Format and reasoning rewards provide auxiliary shaping to suppress degenerate outputs (empty responses, runaway length).

---

## Training Configuration

### SFT

| Parameter | Value |
|-----------|-------|
| Learning rate | 2e-4 |
| LR scheduler | cosine |
| Max steps | 40 (~4.5 epochs over 80 examples) |
| Batch size | 2 |
| Gradient accumulation | 4 |
| Optimizer | adamw_8bit |
| LoRA rank / alpha | 32 / 64 |
| LoRA targets | q, k, v, o projections + gate, up, down MLP projections |
| Max sequence length | 3072 |

### GRPO

| Parameter | Value |
|-----------|-------|
| Learning rate | 5e-6 |
| LR scheduler | cosine + warmup_ratio=0.1 |
| Weight decay | 0.1 |
| Optimizer | adamw_8bit |
| num_generations (G) | 8 |
| Sampling temperature | 0.8 |
| Gradient accumulation | 4 (effective batch = 32) |
| Max completion length | 1024 tokens |
| Training epochs | 1 (~250 steps over 1,000 prompts) |
| Starting checkpoint | SFT LoRA adapters |

Training ran for ~54 minutes on a single NVIDIA RTX PRO 6000 Blackwell (97 GB VRAM).

---

## Evaluation Protocol

Each model is loaded with vLLM (`fast_inference=True`), generates completions for all 200 eval questions, then fully unloaded before the next model loads. vLLM holds a large contiguous GPU memory pool and two models cannot coexist on a 40 GB A100.

**Base model**: 5-shot continuation prompt (`Question: ... Answer: X` format). The base model was never trained on the XML schema, so a simpler format is used to give it a fair baseline.

**SFT / GRPO**: 0-shot ChatML prompt with the identical XML system prompt used during training.

All three conditions use greedy decoding (`temperature=0`). Residual run-to-run variance (~1-3pp) comes from GPU floating-point non-determinism (CUDA parallel op ordering, Flash Attention), not sampling noise.

Three metrics are reported:
- **Accuracy**: fraction of 200 questions answered correctly (primary signal)
- **Extraction rate**: fraction of completions where a valid A-J letter was found
- **Format compliance** (SFT/GRPO only): fraction of completions with all four XML tags in correct order

---

## File Structure

```
physics_rlvr/
├── notebooks/
│   ├── 00_data_preparation.ipynb   # Load MMLU-Pro physics, split, save jsonl files
│   ├── 01_sft_warmup.ipynb         # SFT on 80 examples with cot_content traces
│   ├── 02_grpo_training.ipynb      # GRPO on 1,000 examples (G=8, 1 epoch)
│   └── 03_evaluation.ipynb         # Eval base / SFT / GRPO on 200 held-out questions
├── data/
│   ├── sft_train.jsonl             # 80 SFT examples
│   ├── grpo_train.jsonl            # 1,000 GRPO examples
│   └── eval.jsonl                  # 200 held-out evaluation examples
├── models/                         # Saved model weights (not committed to git)
├── outputs/
│   ├── evaluation_results.json     # Aggregate accuracy / extraction / format metrics
│   ├── base_generations.jsonl      # Full completions for all 200 base-model outputs
│   ├── sft_generations.jsonl       # Full completions for all 200 SFT outputs
│   └── grpo_generations.jsonl      # Full completions for all 200 GRPO outputs
└── artifacts/
    └── grpo_run1/
        ├── train_log.csv           # Step-by-step training metrics
        ├── reward_total.png        # Total reward curve
        ├── reward_correctness_mean.png
        ├── kl.png                  # KL divergence from reference policy
        └── summary.json            # Final step, final reward, max correctness mean
```

---

## Stack

| Component | Version |
|-----------|---------|
| PyTorch | 2.8.0 |
| Transformers | 4.56.2 |
| Unsloth / unsloth-zoo | 2026.1.2 |
| TRL | 0.22.2 |
| vLLM | 0.10.2 |
| Dataset | TIGER-Lab/MMLU-Pro (MIT) |
| Experiment tracking | WandB (`physics-rlvr` project) |

**Note on version pins**: TRL 0.22.2 passes `completions` to reward functions as raw strings (newer versions pass dicts). unsloth-zoo 2026.1.2 is pinned to avoid a `masked_batch_mean` regression introduced in Feb 2026.

---

## Author

Samyak Shrestha
