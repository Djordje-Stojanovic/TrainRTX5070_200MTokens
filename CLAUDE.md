# CLAUDE.md — Project Context for AI Agents

## MANDATORY RULES (read these FIRST, violating ANY is a critical bug)

1. **NEVER `git reset --hard`** — to discard, revert only train.py: `git checkout <commit> -- train.py`
2. **NEVER poll training** — first check after `sleep 240` (4 min), then `sleep 600` (10 min) between checks, max 7 checks per run
3. **ALWAYS follow the post-experiment checklist** (below) — no exceptions, no skipping steps
4. **ALWAYS push after every experiment** — `git push origin autoresearch/apr03`
5. **NEVER stop the loop** — run experiments forever until manually interrupted
6. **NEVER change fairness invariants** — TOKEN_BUDGET, MAX_SEQ_LEN, evaluate_bpb(), dataset/tokenizer
7. **ALWAYS deduplicate before experimenting** — Before writing ANY code for a new experiment, grep `results.tsv` for keywords related to your planned change. If it was already tried, DO NOT repeat it. Read the prior result's description to understand why it failed. Pick something genuinely new instead. Wasting ~50 minutes re-running a failed experiment is a critical bug.
8. **ALWAYS route EVERY experiment through `ideas.tsv`** — ZERO exceptions. You do NOT get to invent an experiment on the fly and run it. The flow is: (a) read ideas.tsv, (b) if you think of a new idea during analysis, WRITE IT INTO ideas.tsv FIRST with all 7 columns filled (id, status=pending, idea, category, impact, evidence, notes), (c) set the chosen row's status to `trying` BEFORE running, (d) delete that row after logging result to results.tsv. Bypassing the queue (even for "obviously good" fresh ideas) is a critical protocol violation. See program.md step 0 for the exact sequence.
9. **ALWAYS prove before skipping a queued idea** — You cannot dismiss a `pending` idea from ideas.tsv unless you can cite SPECIFIC evidence in results.tsv that rules it out (e.g., "ve-wd-09: skipped because 94d986b already tried VE WD and found it catastrophic"). "This looks low-impact to me" is NOT sufficient justification. If you truly think an idea is obsolete, DOCUMENT why in the commit message that deletes it AND cite the specific results.tsv row that proves it.
10. **STRICT FIFO QUEUE — APPEND-ONLY** — New ideas are ALWAYS added to the BOTTOM of ideas.tsv. NEVER insert at the top or middle. NEVER reorder by impact. The NEXT experiment is ALWAYS the OLDEST `pending` row at the TOP of the file. You may NOT run a newer idea before all older ideas are either completed (logged to results.tsv and deleted) or explicitly ruled out per rule 9. No cherry-picking, no "this new idea looks better". FIFO, period.

## Post-Experiment Checklist (execute EVERY time, in order)

```
# 1. Log result to results.tsv (even for crashes)
echo -e "<commit>\t<val_bpb>\t<mem_gb>\t<mfu>\t<tok_sec>\t<steps>\t<params_M>\t<batch>\t<final_loss>\t<status>\t<description>" >> results.tsv

# 2. Commit results
git add results.tsv && git commit -m "results: <status> <short description>"

# 3. Update chart
uv run plot_results.py --save
git add progress.png && git commit -m "chart: update progress.png"

# 4. If DISCARD: revert only train.py to pre-experiment state
git checkout <pre-experiment-commit> -- train.py
git commit -m "revert: undo <description>"

# 5. Update ideas.tsv: DELETE the row for the idea just tried
#    (the result is now in results.tsv — ideas.tsv is scratch only)
git add ideas.tsv && git commit -m "ideas: remove <id> (tried)"

# 6. Push everything
git push origin autoresearch/apr03
```

**Description column MUST be diagnostic.** Include: (1) what changed with values, (2) the hypothesis and evidence that motivated it, (3) what happened, (4) the conclusion — WHY it worked/failed and what this rules out. This is the AI's long-term memory. Future sessions read ONLY results.tsv, not git log.
- Bad: "WARMDOWN_RATIO 0.4->0.3"
- Bad: "reduce batch size for more steps"
- Good: "PARAM_X 0.4->0.3 + PARAM_Y constant (hypothesis: Y decay causes instability, evidence: exp#1 loss spike): loss still explodes 1.77->3.81, proves instability is NOT Y-related"
- Good: "COMPONENT_A changed from X->Y (hypothesis: [bottleneck] limits val_bpb, evidence: [metric or web search]): val_bpb 1.15->1.10, confirms [bottleneck] was the issue"

**Before proposing a new experiment, read results.tsv** to see what was already tried. Do not repeat a failed direction.

**MANDATORY DEDUPLICATION STEP:** Before writing ANY code, run `grep -i "keyword1\|keyword2\|keyword3" results.tsv` with keywords relevant to your planned experiment. If ANY prior experiment attempted something similar, READ its full description to understand why it failed. Only proceed if your approach is fundamentally different. Document what's different in your commit message.

## What is this?

Autonomous LLM pretraining research on a single RTX 5070 (12GB, Blackwell CC 12.0).
The AI agent runs experiments in a loop: modify code, train for 200M tokens, check if val_bpb improved, keep or discard, repeat.

**Key design principles (NEW in this project):**

1. **Token-based stopping (200M tokens)** — Training runs until exactly 200M tokens are processed, not a fixed wall-clock time. This ensures compute-optimal evaluation: architectures that are slower per step but more compute-efficient get properly rewarded.
2. **WSD schedule (Warmup-Stable-Decay)** — During experiment runs, the LR schedule uses only Warmup (5%) + Stable (95%). No decay phase. This measures the model's learning trajectory slope during pure exploration, not how well it settles into a local minimum. For a future long production run, decay would be added in the final phase.
3. **muP (Maximal Update Parameterization)** — Hyperparameters (LR, init) are scaled by model width relative to a reference width of 768. This mathematically guarantees that optimal hyperparameters found on the 200M model transfer to larger models without retuning.

Read **`program.md`** for the full experiment loop protocol, logging format, and operational rules.

## Architecture Overview

| Component | Current State |
|-----------|--------------|
| Model | SwiGLU MLP, partial RoPE, value embeddings, token shifting, d16 (768 dim, 6 heads), ~200M params (AI evolves this) |
| Dataset | ClimbMix (nvidia/Nemotron-ClimbMix), pre-tokenized with GPT-2 |
| Tokenizer | GPT-2 (vocab=50257), EOT token as BOS |
| Optimizer | Muon (matrices) + AdamW (embeddings, scalars) |
| Compile | torch.compile via triton-windows |
| Attention | SDPA with is_causal=True (FlashAttention fast path), QK-norm, 1/sqrt(d_head) scale |
| MFU | ~80-90% **relative to BF16 peak** (see MFU caveat below) |
| Token budget | 200M tokens per experiment (~1526 steps at batch 2^17) |
| LR schedule | WSD: 5% warmup, 95% stable (no decay) |
| Parameterization | muP (base width 768) |
| Metric | val_bpb (bits per byte) — lower is better |

## File Map

```
CLAUDE.md       — YOU ARE HERE. Project context for AI agents.
program.md      — Experiment loop protocol. READ THIS FIRST.
train.py        — Model, optimizer, training loop. PRIMARY EDIT TARGET.
prepare.py      — Data pipeline, tokenizer, evaluation, constants.
pyproject.toml  — Dependencies.
results.tsv     — Experiment log (created during runs).
ideas.tsv       — Scratch queue of untried ideas.
```

## MFU Measurement Caveat

**MFU is measured against BF16 peak FLOPS (~65.6 TFLOPS via runtime matmul benchmark), but training uses MXFP8 matmuls which have ~4x higher theoretical peak (246.9 TFLOPS on RTX 5070).** This means reported MFU values (80-90%) are inflated — real FP8 utilization is ~20-25%. We keep the BF16 benchmark for consistency across experiments. MFU is still useful as a **relative** metric between experiments (higher = better throughput), just not a meaningful absolute efficiency number. Do NOT change the benchmark — it would break cross-experiment comparisons.

## What You Can Change

**`train.py` — primary edit target (anything goes):**
- Model architecture (layers, dimensions, attention, MLP, embeddings)
- Optimizer (learning rates, schedules, weight decay, momentum)
- Hyperparameters (batch size, depth, aspect ratio, warmup)
- Training loop logic (gradient accumulation, loss scaling, etc.)
- muP base width is 768 — if you change model width, muP scales automatically. Do NOT manually adjust LR scaling.

**`prepare.py` — allowed but with constraints:**
- You may modify the dataloader for efficiency (packing, prefetching, etc.)
- You may adjust `EVAL_TOKENS` within reason (must remain enough for reliable BPB)
- You **MUST NOT** change `evaluate_bpb()` — it is the ground truth metric
- You **MUST NOT** change `MAX_SEQ_LEN` — it anchors comparison fairness
- You **MUST NOT** change `TOKEN_BUDGET` — it anchors comparison fairness
- You **MUST NOT** change the tokenizer or vocab for ClimbMix (GPT-2, 50257)

**`pyproject.toml` — only if you genuinely need a new dep:**
- Adding a package is allowed if it enables a real optimization (e.g., a fused kernel)
- Do not add packages speculatively

## What You Must Not Change

These are the **fairness invariants** that make experiments comparable:

1. **`TOKEN_BUDGET = 200_000_000`** (200M tokens) — the fixed training token count
2. **`MAX_SEQ_LEN = 2048`** — context length
3. **`evaluate_bpb()`** — the metric definition (nats per byte -> bits per byte)
4. **Dataset/tokenizer identity** — ClimbMix with GPT-2 tokenizer
5. **Evaluation data** — the val split must remain untouched
6. **URGENT — Minimum model size: ~200M params — DO NOT shrink the model.** Prior research (68 experiments at 20-min budget) proved that every attempt to reduce model size resulted in WORSE val_bpb. Increasing model size above ~200M is allowed. Decreasing below ~200M is FORBIDDEN.

## Hardware Constraints

- **GPU:** RTX 5070, 12GB VRAM, Blackwell CC 12.0
- **Peak VRAM target:** <11.5 GB (96% of 12GB)
- **Autotune:** Automatically finds best device_batch_size + checkpointing combo from candidates (16, 12, 8, 6, 5, 4, 3, 2), then **caches the result**. The cache key is GPU+PyTorch+seq_len — it does NOT include model size or TOTAL_BATCH_SIZE. So if you change model architecture/depth/width, the cached batch_size may be wrong.
- **After model size changes:** refresh autotune with `AUTORESEARCH_AUTOTUNE_REFRESH=1`
- **Autotune cache location:** `~\AppData\Local\autoresearch\gpu-profile-v3.json`
- If OOM at all batch sizes, reduce model size or enable more aggressive checkpointing

## Continuing a Run

If you are dropped into this repo on an `autoresearch/*` branch with results already in `results.tsv`, **you are resuming an existing experiment loop.** Do NOT re-run setup. Just:

1. Read `CLAUDE.md` and `program.md` for context.
2. Read `results.tsv` to see what's been tried and the current best val_bpb.
3. Read `train.py` and `prepare.py` for the current code state.
4. Continue the experiment loop from where it left off.

## Waiting for Training Runs (save context tokens)

Training takes ~55 min total (~50 min training + ~5 min startup/compile/eval).

**Protocol:**
1. Run training in background: `uv run train.py > run.log 2>&1` (use `run_in_background`)
2. **`sleep 240`** (4 min) — use bash `sleep`, with `timeout: 250000` — first check is early to catch startup crashes
3. Check: `grep "^val_bpb:" run.log 2>/dev/null || tail -1 run.log` — if crashed, you'll see the error immediately
4. If not done and no error, **`sleep 600`** (10 min) — use bash `sleep`, with `timeout: 610000`
5. Repeat step 3-4 until training finishes (~50 min). When eval starts, **`sleep 240`** (4 min) for eval to complete.
6. When done, extract all metrics with one grep.

Max ~7 checks per run (1x4min + ~5x10min + 1x4min = ~62min covers full run).

**MANDATORY: Status update on every check.** When you check `tail -1 run.log`, parse the step line and report to the user:
- Current smoothed loss and trend (improving/flat/worsening)
- MFU percentage
- Wall-clock seconds elapsed since start
- Estimated seconds/minutes remaining
- Tokens processed out of 200M (from the step number: `tokens = step * TOTAL_BATCH_SIZE`)
- Any anomalies (loss spikes, MFU drops, etc.)

Example: "Step 800/1526 (52.4%) | loss: 2.91 (stable) | MFU: 85.1% | elapsed: ~26min | remaining: ~24min | 105M/200M tokens"

## Bottleneck-First Rule

**Every experiment must target whatever is most limiting val_bpb.** Check your metrics (MFU, VRAM, training stability, loss curve) and decide what to improve.

**VRAM reporting:** `peak_vram_mb` in the training output uses `torch.cuda.mem_get_info()` which matches nvidia-smi. Use this value divided by 1024 for the `memory_gb` column in results.tsv.

**Hypothesis protocol (mandatory):** Every commit message must include: `Bottleneck: [X]. Hypothesis: [Y] because [Z]. Evidence: [prior experiment / web search / metric].`

## Experiment Prioritization (Explore vs Exploit)

Prioritize experiments by **expected impact x probability of success**:

| Category | Impact | Success Rate | When to Use |
|----------|--------|-------------|-------------|
| Architecture (new attention, MLP, embeddings) | High | Low (~20%) | Plateau or fresh session |
| Training dynamics (schedule, warmup) | Medium | Medium (~40%) | After architecture is stable |
| Hyperparameter tuning (LR, WD, batch) | Low | High (~60%) | Fine-tuning a working setup |
| Memory/throughput (batch sizing, checkpointing) | Indirect | High (~70%) | When VRAM underutilized or MFU < 50% |

**Plateau detection rule (MANDATORY):** Count the last 5 non-crash experiments. If none improved val_bpb by more than 0.005, you are in a plateau. When in a plateau:

1. STOP doing hyperparameter sweeps.
2. Do a focused web search session: 3-4 searches on architecture innovations at your parameter scale (100M-300M params, single GPU, 2025-2026).
3. Your next 2 experiments MUST be high-impact architecture changes.
4. Only return to hyperparameter tuning after an architecture change produces a new KEEP.

**Compound change rule (MANDATORY):** Never change more than one *category* per experiment.

**Don't run 3 consecutive experiments from the same category.**

## Web Search — Your Most Powerful Tool

Don't just search when stuck. **Search proactively.** The field changes monthly.

**When to search:**
- Before your first experiment each session
- Every 5th experiment: "landscape scan"
- After 2-3 failures in the same direction
- Before any major change to a component you haven't researched yet
- Whenever a metric plateaus

**Search areas** (rotate through ALL of these over time):
1. Architecture — attention variants, MLP designs, embeddings, normalization, positional encoding
2. Optimizer — algorithms, LR schedules, momentum, weight decay strategies
3. Memory efficiency — VRAM per parameter, activation checkpointing, mixed precision
4. Throughput / MFU — compute utilization, kernel fusion, batching strategies
5. Hardware-specific — Blackwell/CC 12.0, CUDA features, tensor core utilization
6. Training dynamics — loss stability, convergence speed, regularization, initialization
7. Scaling laws — optimal model size vs tokens vs compute
8. Data efficiency — learning more per token, curriculum strategies
9. Competitive benchmarks — open-source training speedruns, leaderboards
10. Frontier papers — latest arxiv from top labs

**Rules:**
- Always include "2026" in searches
- Never repeat the same search string across experiments
- Read arxiv papers and source code, not blog summaries
- Don't cargo-cult — understand WHY a technique works before implementing

## Context Window Management

Your context is finite. To maximize experiments per session:
- **Never read whole files** — use `grep -n "PATTERN" train.py` instead
- **Never cat run.log** — always use `grep` for specific metrics
- After 8+ experiments, re-read `results.tsv` to refresh what was tried
- Keep commit messages informative

## Tips for Good Experiments

- Make one change at a time when possible
- If val_bpb doesn't improve, revert (don't accumulate neutral changes)
- MFU matters: more compute per second = more learning per experiment
- Check VRAM usage — unused VRAM is wasted potential
- With 200M token budget, you get ~1526 optimizer steps — enough for real learning dynamics
- Simpler is better at equal performance (see program.md simplicity criterion)
- muP means you can safely experiment with different model widths — hyperparameters transfer automatically
- WSD (no decay) means you're measuring pure learning trajectory — improvements here predict long-run improvements
