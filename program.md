# autoresearch

This is an experiment to have the LLM do its own research.

> **Start here:** Read `CLAUDE.md` for project context, architecture overview, file map, and fairness rules.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr03`). The branch `autoresearch/apr03` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/apr03` from current main.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `CLAUDE.md` — project context, what you can/can't change, fairness rules.
   - `train.py` — model architecture (SwiGLU d16, ~200M params), optimizer (Muon+AdamW), training loop.
   - `prepare.py` — data pipeline (ClimbMix + GPT-2 tokenizer), dataloader, evaluation.
4. **Verify data exists**: Check that the cache directory contains ClimbMix shards and a tokenizer. If not, tell the human to run `uv run prepare.py --dataset climbmix`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single RTX 5070 (12GB). The training script runs for a **fixed token budget of 200M tokens** (~1526 steps at TOTAL_BATCH_SIZE=2^17, roughly ~50 minutes wall clock). You launch it simply as: `uv run train.py`.

**What you CAN change** (see `CLAUDE.md` for details):
- `train.py` — primary edit target. Architecture, optimizer, hyperparameters, training loop, batch size, model size. Everything is fair game.
- `prepare.py` — dataloader efficiency improvements are allowed. But do not touch the fairness invariants (see below).
- `pyproject.toml` — add a dependency only if it enables a real optimization.

**Fairness invariants** (DO NOT change these — they make experiments comparable):
- `TOKEN_BUDGET = 200_000_000` (200M tokens)
- `MAX_SEQ_LEN = 2048`
- `evaluate_bpb()` function definition
- Dataset identity (ClimbMix) and tokenizer (GPT-2, vocab 50257)
- Validation split data
- **URGENT — Minimum model size: ~200M params.** DO NOT shrink the model below ~200M params. Prior research (68 experiments at 20-min budget) definitively proved smaller models are worse. Increasing above ~200M is allowed. Decreasing is FORBIDDEN.

**The goal is simple: get the lowest val_bpb on ClimbMix.** Since the token budget is fixed, you don't need to worry about training volume — it's always 200M tokens. Everything else is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and processes 200M tokens.

**VRAM** is a hard constraint at 12GB. The autotune system handles batch sizing automatically. If your architecture change OOMs at all batch sizes, scale back.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

**Minimum improvement threshold** (two tiers):
- **Hard minimum 0.002**: A change can be kept at >=0.002 improvement IF it is lightweight (few lines, zero-parameter, simple) and you have confidence it's real signal not noise. Below 0.002 is always discard.
- **Soft target 0.003**: Novel, complex, or dramatic changes should aim for >=0.003 to justify their complexity.
- Changes that simplify the code can be kept even at noise-level improvement.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          0.997900
training_seconds: 2987.1
total_seconds:    3125.9
peak_vram_mb:     7000.5
eval_vram_mb:     4500.3
mfu_percent:      30.50
total_tokens_M:   200.0
num_steps:        1526
num_params_M:     201.9
depth:            16
dataset:          climbmix
```

`peak_vram_mb` is **real GPU VRAM** during training (measured via `torch.cuda.mem_get_info()`, matches nvidia-smi). Use this value divided by 1024 for the `memory_gb` column in results.tsv.

**MFU caveat:** `mfu_percent` is measured against **BF16 peak FLOPS** (~65.6 TFLOPS), but training uses MXFP8 which has ~4x higher theoretical peak. Reported MFU values (80-90%) are relative to BF16 only — useful for comparing experiments, not as absolute efficiency.

Note that the script stops after processing 200M tokens. You can extract the key metrics from the log file:

```
grep "^val_bpb:\|^peak_vram_mb:\|^mfu_percent:\|^total_tokens_M:\|^training_seconds:" run.log
```

## Monitoring during training

While training is running, check in every ~10 minutes. Report to yourself (in context) and the user:
- Current loss and trend
- MFU percentage
- Wall-clock seconds elapsed since start
- Estimated seconds remaining
- Tokens processed out of 200M

Use: `tail -1 run.log` to get the latest step line, which includes all these metrics.

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 11 columns:

```
commit	val_bpb	memory_gb	mfu	tok_per_sec	num_steps	num_params_M	batch_size	final_loss	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak training VRAM in GB, round to .1f (e.g. 7.0 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. mfu percent (e.g. 24.3) — GPU compute efficiency — use 0.0 for crashes
5. tok_per_sec — throughput (e.g. 37000) — use 0 for crashes
6. num_steps — optimizer steps completed — use 0 for crashes
7. num_params_M — model parameter count in millions (e.g. 201.9) — use 0.0 for crashes
8. batch_size — device batch size selected by autotune — use 0 for crashes
9. final_loss — training loss at last step (e.g. 1.850) — use 0.000 for crashes
10. status: `keep`, `discard`, or `crash`
11. short text description of what this experiment tried

Example:

```
commit	val_bpb	memory_gb	mfu	tok_per_sec	num_steps	num_params_M	batch_size	final_loss	status	description
a1b2c3d	0.950000	7.0	85.3	67000	1526	201.9	4	2.850	keep	baseline (SwiGLU d16 WSD+muP 200M tokens)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/apr03`).

LOOP FOREVER:

0. **Plan — STRICT QUEUE DISCIPLINE (see CLAUDE.md rules 8 & 9):**

   **The queue (`ideas.tsv`) is the ONLY entry point to an experiment.** You do NOT get to invent an experiment on the fly and run it. Inventing-then-running skips prioritization, bypasses user visibility, and destroys the research audit trail. This rule is absolute.

   Exact sequence — execute IN ORDER:

   **a. Read `results.tsv` AND `ideas.tsv`.** Re-read fully every 8+ experiments to refresh memory.

   **b. Triage `ideas.tsv`:** For each `pending` idea, confirm it hasn't been ruled out by `results.tsv`. If an idea IS ruled out, delete that row AND cite the specific results.tsv commit hash that proves it in the deletion commit message. You may NOT skip an idea because it "looks low-impact" — only because results.tsv specifically rules it out. See CLAUDE.md rule 9.

   **c. If during analysis you think of a NEW idea:** STOP. Before writing any code, APPEND the idea to the BOTTOM of `ideas.tsv` (never insert at top) as a new row with ALL 7 columns filled:
   - `id`: short-kebab-case-##
   - `status`: `pending`
   - `idea`: 1-line description with concrete values (e.g. "x0_lambdas init 0.2->0.4")
   - `category`: architecture | optimizer | training-dynamics | hyperparameter | memory-throughput
   - `impact`: low | medium | high
   - `evidence`: cite prior exp / web search / metric
   - `notes`: risks, expected params change, hypothesis

   **d. STRICT FIFO — NO SORTING.** `ideas.tsv` is append-only. New ideas go to the BOTTOM. The next experiment is ALWAYS the OLDEST `pending` row (topmost). Do NOT reorder by impact. Do NOT skip older ideas for newer ones. You may only skip an older idea by ruling it out per CLAUDE.md rule 9 (citing specific results.tsv evidence) and deleting it. No cherry-picking.

   **e. Set that row's `status` to `trying`** and commit `ideas.tsv` BEFORE running the experiment. This is your promise to the audit trail.

   **f. If `ideas.tsv` has zero viable pending ideas:** do a landscape scan (web search across 3-4 of CLAUDE.md's search areas), add 2-5 new ideas (check results.tsv first — never re-add a tried idea), then go to step d.

   **g. Every 5th experiment:** do a wide landscape scan across ALL CLAUDE.md search areas, add promising new ideas to `ideas.tsv`. Do NOT skip because "the queue has enough" — fresh research is required.

   **h. Write your commit message with:** `Bottleneck: [X]. Hypothesis: [Y] will improve because [Z]. Evidence: [results.tsv row / web search / metric]. [idea:id-from-ideas.tsv]`

   **i. When experiment finishes** (keep or discard), DELETE the row from `ideas.tsv` — the result now lives in `results.tsv`.

   **MANDATORY DEDUPLICATION CHECK (before EVERY experiment):**
   Before writing any code, grep `results.tsv` for keywords related to your planned change. If a similar experiment was already tried, DO NOT repeat it. Read the description of the prior attempt to understand WHY it failed, then either (a) pick a genuinely different experiment, or (b) document exactly what's different this time.
1. `git pull origin autoresearch/apr03` — pick up any doc updates pushed between experiments.
2. Make your experimental change (primarily `train.py`, but other files if needed per the rules in `CLAUDE.md`).
3. git commit (with the hypothesis from step 0 in the message)
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_bpb:\|^peak_vram_mb:\|^mfu_percent:\|^training_seconds:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't fix after a few attempts, give up.
7. Record the results in the tsv
8. If val_bpb improved (lower), you "advance" the branch, keeping the git commit
9. If val_bpb is equal or worse, you git reset back to where you started

**Timeout**: Each experiment should take ~55 minutes total (+ a few minutes for startup/compilation and eval overhead). If a run exceeds 75 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug), use your judgment: If it's easy to fix (typo, missing import), fix and re-run. If fundamentally broken, log "crash" and move on.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human might be asleep and expects you to continue working *indefinitely* until manually stopped. You are autonomous.
