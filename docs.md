# Autoresearch Notes

## 2026-05-16

- Ran exactly one experiment from FIFO queue: `gdn-slayer-swap-07`.
- Experiment commit: `7a7d189`.
- Change: replaced late short-window attention layers 16 and 18 with a simplified diagonal GatedDeltaNet-style recurrent mixer while preserving the d20 residual shell, global L attention layers, value embeddings, learnable logit multiplier, WSD, muP, and the 200M-token budget.
- Result: discard. `val_bpb` worsened from current best `0.896775` to `0.901180`.
- Metrics: 8.8 GB peak VRAM, 72.3 MFU, 38,964 tok/sec average, 1,526 steps, 251.5M params, final loss 2.655.
- Conclusion: the simplified GDN-lite scan maintained train fit but hurt validation, so late S attention remains load-bearing. Future GDN work would need a more faithful chunk/kernel formulation or different placement, not this diagonal substitution.
- Current best remains `5b4910a`: learnable logit multiplier at `0.896775` val_bpb.
