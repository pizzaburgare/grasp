# Performance Optimization Log

Benchmark: `.cache/bench/script/bench.py` (3 sections, 3 say() calls, high quality 1080p60)

| Commit | Description | Bench Time | Status |
|--------|-------------|------------|--------|
| 30e1793 | baseline (main, no changes) | 51.3s | — |
| d7e29db | pre-synthesize TTS audio before Manim render | 38.3s | KEPT |
| 33bd8a8 | hard links instead of file copies for cache | 38.6s | KEPT |
| f59448e | disable Manim progress bars + internal caching | 39.9s | REVERTED |
| eab1970 | torch.inference_mode for Qwen TTS | 37.2s | KEPT |
| 50ffd52 | batch TTS synthesis for pre-synthesized audio | 28.2s | KEPT |
| 599be77 | chunk batches to avoid GPU OOM (batch_size=4) | 35.2s (small) / 760.9s (full) | KEPT |
| a71c9a8 | increase batch_size from 4 to 8 | 29.1s (small) / 683.4s (full) | KEPT |
| bee0815 | lazy-import torch in QwenTTSEngine | 28.0s | KEPT |
| a6cbf63 | lazy-load TTS engine modules in registry | 26.3s | KEPT |

## Summary

- **Small benchmark (3 clips):** 51.3s → 26.3s (**49% faster**)
- **Full Erlang-B script (43 clips):** ~900s → ~683s (**~24% faster**)
- Manim render phase: ~51s → ~5s (only for cached re-renders)

### Key optimizations (by impact)

1. **Pre-synthesize audio before Manim render** — extract say() texts via AST, synthesize into cache before launching Manim subprocess. Manim then only hits cache.
2. **Batch TTS synthesis** — pass multiple texts to Qwen's generate_voice_clone() at once. batch_size=8 gives 2.4s/clip vs 7.3s/clip sequential (3x faster).
3. **Lazy imports** — defer torch and engine module imports until actually needed. Saves ~2s in the Manim subprocess.
4. **torch.inference_mode** — skip autograd overhead during TTS inference.
5. **Hard links for cache** — use os.link instead of shutil.copy2 for zero-copy cache serving.
