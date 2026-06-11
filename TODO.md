# TODO / Deferred MLX Work

This project intentionally focuses on a strict, text-only pure-MLX inference path for Mac/Apple Silicon.

## Explicitly skipped for now

The following features are **not implemented** in the strict MLX backend because they are outside the current product scope:

### AudioVAE encoder

Skipped because the current MLX backend does not support prompt/reference audio workflows.

The AudioVAE encoder would mainly be useful for:

- voice cloning
- reference speaker/audio conditioning
- prompt audio continuation
- audio reconstruction/editing workflows

Since strict MLX mode only supports text-to-speech generation from text input, the existing MLX AudioVAE decoder is sufficient for the current runtime path.

### Voice cloning / prompt audio / reference audio

Skipped by design.

Strict MLX mode should reject these inputs explicitly rather than falling back to PyTorch or silently producing partial behavior.

Expected behavior:

- `prompt_wav_path` / prompt audio: raise `NotImplementedError`
- `prompt_text` / reference text conditioning: raise `NotImplementedError`
- reference speaker/audio conditioning: raise `NotImplementedError`

### Denoiser / enhancer

Skipped by design.

The current goal is native MLX text-to-waveform synthesis, not post-processing enhancement. The denoiser/enhancer stack should not be ported unless output quality requires it or product scope changes.

Expected behavior in strict MLX mode:

- `denoise=True`: raise `NotImplementedError`
- enhancer-specific options: raise `NotImplementedError` or be unavailable

### Streaming

Skipped by design.

Streaming would require porting and validating CFM/DiT streaming state, partial decode behavior, buffering, and state carryover. Current strict MLX support is non-streaming batch synthesis.

Expected behavior:

- `streaming=True`: raise `NotImplementedError`

## Recommended next work instead

The following text-only MLX work has now been started and has reusable scripts/tests:

1. Broader quality validation
   - script: `scripts/validate_quality_matrix.py`
   - tests: `tests/test_quality_validation.py`
   - latest artifacts: `/tmp/trendyol_tts_quality/quality_results.json`, `/tmp/trendyol_tts_quality/quality_results.csv`, and generated WAVs
   - latest matrix covered 2 Turkish texts × steps 16/24 with seed 123 and max-len 80; all 4 MLX outputs had zero clipping

2. Performance profiling and optimization
   - script: `scripts/profile_mlx_runtime.py`
   - tests: `tests/test_profile_mlx_runtime.py`
   - optimization: strict MLX generation now keeps LM/residual-LM `forward_step` outputs on MLX with explicit `mx.eval(...)`, avoiding NumPy roundtrips while preventing lazy-graph buildup
   - generated feature patches are concatenated once after the autoregressive loop instead of growing a second tensor every token
   - latest profile artifacts: `/tmp/trendyol_tts_profile/profile_after.json`, `/tmp/trendyol_tts_profile/profile_after.prof`, `/tmp/trendyol_tts_profile/profile_after.txt`

3. Packaging and runtime cleanup
   - script: `scripts/check_strict_mlx_runtime.py`
   - tests: `tests/test_strict_mlx_runtime_check_script.py`
   - `SynthesisConfig.mlx_artifact_dir`, `MLX_ARTIFACT_DIR`, and CLI `--mlx-artifact-dir` allow large MLX artifacts to live outside the Python wheel/package
   - latest runtime-boundary check artifact: `/tmp/trendyol_tts_runtime_check.json`

Further useful follow-up:

- run the quality matrix against MPS for more text/seed/step combinations when longer benchmark time is acceptable
- add perceptual/listening notes beside the numeric RMS/peak/clipping metrics
- investigate deeper MLX kernel-level optimization for DiT/CFM and cached LM stepping

## Revisit only if scope changes

Only revisit AudioVAE encoder, voice cloning, denoiser/enhancer, or streaming if there is a clear product requirement for one of those feature families.
