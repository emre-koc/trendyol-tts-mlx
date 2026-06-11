# Proposed Optimizations

This document tracks optimization opportunities for the strict text-only pure-MLX Trendyol TTS backend.

## Current profile baseline

Latest profiled sample after the LM cached-step optimization:

```bash
python scripts/profile_mlx_runtime.py \
  --local-files-only \
  --text "Merhaba, MLX performans profilini çıkarıyoruz." \
  --seed 123 \
  --steps 16 \
  --max-len 80 \
  --json /tmp/trendyol_tts_profile/profile_lm_cached_step_opt.json \
  --cprofile /tmp/trendyol_tts_profile/profile_lm_cached_step_opt.prof
```

Profile summary:

- total wall time: `26.18s`
- model load: `6.12s`
- synthesis: `20.02s`
- generated audio: `3.2s`
- realtime factor: about `0.12x`

Baseline before this optimization (`profile_after.json`):

- total wall time: `31.44s`
- model load: `5.91s`
- synthesis: `25.27s`
- generated audio: `3.2s`
- realtime factor: about `0.10x`

Top hotspots after this optimization from `profile_lm_cached_step_opt.txt`:

1. `MiniCPMBaseLM.forward_step_mx`: about `15.5s`
   - 38 calls
   - down from about `19.3s`
   - still the main remaining bottleneck
2. load/startup path: about `6.1s`
   - dominated by `transformers` import/tokenizer overhead
   - `torch`, `sklearn`, and `pandas` appear through import side effects
3. AudioVAE / DiT-CFM are smaller than the cached LM step path in the current Python profile

## Completed optimization work

### Keep cached LM step outputs on MLX

Implemented:

- added `MiniCPMBaseLM.forward_step_mx(...)`
- strict MLX generation now keeps base-LM and residual-LM cached step outputs as `mx.array`
- avoids NumPy roundtrips inside each generated token step
- keeps explicit `mx.eval(...)` at the cached-step seam

Important finding:

- Removing NumPy roundtrips helped.
- Removing synchronization entirely caused MLX lazy-graph buildup and severe slowdown.
- Explicit `mx.eval(...)` is required at a stable seam.

Measured improvement on the profiled 16-step sample:

- original pre-MLX-step optimization: synthesis `32.75s`
- after keeping cached step outputs on MLX: synthesis `25.27s`
- after combining base-LM and residual-LM evaluation into one sync per generated token: synthesis `20.02s`

### Combine base-LM and residual-LM sync points

Implemented:

- `MiniCPMBaseLM.forward_step_mx(..., evaluate=True)` now exposes an explicit evaluation flag
- strict generation calls base LM with `evaluate=False`
- residual LM keeps `evaluate=True`, so the base-LM graph and residual-LM graph are evaluated together once per generated token
- `_position_id_to_int(...)` avoids a NumPy/MLX scalar roundtrip for Python integer position ids
- `MLXStaticKVCache.fill_caches(...)` now preserves static capacity with `mx.slice_update(...)` instead of constructing replacement arrays via concatenation

Validation:

- added `tests/test_mlx_kv_cache_optimization.py`
- base-LM and residual-LM forward-step parity tests still pass
- full suite: `54 passed, 8 warnings, 7 subtests passed`
- audio stats on the profiled sample unchanged: RMS `0.183459`, peak `0.983061`, audio seconds `3.2`

Artifacts:

- `/tmp/trendyol_tts_profile/profile_lm_cached_step_opt.json`
- `/tmp/trendyol_tts_profile/profile_lm_cached_step_opt.prof`
- `/tmp/trendyol_tts_profile/profile_lm_cached_step_opt.txt`
- `/tmp/trendyol_tts_bench_lm_opt/mlx_lm_opt.json`
- `/tmp/trendyol_tts_bench_lm_opt/mlx_lm_opt.csv`

### Avoid growing generated feature tensor per token

Implemented:

- removed per-token growth of a second `generated_feat` tensor
- generated feature patches are stored in the existing per-step list
- post-context feature sequence is concatenated once after the autoregressive loop

Relevant test:

- `tests/test_mlx_backend_generated_feat.py`

## Proposed next optimizations

## 1. Continue optimizing LM / residual-LM cached step path

Priority: **highest**

Why:

- `forward_step_mx` remains the dominant hotspot even after improving synthesis from `25.27s` to `20.02s`.
- Current cProfile still attributes about `15.5s` to cached LM stepping.
- Each generated audio patch runs base LM and residual LM cached stepping.

Completed in this pass:

- combined base/residual-LM sync into one `mx.eval(...)` per generated token
- avoided Python-int position-id NumPy roundtrip
- added tests around the evaluation flag and static cache fill behavior

Remaining possible work:

- inspect whether `mx.concatenate` cache updates inside `_forward_step_mx(...)` can be replaced without hurting latency; a quick `mx.slice_update(...)` experiment was not clearly better, so do not assume it helps without profiling
- avoid repeated `get_layer_cache(...)` / `set_layer_cache(...)` overhead
- pre-bind layer/cache handles in the generation loop
- specialize base-LM and residual-LM single-token paths further
- reduce wrapper calls inside `MiniCPMDecoderLayer._forward_step_mx(...)`
- precompute/reuse per-token RoPE position arrays across all base-LM layers
- add focused microbenchmarks for one cached step

Validation:

- preserve base-LM and residual-LM forward-step parity tests
- run full test suite
- profile same text/seed/steps/max-len before and after
- compare audio stats and generated sample length

Expected impact: **high**

## 2. Add finer-grained profiling inside generation

Status: **completed**

Implemented:

- added `GenerationProfileEvent`, `generation_stage_timer(...)`, `aggregate_generation_profile(...)`, and CSV export helpers in `mlx_backend.py`
- added opt-in strict-MLX generation profiling via `TrendyolMLXModel.enable_generation_profile(...)`
- added `scripts/profile_mlx_runtime.py` arguments:
  - `--internal-profile-json`
  - `--internal-profile-csv`
- profiler records prefill, per-token loop stages, final AudioVAE decode, and steady-state stats that exclude iteration `0` warmup/compile effects

Validation:

- added `tests/test_mlx_internal_profile.py`
- focused tests: `7 passed`
- full suite: `58 passed, 8 warnings, 7 subtests passed`
- real internal profile run produced:
  - `/tmp/trendyol_tts_profile/profile_internal_v2.json`
  - `/tmp/trendyol_tts_profile/profile_internal_v2.prof`
  - `/tmp/trendyol_tts_profile/profile_internal_v2.txt`
  - `/tmp/trendyol_tts_profile/internal_profile_v2.json`
  - `/tmp/trendyol_tts_profile/internal_profile_v2.csv`

Important measurement note:

- internal profiling forces `mx.eval(...)` at stage boundaries to attribute wall time to each stage
- this changes synchronization behavior and is intended for bottleneck diagnosis, not as the production latency benchmark

Key profile result from steady-state stages on the 16-step sample:

1. `dit_cfm_sample`: total `6.763449s`, mean `0.39785s`, count `17`
2. `base_lm_step`: total `1.958687s`, mean `0.122418s`, count `16`
3. `feat_encoder_step`: total `0.837585s`, mean `0.04927s`, count `17`
4. `audio_vae_decode`: total `0.656176s`, mean `0.656176s`, count `1`
5. `residual_lm_step`: total `0.511647s`, mean `0.031978s`, count `16`

Conclusion:

- previous Python cProfile attributed most synchronized work to `forward_step_mx`, but stage-synchronized profiling shows the DiT/CFM sampler is the dominant steady-state cost per generated patch
- the next optimization target should shift from cached LM stepping to DiT/CFM sampler internals, especially repeated timestep/conditioning work and estimator allocation/concat paths

## 3. Reduce cold-start / import overhead

Priority: **medium-high**

Why:

- load time is about `5.91s`
- cProfile shows several seconds in `transformers` import paths
- `torch`, `sklearn`, and `pandas` appear as import side effects even though strict `mlx_backend.py` does not import torch directly

Possible work:

- avoid `AutoTokenizer.from_pretrained(..., trust_remote_code=True)` if a lightweight tokenizer path can load the same tokenizer files directly
- create a strict-MLX tokenizer loader that avoids broad `transformers` auto-module imports
- move optional text normalization imports even later
- add a startup import audit script for strict MLX mode
- ensure CLI path does not import optional app/server dependencies

Validation:

- strict runtime check still passes
- tokenizer output parity for representative Turkish text
- cold-start benchmark before/after

Expected impact: **medium/high for CLI cold-start**, less important for long-running server mode

## 4. Tune MLX synchronization / graph boundaries

Priority: **medium**

Why:

MLX lazy evaluation can either help or hurt. The previous attempt showed that removing sync entirely made latency much worse.

Experiments:

- evaluate base and residual steps together vs separately
- evaluate only every N generated tokens
- evaluate after DiT/CFM but not after LM, or vice versa
- measure memory growth as well as latency
- compare `mx.eval(hidden)` vs evaluating a tuple of key intermediate arrays

Validation:

- benchmark same text/seed/steps/max-len
- track peak RSS
- verify audio stats unchanged

Expected impact: **medium**, but risky without careful profiling

## 5. Optimize AudioVAE decoder

Priority: **medium-low**

Why:

`AudioVAEDecoder.decode` takes about `2.14s`. This is not the main bottleneck but still relevant.

Possible work:

- profile individual decoder blocks
- reduce layout transposes
- inspect causal padding/cropping for repeated allocation
- precompute or cache sample-rate conditioning where safe
- test `decode_chunk_size` effects
- check whether any NumPy conversion happens earlier than necessary

Validation:

- preserve AudioVAE parity tests
- compare RMS/peak/clipping against prior samples
- run decode microbenchmark

Expected impact: **medium-low**

## 6. Optimize DiT / CFM sampler

Priority: **highest next target**

Why:

Stage-synchronized profiling shows `dit_cfm_sample` as the dominant steady-state per-patch cost:

- total `6.763449s`
- mean `0.39785s`
- count `17`

This is now larger than steady-state base LM stepping (`1.958687s` total, `0.122418s` mean), so the next optimization pass should focus here.

Completed in this pass:

- added `MLXVoxCPMLocDiT.prepare_static_inputs_mx(...)` so CFM solves project static CFG conditioning and Mu tokens once per generated patch rather than once per Euler step
- added `MLXVoxCPMLocDiT.prepare_timestep_conditioning_mx(...)` and passed precomputed timestep conditioning into estimator calls
- changed `MLXUnifiedCFM._solve_euler_mx(...)` to precompute static CFG inputs and timestep conditioning before the Euler loop
- added `tests/test_mlx_dit_optimization.py` to ensure static estimator inputs are prepared once per sample and reused by CFM estimator calls

Validation/results:

- focused test/parity: `tests/test_mlx_dit_optimization.py` and DiT estimator parity pass
- full suite: `59 passed, 8 warnings, 7 subtests passed`
- same 16-step sample stayed numerically/audio-stat stable: audio seconds `3.2`, RMS `0.183459`, peak `0.983061`
- production profile at steps=16: synthesis `19.517833s`, total wall `25.40604s`
- previous comparable profile after LM sync optimization: synthesis `20.023795s`, total wall `26.011725s`
- measured gain on that sample: about `0.51s` synthesis time (~2.5%)
- production profile at steps=24: synthesis `32.298043s`, total wall `38.654916s`, audio seconds `3.2`, RMS `0.164393`, peak `0.976027`
- production profile at steps=10 produced longer audio (`3.36s`), so latency is not directly comparable for speedup claims

Second pass completed:

- added opt-in nested timers inside `MLXUnifiedCFM`, `MLXVoxCPMLocDiT`, `MiniCPMDiTDecoder`, and `MiniCPMDecoderLayer`
- split DiT/CFM profile into sampler setup/update, static inputs, timestep embedding, estimator in/out projections, concat, decoder, per-layer attention, per-layer MLP, and decoder norm
- added `tests/test_mlx_dit_internal_timing.py` to verify nested decoder/layer timing events and iteration propagation
- extended `tests/test_mlx_dit_optimization.py` to verify sampler setup/step timing events
- added no-profile fast paths so instrumentation does not leave `contextlib.nullcontext()` overhead in the production DiT hot loop

Second-pass validation/results:

- focused timing/optimization/parity tests pass
- full suite: `62 passed, 8 warnings, 7 subtests passed`
- internal diagnostic profile artifacts:
  - `/tmp/trendyol_tts_profile/internal_dit_nested_layer_sync.json`
  - `/tmp/trendyol_tts_profile/internal_dit_nested_layer_sync.csv`
- steady-state diagnostic split for steps=16:
  - `dit_cfm_sample`: total `17.142941s`, mean `0.902260s`, count `19` (diagnostic sync-heavy timing)
  - `dit_cfm.estimator_step`: total `16.546479s`, mean `0.058058s`, count `285`
  - `dit_estimator.decoder`: total `16.096083s`, mean `0.056477s`, count `285`
  - summed decoder attention: `7.319654s`
  - summed decoder MLP: `7.760662s`
  - timestep embedding: `0.300896s`
  - in/out proj + concat together: about `0.409789s`
- attempted fused `mx.fast.scaled_dot_product_attention` for the dominant attention subpart, but production latency regressed badly (`31.063595s` synth at steps=16), so it was reverted
- attempted `mx.compile` around the repeated DiT estimator step behind an experimental flag; a single-step synthetic probe was only marginally faster (~2.7%), and the real synthesis profile with `--compile-dit` timed out after 600s, so the flag/code was removed rather than keeping a broken optimization
- production profile after instrumentation fast paths and fused-attention revert:
  - synthesis `19.606974s`
  - total wall `25.666149s`
  - audio seconds `3.2`, RMS `0.183459`, peak `0.983061`

Updated finding:

- the DiT estimator decoder is overwhelmingly dominant; attention and MLP are roughly equal, so there is no cheap projection/concat/timestep win left
- MLX fused SDPA is not beneficial for this small DiT sequence/shape in production; keep the manual attention path unless a future MLX release changes this
- the safest next performance lever is not another tiny setup micro-optimization; it is either compiling/fusing the repeated DiT decoder step, lowering diffusion steps via quality presets, or changing the DiT estimator architecture/precision after quality validation

Remaining possible work:

- experiment with `mx.compile` on the repeated DiT estimator/decoder call behind a flag and benchmark before keeping it
- inspect whether MLP gate/up projections can be fused at the Linear layer level without changing numerics
- consider quality/speed presets because raw DiT estimator cost scales strongly with `--steps`

Expected impact for remaining work: **medium, but higher risk than previous setup optimizations**

## 7. Add quality/speed presets

Priority: **high UX value, low implementation risk**

Rationale:

Low-level optimization will take time. Presets can immediately make the CLI easier and safer to use.

Potential presets:

- `--quality draft`: steps 8 or 10
- `--quality standard`: steps 16
- `--quality high`: steps 24

Possible behavior:

- preset sets `inference_timesteps`
- user-provided `--steps` overrides preset
- optionally choose max-len based on text length

Validation:

- CLI config tests
- one short synthesis smoke test per preset if runtime allows

Expected impact: **high UX value**

## Suggested next task order

1. Add finer-grained internal profile timers.
2. Microbenchmark and optimize cached LM step / KV cache update.
3. Re-run quality matrix to confirm output stability.
4. Investigate tokenizer/startup overhead.
5. Add quality presets.
6. Then consider AudioVAE and DiT/CFM micro-optimizations.

## Guardrails

- Do not remove `mx.eval(...)` blindly; previous profiling showed lazy graph buildup can drastically hurt performance.
- Do not trade away audio quality for latency without explicit quality validation.
- Keep strict MLX runtime free of upstream PyTorch VoxCPM model allocation.
- Continue rejecting voice cloning, prompt/reference audio, denoiser/enhancer, and streaming unless product scope changes.
- Run focused tests and full suite after changes.
