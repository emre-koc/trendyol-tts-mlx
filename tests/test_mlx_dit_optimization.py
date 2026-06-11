from __future__ import annotations

import mlx.core as mx

from trendyol_tts_mlx.mlx_backend import GenerationProfileEvent, generation_stage_timer


class CountingEstimator:
    def __init__(self) -> None:
        self.prepare_calls = 0
        self.prepare_timestep_calls = 0
        self.call_precomputed_calls = 0
        self.call_calls = 0

    def prepare_static_inputs_mx(self, mu: mx.array, cond: mx.array):
        self.prepare_calls += 1
        return {"mu": mu, "cond": cond}

    def prepare_timestep_conditioning_mx(self, t: mx.array, dt: mx.array, dtype):
        self.prepare_timestep_calls += 1
        return t[:, None] + dt[:, None]

    def _call_with_static_inputs_mx(
        self,
        x: mx.array,
        t: mx.array,
        dt: mx.array,
        static_inputs,
        timestep_conditioning=None,
    ):
        self.call_precomputed_calls += 1
        assert timestep_conditioning is not None
        return mx.zeros_like(x)

    def _call_mx(self, x: mx.array, mu: mx.array, t: mx.array, cond: mx.array, dt: mx.array):
        self.call_calls += 1
        return mx.zeros_like(x)


def test_unified_cfm_precomputes_static_estimator_inputs_once_per_sample() -> None:
    from trendyol_tts_mlx.mlx_dit import MLXUnifiedCFM

    estimator = CountingEstimator()
    cfm = MLXUnifiedCFM(estimator=estimator, in_channels=2)
    mu = mx.zeros((1, 4), dtype=mx.float32)
    cond = mx.zeros((1, 2, 3), dtype=mx.float32)
    noise = mx.zeros((1, 2, 3), dtype=mx.float32)

    out = cfm._call_mx(
        mu=mu,
        cond=cond,
        n_timesteps=4,
        patch_size=3,
        noise=noise,
        use_cfg_zero_star=False,
    )

    assert out.shape == noise.shape
    assert estimator.prepare_calls == 1
    assert estimator.prepare_timestep_calls == 4
    assert estimator.call_precomputed_calls == 4
    assert estimator.call_calls == 0


def test_unified_cfm_records_sampler_setup_and_step_timing_stages() -> None:
    from trendyol_tts_mlx.mlx_dit import MLXUnifiedCFM

    events: list[GenerationProfileEvent] = []
    estimator = CountingEstimator()
    cfm = MLXUnifiedCFM(estimator=estimator, in_channels=2)
    cfm.set_profile_context(lambda stage, iteration=None: generation_stage_timer(events, stage, iteration=iteration))

    cfm._call_mx(
        mu=mx.zeros((1, 4), dtype=mx.float32),
        cond=mx.zeros((1, 2, 3), dtype=mx.float32),
        n_timesteps=2,
        patch_size=3,
        noise=mx.zeros((1, 2, 3), dtype=mx.float32),
        use_cfg_zero_star=False,
        profile_iteration=5,
    )

    assert [event.stage for event in events] == [
        "dit_cfm.t_span",
        "dit_cfm.static_inputs",
        "dit_cfm.timestep_conditioning",
        "dit_cfm.timestep_conditioning",
        "dit_cfm.estimator_step",
        "dit_cfm.euler_update",
        "dit_cfm.estimator_step",
        "dit_cfm.euler_update",
    ]
    assert all(event.iteration == 5 for event in events)
