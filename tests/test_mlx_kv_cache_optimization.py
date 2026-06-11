from __future__ import annotations

import mlx.core as mx
import numpy as np

from trendyol_tts_mlx.mlx_lm import MLXStaticKVCache
from trendyol_tts_mlx.mlx_minicpm import _position_id_to_int


def test_position_id_to_int_fast_path_for_python_int() -> None:
    assert _position_id_to_int(7) == 7


def test_forward_step_mx_exposes_evaluate_flag_for_combined_base_residual_sync() -> None:
    import inspect

    from trendyol_tts_mlx.mlx_lm import MiniCPMBaseLM

    sig = inspect.signature(MiniCPMBaseLM.forward_step_mx)

    assert "evaluate" in sig.parameters
    assert sig.parameters["evaluate"].default is True


def test_fill_caches_preserves_static_cache_capacity() -> None:
    cache = MLXStaticKVCache(num_layers=1, batch_size=1, num_key_value_heads=2, max_length=6, head_dim=3)
    prefill_key = mx.ones((1, 2, 3, 3), dtype=mx.float32) * 4
    prefill_value = mx.ones((1, 2, 3, 3), dtype=mx.float32) * 5

    cache.fill_caches([(prefill_key, prefill_value)])
    key_cache, value_cache = cache.get_layer_cache(0)

    assert key_cache.shape == (1, 2, 6, 3)
    assert value_cache.shape == (1, 2, 6, 3)
    assert cache.current_length == 3
    np.testing.assert_allclose(np.asarray(key_cache[:, :, :3, :]), 4.0)
    np.testing.assert_allclose(np.asarray(key_cache[:, :, 3:, :]), 0.0)
    np.testing.assert_allclose(np.asarray(value_cache[:, :, :3, :]), 5.0)
    np.testing.assert_allclose(np.asarray(value_cache[:, :, 3:, :]), 0.0)
