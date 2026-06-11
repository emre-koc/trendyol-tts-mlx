from __future__ import annotations

import numpy as np
import mlx.core as mx

from trendyol_tts_mlx.mlx_backend import _generated_feat_seq_to_np


def test_generated_feat_seq_to_np_concatenates_after_context_without_loop_growth() -> None:
    seq = [
        mx.ones((1, 1, 2, 3), dtype=mx.float32) * 1,
        mx.ones((1, 1, 2, 3), dtype=mx.float32) * 2,
        mx.ones((1, 1, 2, 3), dtype=mx.float32) * 3,
    ]

    out = _generated_feat_seq_to_np(seq, context_len=1)

    assert out.shape == (2, 2, 3)
    np.testing.assert_allclose(out[0], 2.0)
    np.testing.assert_allclose(out[1], 3.0)


def test_generated_feat_seq_to_np_handles_empty_after_context() -> None:
    seq = [mx.ones((1, 1, 2, 3), dtype=mx.float32)]

    out = _generated_feat_seq_to_np(seq, context_len=1)

    assert out.shape == (0,)
    assert out.dtype == np.float32
