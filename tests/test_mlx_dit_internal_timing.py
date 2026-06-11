from __future__ import annotations

import mlx.core as mx

from trendyol_tts_mlx.mlx_backend import GenerationProfileEvent, generation_stage_timer


class IdentityNorm:
    def _call_mx(self, hidden):
        return hidden


class FakeAttention:
    def _call_mx(self, hidden, position_ids, is_causal=False):
        return hidden + 1.0, (hidden, hidden)


class FakeMLP:
    def _call_mx(self, hidden):
        return hidden + 1.0


def _recorder(events: list[GenerationProfileEvent]):
    return lambda stage, iteration=None: generation_stage_timer(events, stage, iteration=iteration)


def test_dit_decoder_layer_records_attention_and_mlp_timing_stages() -> None:
    from trendyol_tts_mlx.mlx_minicpm import MiniCPMDecoderLayer

    events: list[GenerationProfileEvent] = []
    layer = MiniCPMDecoderLayer(
        input_layernorm=IdentityNorm(),
        self_attn=FakeAttention(),
        post_attention_layernorm=IdentityNorm(),
        mlp=FakeMLP(),
        scale_depth=1.0,
        num_hidden_layers=1,
        use_mup=False,
    )
    layer.set_profile_context(_recorder(events), prefix="dit_decoder_layer_0")

    out, _ = layer._call_mx(mx.zeros((1, 2, 4), dtype=mx.float32), position_ids=mx.arange(2), is_causal=False)

    assert out.shape == (1, 2, 4)
    assert [event.stage for event in events] == [
        "dit_decoder_layer_0.attention",
        "dit_decoder_layer_0.mlp",
    ]


def test_mini_cpm_dit_decoder_records_layer_and_norm_timing_stages() -> None:
    from trendyol_tts_mlx.mlx_dit import MiniCPMDiTDecoder
    from trendyol_tts_mlx.mlx_minicpm import MiniCPMDecoderLayer

    events: list[GenerationProfileEvent] = []
    layer = MiniCPMDecoderLayer(
        input_layernorm=IdentityNorm(),
        self_attn=FakeAttention(),
        post_attention_layernorm=IdentityNorm(),
        mlp=FakeMLP(),
        scale_depth=1.0,
        num_hidden_layers=1,
        use_mup=False,
    )
    decoder = MiniCPMDiTDecoder(layers=[layer], norm=IdentityNorm())
    decoder.set_profile_context(_recorder(events), iteration=7)

    out = decoder._call_mx(mx.zeros((1, 2, 4), dtype=mx.float32), is_causal=False)

    assert out.shape == (1, 2, 4)
    assert [event.stage for event in events] == [
        "dit_decoder_layer_0.attention",
        "dit_decoder_layer_0.mlp",
        "dit_decoder_layer_0.total",
        "dit_decoder_norm",
    ]
    assert all(event.iteration == 7 for event in events)
