from __future__ import annotations

from pathlib import Path

from scripts.check_strict_mlx_runtime import RuntimeCheck, inspect_mlx_runtime_source, summarize_checks


def test_inspect_mlx_runtime_source_rejects_torch_import_and_voxcpm_loader(tmp_path: Path) -> None:
    source = tmp_path / "mlx_backend.py"
    source.write_text(
        "import torch\n"
        "from voxcpm.core import VoxCPM\n"
        "def load():\n"
        "    return VoxCPM.from_pretrained('x')\n",
        encoding="utf-8",
    )

    checks = inspect_mlx_runtime_source(source)

    assert any(check.name == "no_torch_import" and not check.passed for check in checks)
    assert any(check.name == "no_voxcpm_from_pretrained" and not check.passed for check in checks)


def test_inspect_mlx_runtime_source_accepts_strict_native_runtime(tmp_path: Path) -> None:
    source = tmp_path / "mlx_backend.py"
    source.write_text(
        "import mlx.core as mx\n"
        "class TrendyolMLXModel:\n"
        "    def generate(self, text, prompt_wav_path=None, streaming=False, denoise=False):\n"
        "        if prompt_wav_path: raise NotImplementedError('prompt audio unsupported')\n"
        "        if streaming: raise NotImplementedError('streaming unsupported')\n"
        "        if denoise: raise NotImplementedError('denoiser unsupported')\n"
        "def f():\n"
        "    return mx.array([1])\n",
        encoding="utf-8",
    )

    checks = inspect_mlx_runtime_source(source)
    summary = summarize_checks(checks)

    assert all(check.passed for check in checks)
    assert summary["passed"] is True
    assert summary["total"] == len(checks)


def test_summarize_checks_lists_failed_names() -> None:
    summary = summarize_checks([
        RuntimeCheck(name="a", passed=True, detail="ok"),
        RuntimeCheck(name="b", passed=False, detail="bad"),
    ])

    assert summary["passed"] is False
    assert summary["failed"] == ["b"]
