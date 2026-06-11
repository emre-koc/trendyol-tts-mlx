from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class RuntimeCheck:
    name: str
    passed: bool
    detail: str


def _all_imports(tree: ast.AST) -> set[str]:
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    return imports


def _has_call(tree: ast.AST, dotted_name: str) -> bool:
    parts = dotted_name.split(".")
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        attr_parts: list[str] = []
        current = node.func
        while isinstance(current, ast.Attribute):
            attr_parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            attr_parts.append(current.id)
        candidate = ".".join(reversed(attr_parts))
        if candidate.endswith(dotted_name) or candidate == dotted_name:
            return True
        if parts[-1] == "from_pretrained" and candidate.endswith("VoxCPM.from_pretrained"):
            return True
    return False


def inspect_mlx_runtime_source(path: Path) -> list[RuntimeCheck]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = _all_imports(tree)
    checks = [
        RuntimeCheck(
            name="no_torch_import",
            passed="torch" not in imports,
            detail="mlx_backend.py must not import torch in strict MLX runtime",
        ),
        RuntimeCheck(
            name="no_voxcpm_from_pretrained",
            passed=not _has_call(tree, "VoxCPM.from_pretrained"),
            detail="strict MLX runtime must not instantiate upstream VoxCPM.from_pretrained",
        ),
        RuntimeCheck(
            name="has_native_model_wrapper",
            passed="class TrendyolMLXModel" in source and "def generate" in source,
            detail="strict MLX runtime should own TrendyolMLXModel.generate",
        ),
        RuntimeCheck(
            name="explicit_unsupported_paths",
            passed="NotImplementedError" in source and "prompt" in source and "streaming" in source and "denoiser" in source,
            detail="prompt/reference audio, streaming, and denoiser paths should fail explicitly",
        ),
    ]
    return checks


def summarize_checks(checks: Iterable[RuntimeCheck]) -> dict[str, object]:
    checks_list = list(checks)
    failed = [check.name for check in checks_list if not check.passed]
    return {
        "passed": not failed,
        "total": len(checks_list),
        "failed": failed,
        "checks": [asdict(check) for check in checks_list],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check strict MLX runtime packaging/static boundaries")
    p.add_argument("--source", type=Path, default=Path("src/trendyol_tts_mlx/mlx_backend.py"))
    p.add_argument("--json", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    summary = summarize_checks(inspect_mlx_runtime_source(args.source))
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
