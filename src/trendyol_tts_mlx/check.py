from __future__ import annotations

import platform
import sys


def main() -> int:
    print(f"python: {sys.version.split()[0]}")
    print(f"platform: {platform.platform()}")
    print(f"machine: {platform.machine()}")

    try:
        import mlx.core as mx
        print(f"mlx: ok, default_device={mx.default_device()}")
        x = mx.array([1.0, 2.0, 3.0])
        print(f"mlx smoke: {(x * 2).tolist()}")
    except Exception as e:
        print(f"mlx: FAILED: {e}")
        return 1

    try:
        import torch
        print(f"torch: {torch.__version__}")
        print(f"torch mps available: {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}")
    except Exception as e:
        print(f"torch: FAILED: {e}")
        return 1

    try:
        import voxcpm
        print(f"voxcpm: ok ({getattr(voxcpm, '__version__', 'unknown version')})")
    except Exception as e:
        print(f"voxcpm: FAILED: {e}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
