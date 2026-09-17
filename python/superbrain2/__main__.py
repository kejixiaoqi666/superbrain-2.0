"""`python -m superbrain2` 入口：模型安装等维护命令。

    python -m superbrain2 install-model      # 下载真实 embedding 模型（脱离架构单独安装）
    python -m superbrain2 --help
"""
from __future__ import annotations

import sys


def main() -> int:
    from .model import main as _model_main
    return _model_main()


if __name__ == "__main__":
    sys.exit(main())