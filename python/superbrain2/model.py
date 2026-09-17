"""真实嵌入模型的安装渠道（脱离架构单独安装）。

模型以「GitHub Release 资产」放在开源仓库 superbrain-2.0 下，安装时直接从那里
下载——不把数百 MB 二进制塞进 git 历史（那会拖垮 clone / 撑爆仓库体积）。

用法：
    python -m superbrain2.model install-model          # 装默认 BGE 中文模型
    SUPERBRAIN_EMBEDDER_MODEL_DIR=/path python -m superbrain2.model install-model
    SUPERBRAIN_EMBEDDER_RELEASE=<tag> python -m superbrain2.model install-model
"""

from __future__ import annotations

import hashlib
import os
import sys
import tarfile
import tempfile
import urllib.request
from typing import Optional

# 模型分发元数据：tarball 里应含 model.onnx 与 tokenizer.json
_DEFAULT_MODEL = "bge-base-zh-v1.5"
_REPO = "kejixiaoqi666/superbrain-2.0"
_RELEASE_TAG = "models"                 # GitHub Release 标签（放模型资产）
# 资产命名：{model}-onnx-{arch}.tar.gz；此处为 Linux x86_64（也是大多数机房/服务器）
_ARCH = "linux-x64"
_SHA256_DEFAULT = ""                    # 旧版清空；正式发布模型时在此填写并固定

_REQUIRED = ("model.onnx", "tokenizer.json")


def _model_dir(model: str) -> str:
    return os.environ.get(
        "SUPERBRAIN_EMBEDDER_MODEL_DIR",
        os.path.join(os.path.expanduser("~"), ".superbrain", "models", model))


def _asset_url(model: str, arch: str, tag: str) -> str:
    return (f"https://github.com/{_REPO}/releases/download/"
            f"{tag}/{model}-onnx-{arch}.tar.gz")


def _asset_sha_url(model: str, arch: str, tag: str) -> str:
    return _asset_url(model, arch, tag) + ".sha256"


def _download(url: str, dest: str) -> None:
    sys.stdout.write(f"下载 {url}\n")
    sys.stdout.flush()
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        while True:
            block = r.read(1 << 16)
            if not block:
                break
            f.write(block)


def _verify(path: str, expected: Optional[str]) -> bool:
    if not expected:
        return True
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    got = h.hexdigest()
    if got != expected.lower():
        sys.stderr.write(f"sha256 校验失败：期望 {expected} 实得 {got}\n")
        return False
    return True


def install_model(model: str = _DEFAULT_MODEL, arch: str = _ARCH,
                  tag: str = _RELEASE_TAG, sha256: str = _SHA256_DEFAULT,
                  target: Optional[str] = None) -> str:
    """下载并安装模型到目标目录；已存在则跳过。返回模型目录绝对路径。"""
    target = target or _model_dir(model)
    target = os.path.abspath(os.path.expanduser(target))
    if all(os.path.exists(os.path.join(target, f)) for f in _REQUIRED):
        sys.stdout.write(f"已安装于 {target}，跳过下载。\n")
        return target

    os.makedirs(target, exist_ok=True)
    url = _asset_url(model, arch, tag)
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tf:
        tmp = tf.name
    try:
        _download(url, tmp)
        expected = sha256 or _sha_from_remote(model, arch, tag)
        if not _verify(tmp, expected):
            raise RuntimeError("模型包 sha256 校验失败，已中止（防篡改）。")
        with tarfile.open(tmp, "r:gz") as tar:
            tar.extractall(target, filter="data")
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass

    missing = [f for f in _REQUIRED
               if not os.path.exists(os.path.join(target, f))]
    if missing:
        raise RuntimeError(
            f"模型包不完整，缺 {missing}；目标目录 {target}")
    sys.stdout.write(f"✅ 模型已安装于 {target}\n")
    sys.stdout.write("接线：AgentConfig(embedder='bge') 或环境变量 SUPERBRAIN_EMBEDDER=bge\n")
    return target


def _sha_from_remote(model: str, arch: str, tag: str) -> Optional[str]:
    """尽力从远端拉取随附 sha256（存在则用，否则不校验）。"""
    try:
        with urllib.request.urlopen(_asset_sha_url(model, arch, tag), timeout=20) as r:
            return r.read().decode().strip()
    except Exception:
        return None


def main(argv: Optional[list] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    model = os.environ.get("SUPERBRAIN_EMBEDDER_MODEL", _DEFAULT_MODEL)
    arch = os.environ.get("SUPERBRAIN_EMBEDDER_ARCH", _ARCH)
    tag = os.environ.get("SUPERBRAIN_EMBEDDER_RELEASE", _RELEASE_TAG)
    target = os.environ.get("SUPERBRAIN_EMBEDDER_MODEL_DIR")

    if argv and argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if argv and argv[0] != "install-model":
        print(f"未知子命令: {argv[0]}", file=sys.stderr)
        print(__doc__)
        return 2

    try:
        install_model(model=model, arch=arch, tag=tag, target=target)
        return 0
    except Exception as e:  # 网络/校验失败给清晰错误
        sys.stderr.write(f"安装失败：{e}\n")
        sys.stderr.write("若发布资产尚未上传，先构建并上传模型到仓库 Release；本脚本只是拉取端。\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())