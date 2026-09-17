"""嵌入层：文本 → 语义向量 + 嵌入器一致性握手（MemoryBridge 精髓）。

默认零依赖哈希嵌入（跨进程/跨平台结果一致）；可换真实 embedding 模型，
但跨设备同步前必须用 embedder_identity 指纹握手（同模型同维度同 revision
才允许比对向量）。
"""

from __future__ import annotations

import hashlib
import math
import os
from typing import List, Optional, Protocol


def cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    # 内联点积（避免生成器函数调用开销）
    dot = 0.0
    for i in range(len(a)):
        dot += a[i] * b[i]
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def dot_normalized(a: List[float], b: List[float]) -> float:
    """归一化向量的点积（已知 |a|=|b|=1 时等价 cosine，省两次开方）。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    s = 0.0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s


def embedder_identity(emb) -> dict:
    """嵌入器自描述指纹：跨设备同步前用它做一致性握手。"""
    name = getattr(emb, "model", None) or f"hashing-{getattr(emb, 'dim', 0)}"
    dim = int(getattr(emb, "dim", 0) or 0)
    revision = str(getattr(emb, "revision", "") or "")
    basis = f"{type(emb).__name__}:{name}:{dim}" + (f":{revision}" if revision else "")
    fp = hashlib.blake2b(basis.encode("utf-8"), digest_size=8).hexdigest()
    return {"type": type(emb).__name__, "name": name, "dim": dim,
            "revision": revision, "fp": fp}


class Embedder(Protocol):
    def embed(self, text: str) -> List[float]:  # pragma: no cover
        ...


class HashingEmbedder:
    """字符 n-gram 特征哈希嵌入：零依赖、跨进程/平台结果一致。

    关键：用 blake2b 哈希而非 Python 内置 hash()（后者有进程级随机化，
    同一文本不同进程嵌入不同，会导致记忆检索失效）。

    增强（可选，sync 时用增强词汇归一）：同义词归一到原型 token 再哈希，
    能在零依赖下突破部分"零字面重叠的同义词"天花板（如 卡顿↔不稳定）。
    strip_stop=True 时去掉中文高频虚词，减干扰。默认关闭以保持旧库向量一致。
    """

    _SYNONYMS = {
        "卡顿": ["卡顿", "不稳定", "很卡", "很慢", "慢", "延迟高", "卡死"],
        "查询": ["查", "查询", "查看", "查一下", "看看", "检索", "搜索"],
        "重启": ["重启", "重新启动", "重起", "reboot", "重启一下"],
        "启用": ["启用", "开通", "开启", "打开", "开启新"],
        "节点": ["节点", "服务器", "机器", "服务节点", "服务端"],
        "连接": ["连", "连接", "联网", "连上", "接入"],
        "失败": ["失败", "没成功", "上不去", "连不上", "不行", "挂了",
                 "出问题", "故障", "报错", "异常"],
        "退款": ["退款", "退单", "退回", "退钱", "撤销订单", "取消订单",
                "撤销购买", "退费"],
        "路由": ["路由", "线路", "转发规则", "转发", "代理规则", "新线路",
                "新节点", "通道", "入口"],
        "状态": ["状态", "情况", "状况", "运行情况", "health", "运行状态",
                "运行状况"],
    }
    _STOPWORDS = set("的了在是和你我他与它都也还不就从很能会把被对等但并且因为所以于是着过或又及既且然而")

    def __init__(self, dim: int = 256, ngram: int = 2,
                 enhance: bool = False, strip_stop: bool = False) -> None:
        self.dim = dim
        self.ngram = ngram
        self.enhance = enhance      # 默认 False=兼容旧库；True 用同义归一词典
        self.strip_stop = strip_stop  # 去掉中文高频虚词
        self.model = "hashing"
        self.revision = "v2-enhanced" if enhance else "v1"
        if enhance:
            self._fwd = {f: p for p, forms in self._SYNONYMS.items()
                         for f in forms}

    def _normalize(self, text: str) -> str:
        t = text.lower()
        if self.enhance:
            for f, p in self._fwd.items():
                t = t.replace(f, p)
        if self.strip_stop:
            t = "".join(ch for ch in t if ch not in self._STOPWORDS)
        return t

    def _grams(self, text: str) -> List[str]:
        t = "".join(self._normalize(text).split())
        if not t:
            return []
        if len(t) <= self.ngram:
            return [t]
        return [t[i:i + self.ngram] for i in range(len(t) - self.ngram + 1)]

    def embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        for g in self._grams(text):
            h = int.from_bytes(hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest(), "big")
            vec[h % self.dim] += 1.0 if (h >> 32) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


# ---------------------------------------------------------------------------
# 真实 embedding 模型（可插拔）—— 脱离架构单独安装
# ---------------------------------------------------------------------------
# 设计原则：
#   * HashingEmbedder 仍是零依赖默认，真实模型是「装了才生效」的可选组件，
#     因此 onnxruntime / tokenizers 等重依赖不在此文件顶层 import（懒加载），
#     缺依赖 / 缺模型时给出安装渠道提示而非让整个内核 import 失败。
#   * 藏着「语义漂移」守卫：嵌入器身份(指纹)一旦变化，已有记忆向量即不可比，
#     存储层必须用 embedder_identity 握手 + reembed_all 迁移修复，绝不静默混算。


class EmbeddingInstallRequired(RuntimeError):
    """真实嵌入模型或其运行依赖缺失。携带清晰的安装指引。"""

    def __init__(self, hint: str, *args) -> None:
        self.hint = hint
        super().__init__(hint + "\n" + HELP_TEXT)


HELP_TEXT = (
    "\n安装渠道（脱离架构单独安装，默认哈希嵌入不受影响）：\n"
    "  1) 装运行依赖：pip install onnxruntime tokenizers\n"
    "  2) 下载模型：python -m superbrain2 install-model   # 从开源仓库 GitHub Release 拉取\n"
    "     # 或指定：SUPERBRAIN_EMBEDDER_MODEL_DIR=/path python -m superbrain2 install-model\n"
    "  3) 接线使用：EmbedderConfig(embedder='bge') 或环境变量 SUPERBRAIN_EMBEDDER=bge\n"
    "完整说明见 docs/EMBEDDER.md"
)


class OnnxEmbedder:
    """ONNX Runtime 中文语义嵌入器（BGE 系，默认 bge-base-zh-v1.5，768 维，~380MB）。

    懒加载：构造不碰任何重依赖；首次 embed 时才 import onnxruntime/tokenizers 并
    加载模型文件。`.model`/.`dim`/.`revision`/.`normalized` 供 embedder_identity 指纹握手。

    BGE 官方 pooling：sentence_embedding = last_hidden_state[:, 0]（[CLS] 位），
    然后 L2 归一化（与 HashingEmbedder 一致 → 检索可用点积快路径）。
    """

    def __init__(self, model_dir: Optional[str] = None,
                 revision: Optional[str] = "bge-base-zh-v1.5-r1",
                 max_length: int = 512, dim: int = 768) -> None:
        self._model_dir = model_dir or _default_model_dir()
        self.revision = revision or "bge-base-zh-v1.5-r1"
        self.max_length = max_length
        self.dim = dim
        self.model = self.revision.split("-r")[0]      # 指纹里的可读模型名
        self.normalized = True
        self._session = None                      # 懒加载缓存
        self._tok = None

    # -- 常量/路径 ----------------------------------------------------------
    def require_files(self) -> List[str]:
        """列出模型目录必须存在的文件；缺失文件集合用于给出精确错误。"""
        return [os.path.join(self._model_dir, f)
                for f in ("model.onnx", "tokenizer.json")]

    # -- 懒加载 -------------------------------------------------------------
    def _ensure_loaded(self):
        if self._session is not None:
            return
        miss = [f for f in self.require_files() if not os.path.exists(f)]
        if miss:
            raise EmbeddingInstallRequired(
                "缺少模型文件: " + ", ".join(os.path.basename(m) for m in miss))
        try:
            import onnxruntime  # 懒加载：重依赖不影响零依赖默认路径
        except ImportError as e:  # pragma: no cover
            raise EmbeddingInstallRequired("缺少 onnxruntime: " + str(e))
        try:
            from tokenizers import Tokenizer  # 懒加载
        except ImportError as e:  # pragma: no cover
            raise EmbeddingInstallRequired("缺少 tokenizers: " + str(e))
        so = onnxruntime.SessionOptions()
        so.intra_op_num_threads = max(1, (os.cpu_count() or 2) - 1)
        self._session = onnxruntime.InferenceSession(
            os.path.join(self._model_dir, "model.onnx"), so,
            providers=["CPUExecutionProvider"])
        self._tok = Tokenizer.from_file(os.path.join(self._model_dir, "tokenizer.json"))

    def embed(self, text: str) -> List[float]:
        self._ensure_loaded()
        enc = self._tok.encode(str(text or ""))
        ids = enc.ids[:self.max_length]
        mask = [1] * len(ids)
        seg = [0] * len(ids)
        # 填充到 max_length（CLS 位在 index 0，需保留）
        ids = ids + [0] * (self.max_length - len(ids))
        mask = mask + [0] * (self.max_length - len(mask))
        seg = seg + [0] * (self.max_length - len(seg))
        out = self._session.run(
            None, {"input_ids": [ids], "attention_mask": [mask],
                   "token_type_ids": [seg]})[0]   # (1, seq, hidden) 或 (1, hidden)
        if out.ndim == 2:                          # 部分导出归一化后的 (1,hidden)
            vec = out[0].astype(float)
        else:                                      # (1, seq, hidden) → BGE 用 [CLS]
            vec = out[0, 0].astype(float)
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


def _default_model_dir() -> str:
    return os.environ.get(
        "SUPERBRAIN_EMBEDDER_MODEL_DIR",
        os.path.join(os.path.expanduser("~"), ".superbrain", "models", "bge-base-zh-v1.5"))


_EMBEDDER_REGISTRY = {
    "hashing": lambda **kw: HashingEmbedder(
        dim=kw.get("dim", 256), ngram=kw.get("ngram", 2),
        enhance=kw.get("enhance", False), strip_stop=kw.get("strip_stop", False)),
    "bge": lambda **kw: OnnxEmbedder(
        model_dir=kw.get("model_dir"), revision=kw.get("revision"),
        max_length=kw.get("max_length", 512), dim=kw.get("dim", 768)),
}


def build_embedder(name: str = "hashing", **kwargs) -> Embedder:
    """嵌入器注册表：hashing(默认,零依赖) / bge(真实模型,需安装)。"""
    name = (name or "hashing").lower()
    if name not in _EMBEDDER_REGISTRY:
        raise ValueError(
            f"未知嵌入器 '{name}'，可选: {', '.join(sorted(_EMBEDDER_REGISTRY))}")
    return _EMBEDDER_REGISTRY[name](**kwargs)


def registered_embedders() -> List[str]:
    return sorted(_EMBEDDER_REGISTRY)
