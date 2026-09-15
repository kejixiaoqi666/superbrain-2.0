"""主动运行：定时任务调度。

让 agent 不只被动应答，还能主动定时感知——巡检、观察、回顾。
这是「自主智能体」从被动到主动的最后一块。
"""
from __future__ import annotations

import threading
import time
import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("superbrain.scheduler")


@dataclass
class ScheduledJob:
    """一个定时任务。"""
    name: str
    interval_seconds: float
    func: Callable[[], Optional[str]]
    last_run: float = 0.0
    run_count: int = 0

    def due(self, now: float) -> bool:
        return (now - self.last_run) >= self.interval_seconds


class Scheduler:
    """轻量后台调度器（线程驱动，零依赖）。

    线程安全：``_jobs`` 增删读均受 RLock 保护；``func`` 在锁外执行避免持锁阻塞。
    """

    def __init__(self) -> None:
        self._jobs: Dict[str, ScheduledJob] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

    def add(self, name: str, interval_seconds: float, func: Callable[[], Optional[str]]) -> None:
        # last_run 初始化为当前时间：新任务首次触发前等一个完整周期，
        # 避免加入后首次循环立即执行（尤其 start 后 autopilot 任务全跑 + 与 close 竞态）。
        with self._lock:
            self._jobs[name] = ScheduledJob(name=name, interval_seconds=interval_seconds,
                                            func=func, last_run=time.time())

    def remove(self, name: str) -> None:
        with self._lock:
            self._jobs.pop(name, None)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def join(self, timeout: Optional[float] = None) -> None:
        """等待后台线程退出（配合 stop() 干净关闭，避免线程访问已释放资源）。"""
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout)

    def _loop(self) -> None:
        while self._running:
            now = time.time()
            with self._lock:
                jobs = list(self._jobs.values())  # 快照，避免与 add/remove 并发改 dict
            for job in jobs:
                if job.due(now):
                    try:
                        job.func()  # 锁外执行，避免持锁调用外部 func 阻塞调度
                    except Exception:
                        # 后台任务异常不中断调度，但必须记录供诊断（否则持续失败无法察觉）
                        logger.warning("后台任务 %r 执行异常", job.name, exc_info=True)
                    job.last_run = now
                    job.run_count += 1
            time.sleep(0.1)  # 细粒度轮询：新任务 0.1~0.3s 内响应，避免 1s 粒度漏触发

    def list(self) -> List[dict]:
        with self._lock:
            return [{"name": j.name, "interval": j.interval_seconds,
                     "run_count": j.run_count} for j in self._jobs.values()]