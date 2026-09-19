from __future__ import annotations
import tracemalloc
from typing import List, Optional, Any
import gc


class AbstractProfiler:
    r"""
    统计器抽象基类。

    用于对各种信息进行统计，比如：代码执行耗时、内存增减。同一个实例不允许嵌套使用

    Args:
        scene:分析场景，比如：统计代码执行耗时。
        consumer:统计信息消费者，如果没有消费者，则直接``print``统计信息。
    """

    _scene: str
    _consumer: list[Any]

    def __init__(self, scene: str, consumer: list[Any] | None = None):
        self._scene = scene
        self._consumer = consumer

    def set_scene(self, scene: str) -> AbstractProfiler:
        self._scene = scene
        return self

    def set_consumer(self, consumer: list[Any]) -> AbstractProfiler:
        self._consumer = consumer
        return self

    def consum(self, info: Any) -> None:
        if self._consumer is None:
            print(info)
        else:
            self._consumer.append(info)


class CodeExecutionProfiler(AbstractProfiler):
    r"""
    上下文管理器，统计代码块执行耗时，结果追加至外部日志列表。

    Args:
        desc: 阶段名称/描述标识符，用于标记该段代码（如 data_load, forward, backward）
        logs: 外部日志列表引用，统计完成后会将结果 append 到此列表；不做拷贝，外部修改会影响实例
        start_time: 可选外部起始时间戳(纳秒，time.perf_counter_ns())。
            若传入，则 __enter__ 不再重新计时，复用该时间点，用于分段拼接计时场景(统计for循环迭代)。

    使用 `time.perf_counter_ns()` 高精度纳秒时钟，统计**墙上时钟真实耗时**，包含IO、阻塞等待时间。
    退出上下文时自动计算耗时，格式化后追加到传入的 logs 列表；代码块抛出异常时依旧会完成计时，异常继续向外抛出。

    Notes:
        1. 保存外部 list 的引用，**不会拷贝列表**；外部对列表的修改会同步影响实例内部。
        2. 仅配合 `with` 语句使用，禁止手动直接调用 __enter__ / __exit__。
        3. 代码块内部抛出异常，依然会正常完成耗时统计，异常继续向上抛出。
        4. 时钟基于 perf_counter：包含休眠时间，不受系统时间修改影响，适合性能测量。

    Example:
        >>> log_list = []
        >>> with CodeExecutionProfiler("forward", log_list):
        ...     heavy_work()
        >>> print(log_list)
        ['forward 12.34 ms']
    """

    _start_time: int | None

    _per_start_time: int

    def __init__(
        self, scene: str, consumer: List[Any] = None, start_time: int | None = None
    ):
        super().__init__(scene=scene, consumer=consumer)
        self.set_start_time(start_time)

    def __enter__(self):
        if self._start_time is None:
            self._per_start_time = ProfilerService.get_current_time()

    def __exit__(self, exc_type, exc, tb):
        delta_ns = ProfilerService.get_current_time() - self._per_start_time
        delta_ms = delta_ns / 1_000_000
        self.consum(f"[{self._scene}] {delta_ms:.2f} ms")

    def set_start_time(self, start_time: int | None) -> CodeExecutionProfiler:
        r"""
        设置开始时间。

        Args:
            start_time:开始时间，比如：统计每一轮for循环，此时无法写在with中，
            可以在循环外统计好时间（每轮循环结束更新），这样就可以统计了，注意这里的时间必须是``time.percount_ns``产生的。
        """
        self._start_time = self._per_start_time = start_time
        return self


class SnapshotProfiler(AbstractProfiler):

    _start_snapshot: tracemalloc.Snapshot

    _per_start_snapshot: tracemalloc.Snapshot

    _topn: int

    def __init__(
        self,
        scene,
        consumer=None,
        start_snapshot: Optional[tracemalloc.Snapshot] = None,
        topn: int = 5,
    ):
        super().__init__(scene, consumer)
        self.set_start_snapshot(start_snapshot)
        self._topn = topn

    def __enter__(self):

        if not tracemalloc.is_tracing():
            tracemalloc.start()
        if self._start_snapshot is None:
            self._per_start_snapshot = ProfilerService.get_task_py_snapshot()

    def __exit__(self, exc_type, exc, tb):
        diffs = ProfilerService.get_task_py_snapshot().compare_to(
            self._start_snapshot, key_type="lineno"
        )
        self.consum({f"[{self._scene}] ": [str[diff] for diff in diffs[: self._topn]]})

    def set_start_snapshot(
        self, snapshot: Optional[tracemalloc.Snapshot]
    ) -> SnapshotProfiler:
        self._start_snapshot = self._per_start_snapshot = snapshot
        return self

    def set_topn(self, topn: int) -> SnapshotProfiler:
        self._topn = topn
        return self


class MemoryUsageProfiler(AbstractProfiler):
    """内存用量统计器

    注意：只追踪Python层对象；C扩展/Tensor/Numpy底层buffer不计入。

    Attributes:
        delta_mb: 代码块内存增量 MB
        diff_stats: snap_after.compare_to(snap_before) 的原始stat列表，with结束后外部可访问
        snap_before: 进入上下文快照
        snap_after: 退出上下文快照
        start_current: 起始时刻traced_memory current字节数
        end_current: 结束时刻traced_memory current字节数
    """

    _start_py_mm: int
    _start_all_mm: int
    _per_start_py_mm: int
    _per_start_all_mm: int

    def __init__(
        self,
        scene: str,
        consumer: Optional[List[Any]] = None,
        start_py_mm: Optional[int] = None,
        start_all_mm: Optional[int] = None,
    ):
        super().__init__(scene=scene, consumer=consumer)
        self.set_start_am(start_all_mm)
        self.set_start_pm(start_py_mm)

    def __enter__(self):
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        gc.disable()

        if self._start_py_mm is None:
            self._start_py_mm = self._per_start_py_mm = (
                ProfilerService.get_task_py_memory()
            )

        if self._start_all_mm is None:
            self._start_all_mm = self._per_start_all_mm = (
                ProfilerService.get_task_all_memory()
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        delta_py_mm = ProfilerService.get_task_py_memory() - self._per_start_py_mm
        delta_all_mm = ProfilerService.get_task_all_memory() - self._per_start_all_mm
        self.consum(
            f"{[self._scene]} python内存新增 {ProfilerService.format_memory(delta_py_mm)}"
        )
        self.consum(
            f"{[self._scene]} 物理内存新增 {ProfilerService.format_memory(delta_all_mm)}"
        )
        gc.enable()

    def set_start_pm(self, py_mm: float) -> MemoryUsageProfiler:
        self._start_py_mm = self._per_start_py_mm = py_mm
        return self

    def set_start_am(self, all_mm: float) -> MemoryUsageProfiler:
        self._start_all_mm = self._per_start_all_mm = all_mm
        return self


class ProfilerService:

    B = 1
    KB = 1024
    MB = 1024 * 1024
    GB = 1024 * 1024 * 1024

    @staticmethod
    def get_current_time() -> int:
        import time

        return time.perf_counter_ns()

    @staticmethod
    def get_task_py_memory() -> float:
        import tracemalloc

        return tracemalloc.get_traced_memory()[0]

    @staticmethod
    def get_task_py_snapshot() -> float:
        import tracemalloc

        return tracemalloc.take_snapshot()

    @staticmethod
    def get_task_all_memory() -> float:
        import psutil
        import os

        return psutil.Process(os.getpid()).memory_info().rss

    @staticmethod
    def format_memory(bytes: float):
        is_minus = False
        if bytes < 0:
            is_minus = True
            bytes = abs(bytes)
        mm_str = None
        if bytes < ProfilerService.KB:
            mm_str = f"{bytes} B"
        elif bytes < ProfilerService.MB:
            mm_str = f"{bytes / ProfilerService.KB:.2f} KB"
        elif bytes < ProfilerService.GB:
            mm_str = f"{bytes / ProfilerService.MB:.2f} MB"
        else:
            mm_str = f"{bytes / ProfilerService.GB:.2f} GB"
        if is_minus:
            mm_str = f"-{mm_str}"
        return mm_str
