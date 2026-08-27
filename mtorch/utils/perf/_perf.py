import time
import tracemalloc
from typing import List, Optional, Sequence, Any


class CodeExecutionProfiler:
    """上下文管理器，用于统计代码块执行耗时，结果追加到外部日志列表。

    使用 `time.perf_counter_ns()` 高精度纳秒时钟，计算真实墙钟时间（包含IO、阻塞等待），
    退出上下文时自动格式化毫秒字符串，append到传入的logs列表。

    注意：
        1. 保存外部list的引用，不会拷贝列表；外部对list的修改会影响实例内部。
        2. 仅用于with语句，不要手动直接调用 __enter__ / __exit__。
        3. 代码块内部抛出异常，依然会正常统计耗时，异常会继续向外抛出。

    Example:
        >>> log_list: List[str] = []
        >>> with PerfTimer("forward", log_list):
        ...     heavy_work()
        >>> print(log_list)
        ['forward 12.34 ms']
    """

    _desc: str
    _logs: List[str]
    _start_time: int | None

    def __init__(
        self, desc: str, logs: List[str] = None, start_time: int | None = None
    ):
        """
        Args:
            desc: 计时块描述文本，输出日志的标签
            logs: 外部字符串列表，计时结果会append到此列表
        """
        self._desc = desc
        self._logs = logs
        self._start_time = start_time

    def __enter__(self):
        """进入上下文，记录纳秒级起始时间戳。"""
        if self._start_time is None:
            self._start_time = time.perf_counter_ns()

    def __exit__(self, exc_type, exc, tb):
        """退出上下文，计算耗时，格式化并写入日志列表。

        Args:
            exc_type: 异常类型，块内无异常则为 None
            exc: 异常实例，块内无异常则为 None
            tb: 异常traceback对象，块内无异常则为 None
        """
        if self._start_time is None:
            return
        delta_ns = time.perf_counter_ns() - self._start_time
        delta_ms = delta_ns / 1_000_000
        log_str = f"{self._desc} {delta_ms:.2f} ms"
        if self._logs is not None:
            self._logs.append(log_str)
        else:
            print(log_str)


class MemoryUsageProfiler:
    """上下文管理器，tracemalloc快照对比，支持外部传入起始内存与起始快照。

    注意：只追踪Python层对象；C扩展/Tensor/Numpy底层buffer不计入。

    Attributes:
        delta_mb: 代码块内存增量 MB
        diff_stats: snap_after.compare_to(snap_before) 的原始stat列表，with结束后外部可访问
        snap_before: 进入上下文快照
        snap_after: 退出上下文快照
        start_current: 起始时刻traced_memory current字节数
        end_current: 结束时刻traced_memory current字节数
    """

    def __init__(
        self,
        desc: str,
        logs: Optional[List[Any]] = None,
        top_n: int = 5,
        start_current: Optional[int] = None,
        snap_before: Optional[tracemalloc.Snapshot] = None,
        enable_snapshot: bool = False,
    ):
        """
        Args:
            desc: 标签
            logs: 输出文本日志列表
            top_n: 写入logs的top N分配记录
            start_current: 外部传入起始内存current字节；为None则__enter__内部自动采集
            snap_before: 外部传入起始快照；为None则__enter__内部自动take_snapshot()
        """
        self._desc = desc
        self._logs = logs
        self._top_n = top_n

        # 外部预传的基准
        self.start_current: Optional[int] = start_current
        self.snap_before: Optional[tracemalloc.Snapshot] = snap_before

        self.snap_after: Optional[tracemalloc.Snapshot] = None
        self.end_current: int = 0
        self.delta_mb: float = 0.0
        self.diff_stats: Sequence[tracemalloc.StatisticDiff] = []
        self.enable_snapshot = enable_snapshot

    def __enter__(self):
        if not tracemalloc.is_tracing():
            tracemalloc.start()

        # 构造函数没有传入，则进入时现场采集
        if self.start_current is None:
            self.start_current, _ = tracemalloc.get_traced_memory()
        if self.snap_before is None and self.enable_snapshot:
            self.snap_before = tracemalloc.take_snapshot()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # if (
        #     self.snap_before is None and not self.enable_snapshot
        # ) or self.start_current is None:
        #     return

        self.end_current, _ = tracemalloc.get_traced_memory()
        if self.enable_snapshot:
            self.snap_after = tracemalloc.take_snapshot()

            self.diff_stats = self.snap_after.compare_to(
                self.snap_before, key_type="lineno"
            )
        delta_bytes = self.end_current - self.start_current
        self.delta_mb = delta_bytes / (1024 * 1024)
        desc = self._desc
        log = {f"{desc}内存新增:": f"{self.delta_mb:+.2f} MB"}
        has_log = self._logs is not None
        if has_log:
            self._logs.append(log)
        else:
            print(log)
        if self.enable_snapshot:
            stats = []
            log[f"{desc}统计信息"] = stats
            for stat in self.diff_stats[: self._top_n]:
                stats.append(str(stat))
