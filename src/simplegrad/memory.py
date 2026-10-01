"""Deterministic logical memory accounting for simple-grad allocations."""

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Integral

import numpy as np


CATEGORIES = (
    "parameters",
    "gradients",
    "optimizer_state",
    "activations_saved_for_backward",
    "transient_other",
)


def _byte_count(value, name="nbytes"):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


@dataclass(frozen=True)
class MemoryBreakdown:
    """Bytes attributed to each Task 2 memory category."""

    parameters: int = 0
    gradients: int = 0
    optimizer_state: int = 0
    activations_saved_for_backward: int = 0
    transient_other: int = 0

    def __post_init__(self):
        for name in CATEGORIES:
            object.__setattr__(self, name, _byte_count(getattr(self, name), name))

    @property
    def total_bytes(self):
        return sum(getattr(self, name) for name in CATEGORIES)

    @property
    def total(self):
        return self.total_bytes

    def to_dict(self):
        return {name: getattr(self, name) for name in CATEGORIES}


@dataclass(frozen=True)
class MemoryEvent:
    """One deterministic logical allocation or release event."""

    sequence: int
    action: str
    nbytes: int
    category: str
    source: str
    current_bytes: int


@dataclass(frozen=True)
class MemoryStats:
    """Current memory and the category snapshot at the measured peak."""

    current_bytes: int
    current: MemoryBreakdown
    peak_bytes: int
    at_peak: MemoryBreakdown
    peak_source: str | None

    def to_dict(self):
        return {
            "current_bytes": self.current_bytes,
            "current": self.current.to_dict(),
            "peak_bytes": self.peak_bytes,
            "at_peak": self.at_peak.to_dict(),
            "peak_source": self.peak_source,
        }


class AllocationHandle:
    """Explicit ownership token for one accounted allocation."""

    __slots__ = ("_tracker", "_allocation_id", "_released")

    def __init__(self, tracker, allocation_id):
        self._tracker = tracker
        self._allocation_id = allocation_id
        self._released = False

    def release(self):
        if not self._released:
            self._tracker.release(self._allocation_id)
            self._released = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()

    def __del__(self):
        try:
            self.release()
        except Exception:
            # Interpreter shutdown may tear down the tracker first.
            pass


class MemoryTracker:
    """A small ledger that counts library-owned arrays by exact byte size."""

    def __init__(self):
        self._next_id = 0
        self._sequence = 0
        self._live = {}
        self._current = {name: 0 for name in CATEGORIES}
        self._peak_bytes = 0
        self._at_peak = self._breakdown()
        self._peak_source = None
        self._events = deque(maxlen=4096)

    def _breakdown(self):
        return MemoryBreakdown(**self._current)

    def reserve(self, nbytes, category, source):
        nbytes = _byte_count(nbytes)
        if category not in CATEGORIES:
            raise ValueError(f"unknown memory category: {category!r}")
        if not isinstance(source, str) or not source:
            raise ValueError("source must be a non-empty string")

        allocation_id = self._next_id
        self._next_id += 1
        self._live[allocation_id] = (nbytes, category, source)
        self._current[category] += nbytes
        total = sum(self._current.values())
        self._record("allocate", nbytes, category, source, total)
        if total > self._peak_bytes:
            self._peak_bytes = total
            self._at_peak = self._breakdown()
            self._peak_source = source
        return AllocationHandle(self, allocation_id)

    def release(self, allocation_id):
        try:
            nbytes, category, source = self._live.pop(allocation_id)
        except KeyError as exc:
            raise RuntimeError("allocation is not live") from exc
        self._current[category] -= nbytes
        self._record(
            "release", nbytes, category, source, sum(self._current.values())
        )

    def _record(self, action, nbytes, category, source, current_bytes):
        self._events.append(
            MemoryEvent(
                sequence=self._sequence,
                action=action,
                nbytes=nbytes,
                category=category,
                source=source,
                current_bytes=current_bytes,
            )
        )
        self._sequence += 1

    def stats(self):
        current = self._breakdown()
        return MemoryStats(
            current_bytes=current.total_bytes,
            current=current,
            peak_bytes=self._peak_bytes,
            at_peak=self._at_peak,
            peak_source=self._peak_source,
        )

    def reset_peak(self):
        current = self._breakdown()
        self._peak_bytes = current.total_bytes
        self._at_peak = current
        self._peak_source = "reset_peak_memory"
        self._events.clear()
        self._sequence = 0

    def events(self):
        return tuple(self._events)


_TRACKER = MemoryTracker()


def track_bytes(nbytes, category, source):
    """Register bytes and return the handle their owner must retain."""
    return _TRACKER.reserve(nbytes, category, source)


def track_array(array, category, source):
    """Register an already-created NumPy array by its exact payload size."""
    if not isinstance(array, np.ndarray):
        array = np.asarray(array)
    return track_bytes(array.nbytes, category, source)


@contextmanager
def temporary_array(array, source):
    """Account for an array as transient for the duration of a scope."""
    handle = track_array(array, "transient_other", source)
    try:
        yield array
    finally:
        handle.release()


def get_memory_stats():
    """Return an immutable snapshot of current and peak logical memory."""
    return _TRACKER.stats()


def reset_peak_memory():
    """Start a new peak interval while preserving currently live storage."""
    _TRACKER.reset_peak()


def get_memory_event_trace():
    """Return up to the 4,096 most recent events since the peak reset."""
    return _TRACKER.events()
