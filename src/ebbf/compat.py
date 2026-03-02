"""Cross-platform compatibility utilities."""

import os


def get_num_cpus() -> int:
    """Get number of available CPUs, with fallbacks for cross-platform compatibility.

    Tries os.sched_getaffinity (Linux), falls back to os.cpu_count, then to 1.
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        # macOS and Windows don't have sched_getaffinity
        return os.cpu_count() or 1
