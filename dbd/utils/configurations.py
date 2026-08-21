from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RunConfiguration:
    """User-adjustable settings for one community monitoring run."""

    monitor_id: int
    threshold: int
    frontier_threshold: int
    frontier_delay: int
