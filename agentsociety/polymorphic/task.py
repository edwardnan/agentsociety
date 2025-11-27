from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class EventReport:
    """Structured signal emitted by the SupportingAgent when the environment changes."""

    trigger: str
    payload: Dict[str, Any]
    detected_at: float = field(default_factory=time.time)


@dataclass
class Task:
    """Atomic unit of work passed through the event-driven scheduler."""

    id: str
    summary: str
    context: Dict[str, Any]
    rewards: Dict[str, float]
    issued_at: float = field(default_factory=time.time)
    status: TaskStatus = TaskStatus.PENDING
    guidance: Optional[Dict[str, Any]] = None
    actions: List[str] = field(default_factory=list)
    outcome: Optional[Dict[str, Any]] = None

    def mark_in_progress(self) -> None:
        self.status = TaskStatus.IN_PROGRESS

    def mark_completed(self, outcome: Optional[Dict[str, Any]] = None) -> None:
        self.status = TaskStatus.COMPLETED
        self.outcome = outcome or {}

    def mark_failed(self, outcome: Optional[Dict[str, Any]] = None) -> None:
        self.status = TaskStatus.FAILED
        self.outcome = outcome or {}
