from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from .agents import EvaluatingAgent, ExecutingAgent, GuidingAgent, ManagingAgent, SupportingAgent
from .task import EventReport, Task


class EventDrivenScheduler:
    """Event-driven coordinator that replaces random activation with task objects."""

    def __init__(
        self,
        supporting_agent: SupportingAgent,
        managing_agent: ManagingAgent,
        executing_agent: ExecutingAgent,
        guiding_agent: GuidingAgent,
        evaluating_agent: EvaluatingAgent,
    ) -> None:
        self.supporting_agent = supporting_agent
        self.managing_agent = managing_agent
        self.executing_agent = executing_agent
        self.guiding_agent = guiding_agent
        self.evaluating_agent = evaluating_agent
        self.task_queue: asyncio.Queue[Task] = asyncio.Queue()
        self.history: List[Task] = []

    async def ingest_environment(self, env_snapshot: Dict[str, Any]) -> List[EventReport]:
        """Detect environment changes and enqueue resulting tasks."""
        events = await self.supporting_agent.detect_changes(env_snapshot)
        for event in events:
            task = await self.managing_agent.on_event(event)
            if task is None:
                continue
            enriched = await self.guiding_agent.enrich_task(task)
            await self.task_queue.put(enriched)
        return events

    async def dispatch(self) -> Optional[Task]:
        """Assign the next task to the executing agent and update weights post-action."""
        if self.executing_agent.state.name != "IDLE":
            return None
        if self.task_queue.empty():
            return None
        task = await self.task_queue.get()
        decision = await self.executing_agent.receive_task(task)
        # Attach decision for downstream evaluators
        task.outcome = task.outcome or {}
        task.outcome.update(decision)
        await self.evaluating_agent.update_after_action(self.executing_agent, task)
        self.history.append(task)
        return task

    async def tick(self, env_snapshot: Dict[str, Any]) -> List[Task]:
        """Single tick in the event-driven loop.

        1. SupportingAgent emits EventReports when the environment snapshot changes.
        2. ManagingAgent translates reports into Task objects.
        3. GuidingAgent attaches action priors (Ps/Ph/Re) to Task.guidance.
        4. ExecutingAgent pops the next Task (if idle), runs U-I-R softmax selection, and acts.
        5. EvaluatingAgent immediately updates (alpha, beta, gamma) with deterministic rules.
        """

        await self.ingest_environment(env_snapshot)
        completed: List[Task] = []
        dispatched = await self.dispatch()
        if dispatched:
            completed.append(dispatched)
        return completed
