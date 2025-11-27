from __future__ import annotations

import math
import uuid
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..llm import LLM
from .task import EventReport, Task


class CognitiveState(Enum):
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"


class AbstractAgent(ABC):
    """Abstract base class for polymorphic agents."""

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name or self.__class__.__name__
        self.agent_id = str(uuid.uuid4())

    @abstractmethod
    async def on_event(self, event: EventReport) -> Optional[Task]:
        """Process an incoming event. Agents that do not issue tasks may return None."""


class SupportingAgent(AbstractAgent):
    """Environment sensor responsible for emitting structured reports."""

    async def on_event(self, event: EventReport) -> Optional[Task]:  # pragma: no cover - interface only
        return None

    async def detect_changes(self, env_snapshot: Dict[str, Any]) -> List[EventReport]:
        changes = []
        for key, value in env_snapshot.items():
            if value:  # treat truthy values as signals
                changes.append(EventReport(trigger=key, payload={"value": value}))
        return changes


class ManagingAgent(AbstractAgent):
    """Task issuer that translates environment reports into task objects."""

    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__(name=name)
        self.issue_history: List[Task] = []

    async def on_event(self, event: EventReport) -> Optional[Task]:
        summary = f"Respond to {event.trigger}"
        task = Task(
            id=str(uuid.uuid4()),
            summary=summary,
            context=event.payload,
            rewards={"Ps": 1.0, "Ph": 1.0, "Re": 1.0},
        )
        self.issue_history.append(task)
        return task


class GuidingAgent(AbstractAgent):
    """Logic provider that refines tasks with candidate actions and priors."""

    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__(name=name)

    async def on_event(self, event: EventReport) -> Optional[Task]:  # pragma: no cover - guidance is task specific
        return None

    async def enrich_task(self, task: Task) -> Task:
        # Provide a default set of actions and Ps/Ph/Re priors when none exist.
        if not task.actions:
            task.actions = [
                "investigate context",
                "communicate update",
                "wait and observe",
            ]
        task.guidance = task.guidance or {
            "Ps": {action: 0.4 for action in task.actions},
            "Ph": {action: 0.3 for action in task.actions},
            "Re": {action: 0.3 for action in task.actions},
        }
        return task


class ExecutingAgent(AbstractAgent):
    """The only agent with heavy cognitive logic (LLM-powered)."""

    def __init__(self, llm_client: LLM, name: Optional[str] = None) -> None:
        super().__init__(name=name)
        self.llm = llm_client
        self.state = CognitiveState.IDLE
        self.current_task: Optional[Task] = None
        self.weights: Dict[str, float] = {"alpha": 1.0, "beta": 1.0, "gamma": 1.0}

    async def on_event(self, event: EventReport) -> Optional[Task]:
        return None

    def set_weights(self, alpha: float, beta: float, gamma: float) -> None:
        self.weights = {"alpha": alpha, "beta": beta, "gamma": gamma}

    async def receive_task(self, task: Task) -> Dict[str, Any]:
        self.current_task = task
        self.state = CognitiveState.THINKING
        task.mark_in_progress()
        decision = await self._decide(task)
        self.state = CognitiveState.ACTING
        task.mark_completed(outcome=decision)
        self.state = CognitiveState.IDLE
        return decision

    async def _decide(self, task: Task) -> Dict[str, Any]:
        guidance = task.guidance or {}
        actions = task.actions or guidance.get("actions") or []
        if not actions:
            actions = ["noop"]
        # build pairwise preferences using the LLM
        preference_scores = await self._pairwise_preferences(actions, guidance)
        blended_scores = self._blend_preferences(actions, guidance, preference_scores)
        probabilities = self._softmax(blended_scores)
        chosen = actions[probabilities.index(max(probabilities))]
        return {"action": chosen, "probabilities": dict(zip(actions, probabilities))}

    async def _pairwise_preferences(
        self, actions: Sequence[str], guidance: Dict[str, Any]
    ) -> Dict[Tuple[str, str], float]:
        preferences: Dict[Tuple[str, str], float] = {}
        for i, left in enumerate(actions):
            for right in actions[i + 1 :]:
                prompt = (
                    "Between action A and action B, prefer the one that best balances "
                    "Ps (success probability), Ph (human preference), and Re (reward).\n"
                    "Action A: {left}\nAction B: {right}\nAnswer with either A or B."
                )
                dialog = [
                    {"role": "system", "content": "You are ranking actions by pairwise preference."},
                    {"role": "user", "content": prompt.format(left=left, right=right)},
                ]
                resp = await self.llm.atext_request(dialog, temperature=0.0)
                response_text = resp if isinstance(resp, str) else str(resp)
                choose_left = "A" in response_text.upper()
                preferences[(left, right)] = 1.0 if choose_left else -1.0
                preferences[(right, left)] = -preferences[(left, right)]
        return preferences

    def _blend_preferences(
        self, actions: Sequence[str], guidance: Dict[str, Any], pairwise: Dict[Tuple[str, str], float]
    ) -> List[float]:
        ps = guidance.get("Ps", {})
        ph = guidance.get("Ph", {})
        re = guidance.get("Re", {})
        alpha, beta, gamma = self.weights["alpha"], self.weights["beta"], self.weights["gamma"]
        blended: List[float] = []
        for action in actions:
            preference_score = sum(pairwise.get((action, other), 0.0) for other in actions if other != action)
            utility = (
                alpha * float(ps.get(action, 0.0))
                + beta * float(ph.get(action, 0.0))
                + gamma * float(re.get(action, 0.0))
            )
            blended.append(preference_score + utility)
        return blended

    @staticmethod
    def _softmax(scores: Sequence[float]) -> List[float]:
        if not scores:
            return []
        max_score = max(scores)
        exps = [math.exp(score - max_score) for score in scores]
        total = sum(exps)
        return [val / total for val in exps]


class EvaluatingAgent(AbstractAgent):
    """Updates executing agents' internal weights immediately after action."""

    def __init__(self, name: Optional[str] = None, learning_rate: float = 0.1) -> None:
        super().__init__(name=name)
        self.learning_rate = learning_rate

    async def on_event(self, event: EventReport) -> Optional[Task]:  # pragma: no cover - evaluator reacts post-action
        return None

    async def update_after_action(self, executing_agent: ExecutingAgent, task: Task) -> Dict[str, float]:
        outcome = task.outcome or {}
        realized_reward = float(outcome.get("reward", task.rewards.get("Re", 0.0)))
        alpha, beta, gamma = (
            executing_agent.weights["alpha"],
            executing_agent.weights["beta"],
            executing_agent.weights["gamma"],
        )
        alpha += self.learning_rate * realized_reward * task.rewards.get("Ps", 0.0)
        beta += self.learning_rate * realized_reward * task.rewards.get("Ph", 0.0)
        gamma += self.learning_rate * realized_reward * task.rewards.get("Re", 0.0)
        executing_agent.set_weights(alpha, beta, gamma)
        return executing_agent.weights