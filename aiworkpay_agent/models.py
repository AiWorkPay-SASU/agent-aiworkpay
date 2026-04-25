import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    RESEARCH = "research"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    VERIFICATION = "verification"
    CODE = "code"
    CUSTOM = "custom"


class TaskSource(str, Enum):
    AI = "ai"
    HUMAN = "human"


class Task(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: TaskType
    content: str
    context: dict[str, Any] = Field(default_factory=dict)
    budget: float = 0.10
    priority: int = Field(default=5, ge=1, le=10)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class TaskResult(BaseModel):
    task_id: str
    result: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    source: TaskSource
    cost: float = 0.0
    duration_ms: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class FinalResult(BaseModel):
    objective: str
    final_result: str
    sources: list[TaskResult]
    total_cost: float
    total_duration_ms: int
    human_tasks: int
    ai_tasks: int
    loop_iterations: int = 1
