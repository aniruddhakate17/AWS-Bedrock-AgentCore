from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentRequest:
    actor_id: str
    thread_id: str
    source_agent: str
    target_agent: str
    task_type: str
    input: Dict[str, Any]
    context: Dict[str, Any] = field(default_factory=dict)
    request_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentResponse:
    request_id: str
    agent_name: str
    status: str
    result: Dict[str, Any]
    confidence: float
    sources: List[str] = field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WorkerResult:
    worker: str
    output: str
    status: str = "success"
    confidence: float = 0.0
    raw_response: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "worker": self.worker,
            "output": self.output,
            "status": self.status,
            "confidence": self.confidence,
            "raw_response": self.raw_response,
        }

