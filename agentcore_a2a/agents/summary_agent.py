from typing import Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from agentcore_a2a.schemas import WorkerResult
from agentcore_a2a.services.model_factory import GroqModelFactory
from agentcore_a2a.services.request_response import AgentMessageFactory


class SummaryAgent:
    def __init__(self, model_factory: GroqModelFactory, message_factory: AgentMessageFactory) -> None:
        self.model_factory = model_factory
        self.message_factory = message_factory

    def handle(self, payload: Dict[str, object]) -> Dict[str, object]:
        request_id = payload.get("request_id", "unknown-request")
        user_query = payload.get("input", {}).get("user_query", "")
        worker_outputs: List[Dict[str, str]] = payload.get("input", {}).get("worker_outputs", [])
        findings = "\n\n".join(
            [f"[{item.get('worker', 'worker')}]\n{item.get('output', '')}" for item in worker_outputs]
        )

        llm = self.model_factory.build()
        response = llm.invoke(
            [
                SystemMessage(
                    content="""You are Summary Agent.

You synthesize specialist worker outputs into one final answer.

Rules:
1. Use only the worker outputs you are given.
2. Keep the answer concise, clear, and grounded.
3. If a worker says infrastructure is missing, preserve that limitation honestly.
4. Do not fabricate facts beyond the worker outputs.
"""
                ),
                HumanMessage(
                    content=(
                        f"User query:\n{user_query}\n\n"
                        f"Worker outputs:\n{findings}\n\n"
                        "Create the final user-facing answer."
                    )
                ),
            ]
        )

        return self.message_factory.build_response(
            request_id=request_id,
            agent_name="summary",
            status="success",
            result={"answer": response.content, "worker_outputs": worker_outputs},
            confidence=0.88,
            sources=[],
            metadata={"task_type": payload.get("task_type", "summarize")},
        )

    def to_worker_result(self, response: Dict[str, object]) -> WorkerResult:
        return WorkerResult(
            worker="summary_agent",
            output=response.get("result", {}).get("answer", str(response)),
            status=response.get("status", "success"),
            confidence=response.get("confidence", 0.0),
            raw_response=response,
        )

