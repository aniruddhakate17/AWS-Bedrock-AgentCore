from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

from a2a_common import build_agent_response
from groq_provider import build_groq_model

_ = load_dotenv()


def handle_summary_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    request_id = payload.get("request_id", "unknown-request")
    user_query = payload.get("input", {}).get("user_query", "")
    worker_outputs: List[Dict[str, Any]] = payload.get("input", {}).get("worker_outputs", [])

    findings = "\n\n".join(
        [f"[{item.get('worker', 'worker')}]\n{item.get('output', '')}" for item in worker_outputs]
    )

    llm = build_groq_model()
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

    return build_agent_response(
        request_id=request_id,
        agent_name="summary",
        status="success",
        result={
            "answer": response.content,
            "worker_outputs": worker_outputs,
        },
        confidence=0.88,
        sources=[],
        metadata={"task_type": payload.get("task_type", "summarize")},
    )
