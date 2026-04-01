from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from agentcore_a2a.schemas import WorkerResult
from agentcore_a2a.services.model_factory import GroqModelFactory
from agentcore_a2a.services.request_response import AgentMessageFactory
from agentcore_a2a.tools.faq_search import FAQSearchTool


class SearchAgent:
    def __init__(
        self,
        faq_search_tool: FAQSearchTool,
        model_factory: GroqModelFactory,
        message_factory: AgentMessageFactory,
    ) -> None:
        self.faq_search_tool = faq_search_tool
        self.model_factory = model_factory
        self.message_factory = message_factory

    def handle(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        request_id = payload.get("request_id", "unknown-request")
        query = payload.get("input", {}).get("query") or payload.get("prompt", "")
        top_k = int(payload.get("input", {}).get("top_k", 3))

        matches = self.faq_search_tool.search(query=query, top_k=top_k)
        if not matches:
            return self.message_factory.build_response(
                request_id=request_id,
                agent_name="search",
                status="success",
                result={"answer": "No relevant FAQ entries found for this request.", "matches": []},
                confidence=0.2,
                sources=["lauki_qna.csv"],
                metadata={"task_type": payload.get("task_type", "faq_lookup")},
            )

        faq_context = "\n\n".join(
            [f"FAQ Match {item['rank']}:\n{item['content']}" for item in matches]
        )
        llm = self.model_factory.build()
        response = llm.invoke(
            [
                SystemMessage(
                    content="""You are Search Agent.

You answer only from the provided FAQ matches.

Rules:
1. Stay grounded in the retrieved FAQ entries.
2. Do not invent account-specific information.
3. Give a useful answer that can later be summarized by another agent if needed.
4. If the FAQ is incomplete, say that clearly.
"""
                ),
                HumanMessage(
                    content=(
                        f"User query:\n{query}\n\n"
                        f"Retrieved FAQ matches:\n{faq_context}\n\n"
                        "Return a grounded answer based only on the FAQ matches."
                    )
                ),
            ]
        )

        return self.message_factory.build_response(
            request_id=request_id,
            agent_name="search",
            status="success",
            result={"answer": response.content, "matches": matches},
            confidence=0.9,
            sources=["lauki_qna.csv"],
            metadata={"task_type": payload.get("task_type", "faq_lookup")},
        )

    def to_worker_result(self, response: Dict[str, Any]) -> WorkerResult:
        return WorkerResult(
            worker="search_agent",
            output=response.get("result", {}).get("answer", str(response)),
            status=response.get("status", "success"),
            confidence=response.get("confidence", 0.0),
            raw_response=response,
        )

