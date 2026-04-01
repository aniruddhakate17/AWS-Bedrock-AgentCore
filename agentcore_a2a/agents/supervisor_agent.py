import json
import os
from typing import Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from agentcore_a2a.agents.db_agent import DatabaseAgent
from agentcore_a2a.agents.search_agent import SearchAgent
from agentcore_a2a.agents.summary_agent import SummaryAgent
from agentcore_a2a.config import AgentCoreConfigLoader
from agentcore_a2a.schemas import WorkerResult
from agentcore_a2a.services.model_factory import GroqModelFactory
from agentcore_a2a.services.request_response import AgentMessageFactory
from agentcore_a2a.services.runtime_invoker import AgentRuntimeInvoker


class SupervisorAgent:
    def __init__(
        self,
        model_factory: GroqModelFactory,
        message_factory: AgentMessageFactory,
        config_loader: AgentCoreConfigLoader,
        runtime_invoker: AgentRuntimeInvoker,
        search_agent: SearchAgent,
        summary_agent: SummaryAgent,
        db_agent: DatabaseAgent,
    ) -> None:
        self.model_factory = model_factory
        self.message_factory = message_factory
        self.config_loader = config_loader
        self.runtime_invoker = runtime_invoker
        self.search_agent = search_agent
        self.summary_agent = summary_agent
        self.db_agent = db_agent
        self.search_runtime_arn = os.getenv("A2A_SEARCH_RUNTIME_ARN", "").strip()
        self.summary_runtime_arn = os.getenv("A2A_SUMMARY_RUNTIME_ARN", "").strip()

    def choose_route(self, query: str) -> Dict[str, str]:
        llm = self.model_factory.build()
        response = llm.invoke(
            [
                SystemMessage(
                    content="""You are Supervisor Agent.

Return only valid JSON with this schema:
{
  "route": "search_only" | "db_only" | "search_then_summary" | "db_then_summary" | "search_and_db_then_summary",
  "reason": "short explanation"
}

Routing guidance:
- Use search for FAQ, policy, activation, troubleshooting, roaming, pricing, and product knowledge questions.
- Use db for account-specific, plan-specific, usage-specific, transaction-specific, or SQL/database-backed questions.
- Use search_and_db_then_summary when the request needs both knowledge retrieval and account-specific facts.
- Use a *_then_summary route when the user asks for comparison, explanation, summarization, or a polished final response.
"""
                ),
                HumanMessage(content=query),
            ]
        )

        raw = response.content.strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            lower_query = query.lower()
            if any(word in lower_query for word in ["my ", "account", "plan", "usage", "bill", "sql", "database"]):
                return {"route": "db_then_summary", "reason": "Fallback routing for account-specific question."}
            return {"route": "search_then_summary", "reason": "Fallback routing for knowledge question."}

        valid_routes = {
            "search_only",
            "db_only",
            "search_then_summary",
            "db_then_summary",
            "search_and_db_then_summary",
        }
        if parsed.get("route") not in valid_routes:
            return {"route": "search_then_summary", "reason": "Invalid route from supervisor model."}
        return parsed

    def _get_runtime_arn(self, agent_name: str, env_override: str = "") -> str:
        if env_override:
            return env_override
        return self.config_loader.get_runtime(agent_name).agent_arn

    def _invoke_search(self, actor_id: str, thread_id: str, query: str) -> WorkerResult:
        payload = self.message_factory.build_request(
            actor_id=actor_id,
            thread_id=thread_id,
            source_agent="supervisor",
            target_agent="search",
            task_type="faq_lookup",
            input_data={"query": query, "top_k": 3},
        )
        runtime_arn = self._get_runtime_arn("search_agent", self.search_runtime_arn)
        response = self.runtime_invoker.invoke(runtime_arn, payload, "search", thread_id)
        return self.search_agent.to_worker_result(response)

    def _invoke_summary(self, actor_id: str, thread_id: str, user_query: str, worker_outputs: List[WorkerResult]) -> WorkerResult:
        payload = self.message_factory.build_request(
            actor_id=actor_id,
            thread_id=thread_id,
            source_agent="supervisor",
            target_agent="summary",
            task_type="summarize",
            input_data={
                "user_query": user_query,
                "worker_outputs": [{"worker": item.worker, "output": item.output} for item in worker_outputs],
            },
        )
        runtime_arn = self._get_runtime_arn("summary_agent", self.summary_runtime_arn)
        response = self.runtime_invoker.invoke(runtime_arn, payload, "summary", thread_id)
        return self.summary_agent.to_worker_result(response)

    def handle(self, query: str, actor_id: str, thread_id: str) -> Dict[str, object]:
        route = self.choose_route(query)
        worker_outputs: List[WorkerResult] = []

        if route["route"] == "search_only":
            search_result = self._invoke_search(actor_id, thread_id, query)
            worker_outputs.append(search_result)
            final_answer = search_result.output
        elif route["route"] == "db_only":
            db_result = self.db_agent.handle(query, actor_id)
            worker_outputs.append(db_result)
            final_answer = db_result.output
        elif route["route"] == "db_then_summary":
            db_result = self.db_agent.handle(query, actor_id)
            worker_outputs.append(db_result)
            summary_result = self._invoke_summary(actor_id, thread_id, query, worker_outputs)
            worker_outputs.append(summary_result)
            final_answer = summary_result.output
        elif route["route"] == "search_and_db_then_summary":
            search_result = self._invoke_search(actor_id, thread_id, query)
            db_result = self.db_agent.handle(query, actor_id)
            worker_outputs.extend([search_result, db_result])
            summary_result = self._invoke_summary(actor_id, thread_id, query, worker_outputs)
            worker_outputs.append(summary_result)
            final_answer = summary_result.output
        else:
            search_result = self._invoke_search(actor_id, thread_id, query)
            worker_outputs.append(search_result)
            summary_result = self._invoke_summary(actor_id, thread_id, query, worker_outputs)
            worker_outputs.append(summary_result)
            final_answer = summary_result.output

        return {
            "result": final_answer,
            "route": route["route"],
            "route_reason": route["reason"],
            "worker_outputs": [{"worker": item.worker, "output": item.output} for item in worker_outputs],
            "actor_id": actor_id,
            "thread_id": thread_id,
        }

