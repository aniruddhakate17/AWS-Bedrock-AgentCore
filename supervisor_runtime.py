import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore_starter_toolkit.services.runtime import BedrockAgentCoreClient
import yaml

from a2a_common import (
    build_agent_request,
    build_runtime_session_id,
    parse_runtime_response,
)
from groq_provider import build_groq_model
from search_service import handle_search_request
from summary_service import handle_summary_request

_ = load_dotenv()

app = BedrockAgentCoreApp()
CONFIG_PATH = Path(__file__).with_name(".bedrock_agentcore.yaml")

REGION = os.getenv("A2A_REGION", os.getenv("AWS_REGION", "ap-south-1"))
SEARCH_RUNTIME_ARN = os.getenv("A2A_SEARCH_RUNTIME_ARN", "").strip()
SUMMARY_RUNTIME_ARN = os.getenv("A2A_SUMMARY_RUNTIME_ARN", "").strip()


def choose_route(query: str) -> Dict[str, Any]:
    llm = build_groq_model()
    routing_prompt = [
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

    raw = llm.invoke(routing_prompt).content.strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        lower_query = query.lower()
        if any(word in lower_query for word in ["my ", "account", "plan", "usage", "bill", "sql", "database"]):
            return {"route": "db_then_summary", "reason": "Fallback routing for account-specific question."}
        return {"route": "search_then_summary", "reason": "Fallback routing for knowledge question."}

    if parsed.get("route") not in {
        "search_only",
        "db_only",
        "search_then_summary",
        "db_then_summary",
        "search_and_db_then_summary",
    }:
        return {"route": "search_then_summary", "reason": "Invalid route from supervisor model."}
    return parsed


def run_db_agent(query: str, actor_id: str) -> Dict[str, Any]:
    llm = build_groq_model()
    response = llm.invoke(
        [
            SystemMessage(
                content="""You are DB Agent.

You handle structured account and database-style questions.

This runtime currently has no live database connection.
If asked for account-specific or SQL-backed information:
1. Clearly state that no live database is configured in this runtime.
2. Explain what kind of data would normally be retrieved.
3. Do not fabricate user/account data.
4. Keep the response short and operationally clear.
"""
            ),
            HumanMessage(
                content=(
                    f"User actor_id: {actor_id}\n"
                    f"Requested structured/account lookup: {query}\n"
                    "Respond as DB Agent."
                )
            ),
        ]
    )
    return {
        "worker": "db_agent",
        "output": response.content,
        "status": "success",
        "confidence": 0.5,
    }


def invoke_remote_runtime(runtime_arn: str, payload: Dict[str, Any], target_agent: str, thread_id: str) -> Dict[str, Any]:
    client = BedrockAgentCoreClient(REGION)
    raw = client.invoke_endpoint(
        agent_arn=runtime_arn,
        payload=json.dumps(payload),
        session_id=build_runtime_session_id(thread_id, target_agent, payload["request_id"]),
        endpoint_name="DEFAULT",
        user_id=payload["actor_id"],
    )
    return parse_runtime_response(raw)


def get_runtime_arn_from_config(agent_name: str) -> str:
    if not CONFIG_PATH.exists():
        return ""

    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        return config.get("agents", {}).get(agent_name, {}).get("bedrock_agentcore", {}).get("agent_arn", "") or ""
    except Exception:
        return ""


def invoke_search_agent(actor_id: str, thread_id: str, query: str) -> Dict[str, Any]:
    payload = build_agent_request(
        actor_id=actor_id,
        thread_id=thread_id,
        source_agent="supervisor",
        target_agent="search",
        task_type="faq_lookup",
        input_data={"query": query, "top_k": 3},
    )

    runtime_arn = SEARCH_RUNTIME_ARN or get_runtime_arn_from_config("search_agent")

    if runtime_arn:
        response = invoke_remote_runtime(runtime_arn, payload, "search", thread_id)
    else:
        response = handle_search_request(payload)

    return {
        "worker": "search_agent",
        "output": response.get("result", {}).get("answer", str(response)),
        "raw_response": response,
        "status": response.get("status", "success"),
        "confidence": response.get("confidence", 0.0),
    }


def invoke_summary_agent(actor_id: str, thread_id: str, user_query: str, worker_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary_input = [
        {"worker": item["worker"], "output": item["output"]}
        for item in worker_outputs
    ]
    payload = build_agent_request(
        actor_id=actor_id,
        thread_id=thread_id,
        source_agent="supervisor",
        target_agent="summary",
        task_type="summarize",
        input_data={"user_query": user_query, "worker_outputs": summary_input},
    )

    runtime_arn = SUMMARY_RUNTIME_ARN or get_runtime_arn_from_config("summary_agent")

    if runtime_arn:
        response = invoke_remote_runtime(runtime_arn, payload, "summary", thread_id)
    else:
        response = handle_summary_request(payload)

    return {
        "worker": "summary_agent",
        "output": response.get("result", {}).get("answer", str(response)),
        "raw_response": response,
        "status": response.get("status", "success"),
        "confidence": response.get("confidence", 0.0),
    }


def orchestrate_query(query: str, actor_id: str, thread_id: str) -> Dict[str, Any]:
    route = choose_route(query)
    worker_outputs: List[Dict[str, Any]] = []

    if route["route"] == "search_only":
        search_result = invoke_search_agent(actor_id, thread_id, query)
        worker_outputs.append(search_result)
        final_answer = search_result["output"]
    elif route["route"] == "db_only":
        db_result = run_db_agent(query, actor_id)
        worker_outputs.append(db_result)
        final_answer = db_result["output"]
    elif route["route"] == "db_then_summary":
        db_result = run_db_agent(query, actor_id)
        worker_outputs.append(db_result)
        summary_result = invoke_summary_agent(actor_id, thread_id, query, worker_outputs)
        worker_outputs.append(summary_result)
        final_answer = summary_result["output"]
    elif route["route"] == "search_and_db_then_summary":
        search_result = invoke_search_agent(actor_id, thread_id, query)
        db_result = run_db_agent(query, actor_id)
        worker_outputs.append(search_result)
        worker_outputs.append(db_result)
        summary_result = invoke_summary_agent(actor_id, thread_id, query, worker_outputs)
        worker_outputs.append(summary_result)
        final_answer = summary_result["output"]
    else:
        search_result = invoke_search_agent(actor_id, thread_id, query)
        worker_outputs.append(search_result)
        summary_result = invoke_summary_agent(actor_id, thread_id, query, worker_outputs)
        worker_outputs.append(summary_result)
        final_answer = summary_result["output"]

    return {
        "result": final_answer,
        "route": route["route"],
        "route_reason": route["reason"],
        "worker_outputs": [
            {"worker": item["worker"], "output": item["output"]}
            for item in worker_outputs
        ],
        "actor_id": actor_id,
        "thread_id": thread_id,
    }


@app.entrypoint
def agent_invocation(payload, context):
    print("Supervisor runtime payload:", payload)
    print("Supervisor runtime context:", context)

    query = payload.get("prompt", "No prompt found in input")
    actor_id = payload.get("actor_id", "default-user")
    thread_id = payload.get("thread_id", payload.get("session_id", f"thread-{uuid.uuid4().hex}"))

    return orchestrate_query(query, actor_id, thread_id)


if __name__ == "__main__":
    app.run()
