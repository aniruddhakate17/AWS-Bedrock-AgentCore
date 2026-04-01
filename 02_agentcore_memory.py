import csv
import json
import os
import re
import uuid
from collections import Counter
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.store.base import BaseStore

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from langgraph_checkpoint_aws import AgentCoreMemorySaver, AgentCoreMemoryStore

_ = load_dotenv()

app = BedrockAgentCoreApp()

REGION = "ap-south-1"
MEMORY_ID = "agentcore_myagent_memory-HPEmQF7XcU"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

checkpointer = AgentCoreMemorySaver(memory_id=MEMORY_ID)
store = AgentCoreMemoryStore(memory_id=MEMORY_ID)


def load_faq_csv(path: str) -> List[Document]:
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row["question"].strip()
            a = row["answer"].strip()
            docs.append(Document(page_content=f"Q: {q}\nA: {a}"))
    return docs


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


docs = load_faq_csv("./lauki_qna.csv")
doc_tokens = [Counter(tokenize(doc.page_content)) for doc in docs]


def extract_text(result: Dict[str, Any]) -> str:
    messages = result.get("messages", [])
    if not messages:
        return "No response generated."
    return messages[-1].content


def search_docs(query: str, k: int) -> List[Document]:
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    query_counts = Counter(query_tokens)
    scored_docs = []

    for doc, tokens in zip(docs, doc_tokens):
        overlap_score = sum(min(tokens[token], count) for token, count in query_counts.items())
        if overlap_score == 0:
            continue

        text = doc.page_content.lower()
        phrase_bonus = 3 if query.lower() in text else 0
        scored_docs.append((overlap_score + phrase_bonus, doc))

    scored_docs.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in scored_docs[:k]]


@tool
def search_faq(query: str) -> str:
    """Retrieve relevant FAQ entries for product, policy, activation, and support questions."""
    results = search_docs(query, k=3)
    if not results:
        return "No relevant FAQ entries found."

    context = "\n\n---\n\n".join(
        [f"FAQ Entry {i + 1}:\n{doc.page_content}" for i, doc in enumerate(results)]
    )
    return f"Found {len(results)} relevant FAQ entries:\n\n{context}"


@tool
def search_detailed_faq(query: str, num_results: int = 5) -> str:
    """Retrieve more FAQ context when the initial search is not enough."""
    results = search_docs(query, k=num_results)
    if not results:
        return "No relevant FAQ entries found."

    context = "\n\n---\n\n".join(
        [f"FAQ Entry {i + 1}:\n{doc.page_content}" for i, doc in enumerate(results)]
    )
    return f"Found {len(results)} detailed FAQ entries:\n\n{context}"


@tool
def reformulate_query(original_query: str, focus_aspect: str) -> str:
    """Search the FAQ from a specific angle such as pricing, activation, or troubleshooting."""
    reformulated = f"{focus_aspect} related to {original_query}"
    results = search_docs(reformulated, k=3)
    if not results:
        return f"No results found for aspect: {focus_aspect}"

    context = "\n\n---\n\n".join(
        [f"Entry {i + 1}:\n{doc.page_content}" for i, doc in enumerate(results)]
    )
    return f"Results for '{focus_aspect}' aspect:\n\n{context}"


llm = init_chat_model(
    model="openai/gpt-oss-20b",
    model_provider="groq",
    api_key=GROQ_API_KEY,
)

search_agent = create_agent(
    model=llm,
    tools=[search_faq, search_detailed_faq, reformulate_query],
    system_prompt="""You are Search Agent.

You specialize in retrieving useful information from the FAQ knowledge base.

Rules:
1. Use the FAQ tools to gather relevant evidence.
2. Prefer search_faq first and use search_detailed_faq only when needed.
3. Use reformulate_query when the question has multiple angles.
4. Return factual findings grounded in the retrieved FAQ entries.
5. Do not invent account-specific data or database results.
""",
)

def run_search_agent(query: str) -> str:
    result = search_agent.invoke({"messages": [("human", query)]})
    return extract_text(result)


def run_db_agent(query: str, actor_id: str) -> str:
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
    return response.content


def run_summary_agent(user_query: str, worker_outputs: List[Dict[str, str]]) -> str:
    findings = "\n\n".join(
        [f"[{item['worker']}]\n{item['output']}" for item in worker_outputs]
    )

    response = llm.invoke(
        [
            SystemMessage(
                content="""You are Summary Agent.

Your job is to synthesize outputs from specialist workers into one final answer.

Rules:
1. Use only the worker outputs you are given.
2. Prefer factual, grounded statements.
3. If a worker reports missing infrastructure, state that plainly.
4. Keep the answer concise and useful.
5. If worker outputs are incomplete, say what is missing instead of guessing.
"""
            ),
            HumanMessage(
                content=(
                    f"User question:\n{user_query}\n\n"
                    f"Worker outputs:\n{findings}\n\n"
                    "Create the final user-facing answer."
                )
            ),
        ]
    )
    return response.content


def choose_route(query: str) -> Dict[str, Any]:
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


class MemoryMiddleware(AgentMiddleware):
    def pre_model_hook(self, state: AgentState, config: RunnableConfig, *, store: BaseStore):
        actor_id = config["configurable"]["actor_id"]
        thread_id = config["configurable"]["thread_id"]

        namespace = (actor_id, thread_id)
        messages = state.get("messages", [])

        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                store.put(namespace, str(uuid.uuid4()), {"message": msg})
                break

        return {"messages": messages}

    def post_model_hook(self, state: AgentState, config: RunnableConfig, *, store: BaseStore):
        actor_id = config["configurable"]["actor_id"]
        thread_id = config["configurable"]["thread_id"]
        namespace = (actor_id, thread_id)

        messages = state.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                store.put(namespace, str(uuid.uuid4()), {"message": msg})
                break

        return state


@tool
def call_search_agent(query: str) -> str:
    """Call Search Agent for FAQ, policy, roaming, activation, pricing, and troubleshooting retrieval."""
    return run_search_agent(query)


@tool
def call_db_agent(query: str, actor_id: str = "default-user") -> str:
    """Call DB Agent for account-specific or structured data questions."""
    return run_db_agent(query, actor_id)


@tool
def call_summary_agent(user_query: str, collected_findings: str) -> str:
    """Call Summary Agent to combine specialist findings into one final user-facing answer."""
    worker_outputs = [{"worker": "collected_findings", "output": collected_findings}]
    return run_summary_agent(user_query, worker_outputs)


supervisor_agent = create_agent(
    model=llm,
    tools=[call_search_agent, call_db_agent, call_summary_agent],
    checkpointer=checkpointer,
    store=store,
    middleware=[MemoryMiddleware()],
    system_prompt="""You are Supervisor Agent.

You orchestrate specialist workers for this runtime.

Available workers:
- Search Agent: FAQ and knowledge retrieval
- DB Agent: account/database questions
- Summary Agent: combines worker outputs into the final answer

Rules:
1. Decide which worker or workers are needed.
2. If the request is domain/FAQ based, call Search Agent.
3. If the request is account-specific or structured-data based, call DB Agent.
4. If multiple worker outputs need combining, call Summary Agent at the end.
5. Do not fabricate database data.
6. Return a concise, useful final answer.
""",
)


def orchestrate_query(query: str, actor_id: str, thread_id: str) -> Dict[str, Any]:
    route = choose_route(query)
    worker_outputs: List[Dict[str, str]] = []

    if route["route"] == "search_only":
        search_output = run_search_agent(query)
        worker_outputs.append({"worker": "search_agent", "output": search_output})
        final_answer = search_output
    elif route["route"] == "db_only":
        db_output = run_db_agent(query, actor_id)
        worker_outputs.append({"worker": "db_agent", "output": db_output})
        final_answer = db_output
    elif route["route"] == "db_then_summary":
        db_output = run_db_agent(query, actor_id)
        worker_outputs.append({"worker": "db_agent", "output": db_output})
        final_answer = run_summary_agent(query, worker_outputs)
    elif route["route"] == "search_and_db_then_summary":
        search_output = run_search_agent(query)
        db_output = run_db_agent(query, actor_id)
        worker_outputs.append({"worker": "search_agent", "output": search_output})
        worker_outputs.append({"worker": "db_agent", "output": db_output})
        final_answer = run_summary_agent(query, worker_outputs)
    else:
        search_output = run_search_agent(query)
        worker_outputs.append({"worker": "search_agent", "output": search_output})
        final_answer = run_summary_agent(query, worker_outputs)

    return {
        "route": route["route"],
        "route_reason": route["reason"],
        "worker_outputs": worker_outputs,
        "final_answer": final_answer,
        "actor_id": actor_id,
        "thread_id": thread_id,
    }


@app.entrypoint
def agent_invocation(payload, context):
    """AgentCore entrypoint for the supervisor runtime."""
    print("Received payload:", payload)
    print("Context:", context)

    query = payload.get("prompt", "No prompt found in input")
    actor_id = payload.get("actor_id", "default-user")
    thread_id = payload.get("thread_id", payload.get("session_id", "default-session"))

    config = {"configurable": {"thread_id": thread_id, "actor_id": actor_id}}

    supervisor_result = supervisor_agent.invoke(
        {"messages": [("human", query)]},
        config=config,
    )
    print("Supervisor direct result:", extract_text(supervisor_result))

    orchestration = orchestrate_query(query, actor_id, thread_id)
    print("Orchestration result:", orchestration)

    return {
        "result": orchestration["final_answer"],
        "route": orchestration["route"],
        "route_reason": orchestration["route_reason"],
        "worker_outputs": orchestration["worker_outputs"],
        "actor_id": actor_id,
        "thread_id": thread_id,
    }


if __name__ == "__main__":
    app.run()
