from typing import Any, Dict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

from a2a_common import build_agent_response, doc_to_match, search_docs
from groq_provider import build_groq_model

_ = load_dotenv()


def handle_search_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    request_id = payload.get("request_id", "unknown-request")
    query = payload.get("input", {}).get("query") or payload.get("prompt", "")
    top_k = int(payload.get("input", {}).get("top_k", 3))

    matches = [doc_to_match(doc, rank=i + 1) for i, doc in enumerate(search_docs(query, k=top_k))]

    if not matches:
        return build_agent_response(
            request_id=request_id,
            agent_name="search",
            status="success",
            result={
                "answer": "No relevant FAQ entries found for this request.",
                "matches": [],
            },
            confidence=0.2,
            sources=["lauki_qna.csv"],
            metadata={"task_type": payload.get("task_type", "faq_lookup")},
        )

    faq_context = "\n\n".join(
        [f"FAQ Match {item['rank']}:\n{item['content']}" for item in matches]
    )
    llm = build_groq_model()
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

    return build_agent_response(
        request_id=request_id,
        agent_name="search",
        status="success",
        result={
            "answer": response.content,
            "matches": matches,
        },
        confidence=0.9,
        sources=["lauki_qna.csv"],
        metadata={"task_type": payload.get("task_type", "faq_lookup")},
    )
