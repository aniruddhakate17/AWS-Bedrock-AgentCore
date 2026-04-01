import csv
import json
import re
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document


BASE_DIR = Path(__file__).parent
FAQ_PATH = BASE_DIR / "lauki_qna.csv"


def load_faq_csv(path: Path = FAQ_PATH) -> List[Document]:
    docs = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = row["question"].strip()
            answer = row["answer"].strip()
            docs.append(Document(page_content=f"Q: {question}\nA: {answer}"))
    return docs


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


FAQ_DOCS = load_faq_csv()
FAQ_DOC_TOKENS = [Counter(tokenize(doc.page_content)) for doc in FAQ_DOCS]


def search_docs(query: str, k: int = 3) -> List[Document]:
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    query_counts = Counter(query_tokens)
    scored_docs = []

    for doc, tokens in zip(FAQ_DOCS, FAQ_DOC_TOKENS):
        overlap_score = sum(min(tokens[token], count) for token, count in query_counts.items())
        if overlap_score == 0:
            continue

        text = doc.page_content.lower()
        phrase_bonus = 3 if query.lower() in text else 0
        scored_docs.append((overlap_score + phrase_bonus, doc))

    scored_docs.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in scored_docs[:k]]


def doc_to_match(doc: Document, rank: int) -> Dict[str, Any]:
    return {
        "rank": rank,
        "content": doc.page_content,
    }


def build_agent_request(
    *,
    actor_id: str,
    thread_id: str,
    source_agent: str,
    target_agent: str,
    task_type: str,
    input_data: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "request_id": request_id or uuid.uuid4().hex,
        "actor_id": actor_id,
        "thread_id": thread_id,
        "source_agent": source_agent,
        "target_agent": target_agent,
        "task_type": task_type,
        "input": input_data,
        "context": context or {},
    }


def build_agent_response(
    *,
    request_id: str,
    agent_name: str,
    status: str,
    result: Dict[str, Any],
    confidence: float,
    sources: Optional[List[str]] = None,
    error: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "request_id": request_id,
        "agent_name": agent_name,
        "status": status,
        "result": result,
        "confidence": confidence,
        "sources": sources or [],
        "error": error,
        "metadata": metadata or {},
    }


def unwrap_json_like(value: Any, max_depth: int = 5) -> Any:
    current = value
    for _ in range(max_depth):
        if isinstance(current, str):
            stripped = current.strip()
            if not stripped:
                return current
            try:
                current = json.loads(stripped, strict=False)
                continue
            except json.JSONDecodeError:
                return current
        if isinstance(current, dict) and isinstance(current.get("result"), str):
            nested = current["result"].strip()
            if nested.startswith("{") and nested.endswith("}"):
                try:
                    current["result"] = json.loads(nested, strict=False)
                except json.JSONDecodeError:
                    pass
        return current
    return current


def parse_runtime_response(response: Dict[str, Any]) -> Dict[str, Any]:
    raw_response = response.get("response")
    if isinstance(raw_response, list):
        joined = "\n".join(str(item) for item in raw_response if item is not None)
        parsed = unwrap_json_like(joined)
    else:
        parsed = unwrap_json_like(raw_response)

    if isinstance(parsed, dict):
        nested_result = parsed.get("result")
        if isinstance(nested_result, dict):
            merged = {**nested_result}
            for key, value in parsed.items():
                if key != "result" and key not in merged:
                    merged[key] = value
            return merged
        return parsed

    return {"result": str(parsed)}


def build_runtime_session_id(thread_id: str, target_agent: str, request_id: str) -> str:
    base = f"{thread_id}-{target_agent}-{request_id}"
    if len(base) >= 33:
        return base[:128]
    return f"{base}-{uuid.uuid4().hex}"

