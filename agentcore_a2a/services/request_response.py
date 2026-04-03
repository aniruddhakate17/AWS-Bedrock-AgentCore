import json
import uuid
from typing import Any, Dict, Optional

from agentcore_a2a.schemas import AgentRequest, AgentResponse


class AgentMessageFactory:
    def build_request(
        self,
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
        return AgentRequest(
            actor_id=actor_id,
            thread_id=thread_id,
            source_agent=source_agent,
            target_agent=target_agent,
            task_type=task_type,
            input=input_data,
            context=context or {},
            request_id=request_id or uuid.uuid4().hex,
        ).to_dict()

    def build_response(
        self,
        *,
        request_id: str,
        agent_name: str,
        status: str,
        result: Dict[str, Any],
        confidence: float,
        sources: Optional[list[str]] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return AgentResponse(
            request_id=request_id,
            agent_name=agent_name,
            status=status,
            result=result,
            confidence=confidence,
            sources=sources or [],
            error=error,
            metadata=metadata or {},
        ).to_dict()


class RuntimeResponseParser:
    def unwrap_json_like(self, value: Any, max_depth: int = 5) -> Any:
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

    def parse_runtime_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        raw_response = response.get("response")
        if isinstance(raw_response, list):
            joined = "\n".join(str(item) for item in raw_response if item is not None)
            parsed = self.unwrap_json_like(joined)
        else:
            parsed = self.unwrap_json_like(raw_response)

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

    @staticmethod
    def build_runtime_session_id(thread_id: str, target_agent: str, request_id: str) -> str:
        base = f"{thread_id}-{target_agent}-{request_id}"
        if len(base) >= 33:
            return base[:128]
        return f"{base}-{uuid.uuid4().hex}"

