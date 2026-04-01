import json
from typing import Any, Dict

from bedrock_agentcore_starter_toolkit.services.runtime import BedrockAgentCoreClient

from agentcore_a2a.services.request_response import RuntimeResponseParser


class AgentRuntimeInvoker:
    def __init__(self, region: str, parser: RuntimeResponseParser | None = None) -> None:
        self.region = region
        self.parser = parser or RuntimeResponseParser()

    def invoke(self, runtime_arn: str, payload: Dict[str, Any], target_agent: str, thread_id: str) -> Dict[str, Any]:
        client = BedrockAgentCoreClient(self.region)
        raw_response = client.invoke_endpoint(
            agent_arn=runtime_arn,
            payload=json.dumps(payload),
            session_id=self.parser.build_runtime_session_id(thread_id, target_agent, payload["request_id"]),
            endpoint_name="DEFAULT",
            user_id=payload["actor_id"],
        )
        return self.parser.parse_runtime_response(raw_response)
