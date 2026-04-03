import uuid

from bedrock_agentcore.runtime import BedrockAgentCoreApp

from agentcore_a2a.container import ApplicationContainer


container = ApplicationContainer()


class SearchRuntimeApplication:
    def __init__(self) -> None:
        self.app = BedrockAgentCoreApp()

        @self.app.entrypoint
        def agent_invocation(payload, context):
            print("Search runtime payload:", payload)
            print("Search runtime context:", context)
            return container.search_agent.handle(payload)


class SummaryRuntimeApplication:
    def __init__(self) -> None:
        self.app = BedrockAgentCoreApp()

        @self.app.entrypoint
        def agent_invocation(payload, context):
            print("Summary runtime payload:", payload)
            print("Summary runtime context:", context)
            return container.summary_agent.handle(payload)


class SupervisorRuntimeApplication:
    def __init__(self) -> None:
        self.app = BedrockAgentCoreApp()

        @self.app.entrypoint
        def agent_invocation(payload, context):
            print("Supervisor runtime payload:", payload)
            print("Supervisor runtime context:", context)
            query = payload.get("prompt", "No prompt found in input")
            actor_id = payload.get("actor_id", "default-user")
            thread_id = payload.get("thread_id", payload.get("session_id", f"thread-{uuid.uuid4().hex}"))
            return container.supervisor_agent.handle(query, actor_id, thread_id)
