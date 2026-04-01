from agentcore_a2a.schemas import WorkerResult
from agentcore_a2a.services.model_factory import GroqModelFactory


class DatabaseAgent:
    def __init__(self, model_factory: GroqModelFactory) -> None:
        self.model_factory = model_factory

    def handle(self, query: str, actor_id: str) -> WorkerResult:
        llm = self.model_factory.build()
        response = llm.invoke(
            [
                (
                    "system",
                    """You are DB Agent.

You handle structured account and database-style questions.

This runtime currently has no live database connection.
If asked for account-specific or SQL-backed information:
1. Clearly state that no live database is configured in this runtime.
2. Explain what kind of data would normally be retrieved.
3. Do not fabricate user/account data.
4. Keep the response short and operationally clear.
""",
                ),
                (
                    "human",
                    f"User actor_id: {actor_id}\nRequested structured/account lookup: {query}\nRespond as DB Agent.",
                ),
            ]
        )
        return WorkerResult(worker="db_agent", output=response.content, status="success", confidence=0.5)
