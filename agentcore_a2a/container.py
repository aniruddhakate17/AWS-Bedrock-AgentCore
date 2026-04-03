from agentcore_a2a.agents.db_agent import DatabaseAgent
from agentcore_a2a.agents.search_agent import SearchAgent
from agentcore_a2a.agents.summary_agent import SummaryAgent
from agentcore_a2a.agents.supervisor_agent import SupervisorAgent
from agentcore_a2a.config import AgentCoreConfigLoader, ProjectPaths
from agentcore_a2a.services.model_factory import GroqModelFactory
from agentcore_a2a.services.request_response import AgentMessageFactory, RuntimeResponseParser
from agentcore_a2a.services.runtime_invoker import AgentRuntimeInvoker
from agentcore_a2a.tools.faq_search import FAQRepository, FAQSearchTool


class ApplicationContainer:
    def __init__(self) -> None:
        self.config_loader = AgentCoreConfigLoader()
        self.message_factory = AgentMessageFactory()
        self.model_factory = GroqModelFactory()
        self.response_parser = RuntimeResponseParser()
        self.faq_repository = FAQRepository(ProjectPaths.FAQ_PATH)
        self.faq_search_tool = FAQSearchTool(self.faq_repository)
        self.search_agent = SearchAgent(self.faq_search_tool, self.model_factory, self.message_factory)
        self.summary_agent = SummaryAgent(self.model_factory, self.message_factory)
        self.db_agent = DatabaseAgent(self.model_factory)
        region = self.config_loader.get_runtime("supervisor_agent_2").region
        self.runtime_invoker = AgentRuntimeInvoker(region=region, parser=self.response_parser)
        self.supervisor_agent = SupervisorAgent(
            model_factory=self.model_factory,
            message_factory=self.message_factory,
            config_loader=self.config_loader,
            runtime_invoker=self.runtime_invoker,
            search_agent=self.search_agent,
            summary_agent=self.summary_agent,
            db_agent=self.db_agent,
        )

