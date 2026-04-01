from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass(frozen=True)
class RuntimeConfig:
    name: str
    agent_arn: str
    region: str


class ProjectPaths:
    ROOT = Path(__file__).resolve().parent.parent
    CONFIG_PATH = ROOT / ".bedrock_agentcore.yaml"
    FAQ_PATH = ROOT / "lauki_qna.csv"
    LOG_DIR = ROOT / "logs"


class AgentCoreConfigLoader:
    def __init__(self, config_path: Path | None = None) -> None:
        self.config_path = config_path or ProjectPaths.CONFIG_PATH

    def load(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise RuntimeError(f"Missing config file: {self.config_path}")

        with self.config_path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}

    def get_runtime(self, agent_name: str) -> RuntimeConfig:
        config = self.load()
        agents = config.get("agents", {})
        agent_cfg = agents.get(agent_name)
        if not agent_cfg:
            raise RuntimeError(f"Agent {agent_name!r} is missing in {self.config_path}")

        agent_arn = agent_cfg.get("bedrock_agentcore", {}).get("agent_arn")
        region = agent_cfg.get("aws", {}).get("region")
        if not agent_arn or not region:
            raise RuntimeError(f"Agent ARN or region missing for {agent_name!r}")

        return RuntimeConfig(name=agent_name, agent_arn=agent_arn, region=region)

