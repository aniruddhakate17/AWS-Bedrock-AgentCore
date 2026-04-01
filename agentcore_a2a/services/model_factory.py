import os

from bedrock_agentcore.identity.auth import requires_api_key
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model


MODEL_PROVIDER_NAME = os.getenv("BEDROCK_AGENTCORE_MODEL_PROVIDER_API_KEY_NAME", "")
MODEL_NAME = "openai/gpt-oss-20b"


@requires_api_key(provider_name=MODEL_PROVIDER_NAME)
def _agentcore_identity_api_key(api_key: str) -> str:
    return api_key


class GroqModelFactory:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name

    def get_api_key(self) -> str:
        if os.getenv("LOCAL_DEV") == "1":
            load_dotenv(".env.local")
            return os.getenv("GROQ_API_KEY", "")

        if MODEL_PROVIDER_NAME:
            return _agentcore_identity_api_key()

        load_dotenv()
        return os.getenv("GROQ_API_KEY", "")

    def build(self):
        return init_chat_model(
            model=self.model_name,
            model_provider="groq",
            api_key=self.get_api_key(),
        )

