import os
from typing import Callable

from bedrock_agentcore.identity.auth import requires_api_key
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

MODEL_PROVIDER_NAME = os.getenv("BEDROCK_AGENTCORE_MODEL_PROVIDER_API_KEY_NAME", "")
MODEL_NAME = "openai/gpt-oss-20b"


@requires_api_key(provider_name=MODEL_PROVIDER_NAME)
def _agentcore_identity_api_key(api_key: str) -> str:
    return api_key


def get_groq_api_key() -> str:
    """Resolve the Groq API key for local development or deployed AgentCore runtimes."""
    if os.getenv("LOCAL_DEV") == "1":
        load_dotenv(".env.local")
        return os.getenv("GROQ_API_KEY", "")

    if MODEL_PROVIDER_NAME:
        return _agentcore_identity_api_key()

    load_dotenv()
    return os.getenv("GROQ_API_KEY", "")


def build_groq_model():
    api_key = get_groq_api_key()
    return init_chat_model(
        model=MODEL_NAME,
        model_provider="groq",
        api_key=api_key,
    )


def with_groq_model(fn: Callable):
    """Small helper for call sites that want a fresh authenticated model instance."""
    def wrapper(*args, **kwargs):
        return fn(build_groq_model(), *args, **kwargs)

    return wrapper
