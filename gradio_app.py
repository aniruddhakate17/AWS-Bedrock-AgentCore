import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gradio as gr
import yaml
from bedrock_agentcore_starter_toolkit.services.runtime import BedrockAgentCoreClient


BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / ".bedrock_agentcore.yaml"
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "gradio_agentcore_ui.log"
AUDIT_FILE = LOG_DIR / "gradio_agentcore_ui.jsonl"


logger = logging.getLogger("gradio_agentcore_ui")
logger.setLevel(logging.INFO)
logger.handlers.clear()

file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(file_handler)


def audit_event(event_type: str, payload: Dict[str, Any]) -> None:
    def json_default(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        **payload,
    }
    with AUDIT_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=json_default) + "\n")


def load_agentcore_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise RuntimeError(f"Missing config file: {CONFIG_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    default_agent = data.get("default_agent")
    agents = data.get("agents", {})
    if not default_agent or default_agent not in agents:
        raise RuntimeError("default_agent is missing or invalid in .bedrock_agentcore.yaml")

    agent_cfg = agents[default_agent]
    agent_arn = agent_cfg.get("bedrock_agentcore", {}).get("agent_arn")
    region = agent_cfg.get("aws", {}).get("region")

    if not agent_arn or not region:
        raise RuntimeError("Agent ARN or region missing in .bedrock_agentcore.yaml")

    return {
        "default_agent": default_agent,
        "agent_arn": agent_arn,
        "region": region,
    }


RUNTIME_CONFIG = load_agentcore_config()


def create_runtime_client() -> BedrockAgentCoreClient:
    return BedrockAgentCoreClient(RUNTIME_CONFIG["region"])


def ensure_session(session: Dict[str, str] | None) -> Dict[str, str]:
    if session is None:
        session = {}

    actor_id = session.get("actor_id") or f"user-{uuid.uuid4().hex[:8]}"
    thread_id = session.get("thread_id")
    if not thread_id or len(thread_id) < 33:
        thread_id = f"thread-{uuid.uuid4().hex}"
    return {"actor_id": actor_id, "thread_id": thread_id}


def reset_session() -> Tuple[List[Dict[str, str]], Dict[str, str], str, str]:
    session = ensure_session({})
    logger.info("Started new chat session actor_id=%s thread_id=%s", session["actor_id"], session["thread_id"])
    audit_event(
        "session_reset",
        {"actor_id": session["actor_id"], "thread_id": session["thread_id"]},
    )
    return [], session, session["actor_id"], session["thread_id"]


def parse_runtime_response(response: Dict[str, Any]) -> Dict[str, Any]:
    def unwrap_json(value: Any, max_depth: int = 3) -> Any:
        current = value
        for _ in range(max_depth):
            if isinstance(current, str):
                try:
                    current = json.loads(current, strict=False)
                    continue
                except json.JSONDecodeError:
                    return current
            if isinstance(current, dict) and isinstance(current.get("result"), str):
                try:
                    current["result"] = unwrap_json(current["result"], max_depth=max_depth - 1)
                except Exception:
                    pass
            return current
        return current

    def normalize(value: Any) -> Dict[str, Any]:
        value = unwrap_json(value)
        if isinstance(value, dict):
            nested_result = value.get("result")
            if isinstance(nested_result, dict):
                merged = {**nested_result}
                for key, item in value.items():
                    if key != "result" and key not in merged:
                        merged[key] = item
                return merged
            return value

        if isinstance(value, str):
            return {"result": value}

        return {"result": str(value)}

    raw_response = response.get("response")

    if isinstance(raw_response, list):
        joined = "\n".join(str(item) for item in raw_response if item is not None)
        return normalize(joined)

    return normalize(raw_response)


def invoke_deployed_runtime(payload: Dict[str, Any], session: Dict[str, str]) -> Dict[str, Any]:
    request_id = uuid.uuid4().hex
    logger.info(
        "Invoking runtime request_id=%s agent=%s actor_id=%s thread_id=%s",
        request_id,
        RUNTIME_CONFIG["default_agent"],
        session["actor_id"],
        session["thread_id"],
    )
    audit_event(
        "request",
        {
            "request_id": request_id,
            "default_agent": RUNTIME_CONFIG["default_agent"],
            "agent_arn": RUNTIME_CONFIG["agent_arn"],
            "region": RUNTIME_CONFIG["region"],
            "actor_id": session["actor_id"],
            "thread_id": session["thread_id"],
            "payload": payload,
        },
    )

    runtime_client = create_runtime_client()

    raw_response = runtime_client.invoke_endpoint(
        agent_arn=RUNTIME_CONFIG["agent_arn"],
        payload=json.dumps(payload),
        session_id=session["thread_id"],
        endpoint_name="DEFAULT",
        user_id=session["actor_id"],
    )
    parsed = parse_runtime_response(raw_response)
    for _ in range(5):
        nested_result = parsed.get("result")
        if "route" in parsed or not isinstance(nested_result, str):
            break
        stripped = nested_result.strip()
        if not (stripped.startswith("{") and stripped.endswith("}")):
            break
        try:
            nested = json.loads(stripped, strict=False)
        except json.JSONDecodeError:
            break
        if isinstance(nested, dict):
            parsed = nested
        else:
            break

    if "route" not in parsed and isinstance(parsed.get("result"), str):
        wrapped = parsed["result"]
        result_match = re.search(r'"result"\s*:\s*"((?:\\.|[^"\\])*)"', wrapped, flags=re.DOTALL)
        route_match = re.search(r'"route"\s*:\s*"((?:\\.|[^"\\])*)"', wrapped)
        reason_match = re.search(r'"route_reason"\s*:\s*"((?:\\.|[^"\\])*)"', wrapped, flags=re.DOTALL)
        if result_match:
            answer = bytes(result_match.group(1), "utf-8").decode("unicode_escape")
            parsed["result"] = answer
        if route_match:
            parsed["route"] = bytes(route_match.group(1), "utf-8").decode("unicode_escape")
        if reason_match:
            parsed["route_reason"] = bytes(reason_match.group(1), "utf-8").decode("unicode_escape")

    logger.info(
        "Runtime response request_id=%s route=%s",
        request_id,
        parsed.get("route", "unknown"),
    )
    audit_event(
        "response",
        {
            "request_id": request_id,
            "actor_id": session["actor_id"],
            "thread_id": session["thread_id"],
            "raw_response": raw_response,
            "parsed_response": parsed,
        },
    )
    return parsed


def chat(
    message: str,
    history: List[Dict[str, str]] | None,
    session: Dict[str, str] | None,
) -> Tuple[str, List[Dict[str, str]], Dict[str, str], str, str]:
    history = history or []
    session = ensure_session(session)

    payload = {
        "prompt": message,
        "actor_id": session["actor_id"],
        "thread_id": session["thread_id"],
    }

    history.append({"role": "user", "content": message})

    try:
        result = invoke_deployed_runtime(payload, session)
        answer = result.get("result", "No response generated.")
        route = result.get("route", "unknown")
        route_reason = result.get("route_reason", "n/a")
        assistant_text = f"{answer}\n\nRoute: {route}\nRoute reason: {route_reason}"
    except Exception as exc:
        logger.exception("Runtime invocation failed")
        audit_event(
            "error",
            {
                "actor_id": session["actor_id"],
                "thread_id": session["thread_id"],
                "message": str(exc),
                "exception_type": type(exc).__name__,
                "payload": payload,
            },
        )
        assistant_text = f"Invocation failed: {type(exc).__name__}: {exc}"

    history.append({"role": "assistant", "content": assistant_text})
    return "", history, session, session["actor_id"], session["thread_id"]


with gr.Blocks(title="AgentCore Supervisor Chat") as demo:
    gr.Markdown(
        f"""
        # AgentCore Supervisor Chat
        This UI invokes the deployed AgentCore runtime `{RUNTIME_CONFIG["default_agent"]}` in `{RUNTIME_CONFIG["region"]}`.
        Logs are written to `{LOG_FILE}` and `{AUDIT_FILE}`.
        """
    )

    session_state = gr.State(ensure_session(None))

    with gr.Row():
        actor_id_box = gr.Textbox(label="Actor ID", interactive=False)
        thread_id_box = gr.Textbox(label="Thread ID", interactive=False)

    with gr.Row():
        gr.Textbox(value=RUNTIME_CONFIG["default_agent"], label="Runtime Name", interactive=False)
        gr.Textbox(value=RUNTIME_CONFIG["region"], label="AWS Region", interactive=False)

    chatbot = gr.Chatbot(label="Conversation", height=500)
    message_box = gr.Textbox(
        label="Message",
        placeholder="Ask about roaming, plans, charges, or account questions...",
    )

    with gr.Row():
        send_btn = gr.Button("Send", variant="primary")
        clear_btn = gr.Button("New Chat")

    demo.load(
        fn=lambda: reset_session()[1:],
        inputs=None,
        outputs=[session_state, actor_id_box, thread_id_box],
    )

    send_btn.click(
        fn=chat,
        inputs=[message_box, chatbot, session_state],
        outputs=[message_box, chatbot, session_state, actor_id_box, thread_id_box],
    )
    message_box.submit(
        fn=chat,
        inputs=[message_box, chatbot, session_state],
        outputs=[message_box, chatbot, session_state, actor_id_box, thread_id_box],
    )

    clear_btn.click(
        fn=reset_session,
        inputs=None,
        outputs=[chatbot, session_state, actor_id_box, thread_id_box],
    )


if __name__ == "__main__":
    demo.launch()
