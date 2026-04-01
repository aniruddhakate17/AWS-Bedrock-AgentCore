import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gradio as gr
from bedrock_agentcore_starter_toolkit.services.runtime import BedrockAgentCoreClient

from agentcore_a2a.config import AgentCoreConfigLoader, ProjectPaths, RuntimeConfig


class UiAuditLogger:
    def __init__(self, log_dir: Path) -> None:
        self.log_dir = log_dir
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / "gradio_agentcore_ui.log"
        self.audit_file = self.log_dir / "gradio_agentcore_ui.jsonl"

        self.logger = logging.getLogger("gradio_agentcore_ui")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        self.logger.addHandler(file_handler)

    def info(self, message: str, *args) -> None:
        self.logger.info(message, *args)

    def exception(self, message: str) -> None:
        self.logger.exception(message)

    def audit_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        def json_default(value: Any) -> str:
            if isinstance(value, bytes):
                return value.decode("utf-8", errors="replace")
            return str(value)

        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            **payload,
        }
        with self.audit_file.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False, default=json_default) + "\n")


class RuntimeResponseFormatter:
    @staticmethod
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

    @staticmethod
    def normalize_nested_result(parsed: Dict[str, Any]) -> Dict[str, Any]:
        current = parsed
        for _ in range(5):
            nested_result = current.get("result")
            if "route" in current or not isinstance(nested_result, str):
                break
            stripped = nested_result.strip()
            if not (stripped.startswith("{") and stripped.endswith("}")):
                break
            try:
                nested = json.loads(stripped, strict=False)
            except json.JSONDecodeError:
                break
            if isinstance(nested, dict):
                current = nested
            else:
                break

        if "route" not in current and isinstance(current.get("result"), str):
            wrapped = current["result"]
            result_match = re.search(r'"result"\s*:\s*"((?:\\.|[^"\\])*)"', wrapped, flags=re.DOTALL)
            route_match = re.search(r'"route"\s*:\s*"((?:\\.|[^"\\])*)"', wrapped)
            reason_match = re.search(r'"route_reason"\s*:\s*"((?:\\.|[^"\\])*)"', wrapped, flags=re.DOTALL)
            if result_match:
                current["result"] = bytes(result_match.group(1), "utf-8").decode("unicode_escape")
            if route_match:
                current["route"] = bytes(route_match.group(1), "utf-8").decode("unicode_escape")
            if reason_match:
                current["route_reason"] = bytes(reason_match.group(1), "utf-8").decode("unicode_escape")
        return current


class DeployedRuntimeGateway:
    def __init__(self, runtime_config: RuntimeConfig, audit_logger: UiAuditLogger) -> None:
        self.runtime_config = runtime_config
        self.audit_logger = audit_logger
        self.response_formatter = RuntimeResponseFormatter()

    def create_runtime_client(self) -> BedrockAgentCoreClient:
        return BedrockAgentCoreClient(self.runtime_config.region)

    def invoke(self, payload: Dict[str, Any], session: Dict[str, str]) -> Dict[str, Any]:
        request_id = uuid.uuid4().hex
        self.audit_logger.info(
            "Invoking runtime request_id=%s agent=%s actor_id=%s thread_id=%s",
            request_id,
            self.runtime_config.name,
            session["actor_id"],
            session["thread_id"],
        )
        self.audit_logger.audit_event(
            "request",
            {
                "request_id": request_id,
                "default_agent": self.runtime_config.name,
                "agent_arn": self.runtime_config.agent_arn,
                "region": self.runtime_config.region,
                "actor_id": session["actor_id"],
                "thread_id": session["thread_id"],
                "payload": payload,
            },
        )

        raw_response = self.create_runtime_client().invoke_endpoint(
            agent_arn=self.runtime_config.agent_arn,
            payload=json.dumps(payload),
            session_id=session["thread_id"],
            endpoint_name="DEFAULT",
            user_id=session["actor_id"],
        )
        parsed = self.response_formatter.parse_runtime_response(raw_response)
        parsed = self.response_formatter.normalize_nested_result(parsed)

        self.audit_logger.info(
            "Runtime response request_id=%s route=%s",
            request_id,
            parsed.get("route", "unknown"),
        )
        self.audit_logger.audit_event(
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


class GradioChatController:
    def __init__(self, runtime_gateway: DeployedRuntimeGateway, audit_logger: UiAuditLogger) -> None:
        self.runtime_gateway = runtime_gateway
        self.audit_logger = audit_logger

    def ensure_session(self, session: Dict[str, str] | None) -> Dict[str, str]:
        session = session or {}
        actor_id = session.get("actor_id") or f"user-{uuid.uuid4().hex[:8]}"
        thread_id = session.get("thread_id")
        if not thread_id or len(thread_id) < 33:
            thread_id = f"thread-{uuid.uuid4().hex}"
        return {"actor_id": actor_id, "thread_id": thread_id}

    def reset_session(self) -> Tuple[List[Dict[str, str]], Dict[str, str], str, str]:
        session = self.ensure_session({})
        self.audit_logger.info(
            "Started new chat session actor_id=%s thread_id=%s",
            session["actor_id"],
            session["thread_id"],
        )
        self.audit_logger.audit_event(
            "session_reset",
            {"actor_id": session["actor_id"], "thread_id": session["thread_id"]},
        )
        return [], session, session["actor_id"], session["thread_id"]

    def chat(
        self,
        message: str,
        history: List[Dict[str, str]] | None,
        session: Dict[str, str] | None,
    ) -> Tuple[str, List[Dict[str, str]], Dict[str, str], str, str]:
        history = history or []
        session = self.ensure_session(session)
        payload = {
            "prompt": message,
            "actor_id": session["actor_id"],
            "thread_id": session["thread_id"],
        }
        history.append({"role": "user", "content": message})

        try:
            result = self.runtime_gateway.invoke(payload, session)
            answer = result.get("result", "No response generated.")
            route = result.get("route", "unknown")
            route_reason = result.get("route_reason", "n/a")
            assistant_text = f"{answer}\n\nRoute: {route}\nRoute reason: {route_reason}"
        except Exception as exc:
            self.audit_logger.exception("Runtime invocation failed")
            self.audit_logger.audit_event(
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


class GradioChatApplication:
    def __init__(self, pinned_agent_name: str = "supervisor_agent_2") -> None:
        self.audit_logger = UiAuditLogger(ProjectPaths.LOG_DIR)
        runtime_config = AgentCoreConfigLoader().get_runtime(pinned_agent_name)
        self.runtime_gateway = DeployedRuntimeGateway(runtime_config, self.audit_logger)
        self.controller = GradioChatController(self.runtime_gateway, self.audit_logger)

    def create_demo(self) -> gr.Blocks:
        runtime_config = self.runtime_gateway.runtime_config
        audit_logger = self.audit_logger
        controller = self.controller

        with gr.Blocks(title="AgentCore Supervisor Chat") as demo:
            gr.Markdown(
                f"""
                # AgentCore Supervisor Chat
                This UI invokes the deployed AgentCore runtime `{runtime_config.name}` in `{runtime_config.region}`.
                Logs are written to `{audit_logger.log_file}` and `{audit_logger.audit_file}`.
                """
            )

            session_state = gr.State(controller.ensure_session(None))

            with gr.Row():
                actor_id_box = gr.Textbox(label="Actor ID", interactive=False)
                thread_id_box = gr.Textbox(label="Thread ID", interactive=False)

            with gr.Row():
                gr.Textbox(value=runtime_config.name, label="Runtime Name", interactive=False)
                gr.Textbox(value=runtime_config.region, label="AWS Region", interactive=False)

            chatbot = gr.Chatbot(label="Conversation", height=500)
            message_box = gr.Textbox(
                label="Message",
                placeholder="Ask about roaming, plans, charges, or account questions...",
            )

            with gr.Row():
                send_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("New Chat")

            demo.load(
                fn=lambda: controller.reset_session()[1:],
                inputs=None,
                outputs=[session_state, actor_id_box, thread_id_box],
            )

            send_btn.click(
                fn=controller.chat,
                inputs=[message_box, chatbot, session_state],
                outputs=[message_box, chatbot, session_state, actor_id_box, thread_id_box],
            )
            message_box.submit(
                fn=controller.chat,
                inputs=[message_box, chatbot, session_state],
                outputs=[message_box, chatbot, session_state, actor_id_box, thread_id_box],
            )

            clear_btn.click(
                fn=controller.reset_session,
                inputs=None,
                outputs=[chatbot, session_state, actor_id_box, thread_id_box],
            )

        return demo
