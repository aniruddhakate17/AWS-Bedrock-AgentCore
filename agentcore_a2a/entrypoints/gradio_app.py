from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agentcore_a2a.ui.gradio_chat_app import GradioChatApplication

demo = GradioChatApplication().create_demo()


if __name__ == "__main__":
    demo.launch()
