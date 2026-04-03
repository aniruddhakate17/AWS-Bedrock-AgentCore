from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agentcore_a2a.runtime_apps import SupervisorRuntimeApplication

app = SupervisorRuntimeApplication().app


if __name__ == "__main__":
    app.run()
