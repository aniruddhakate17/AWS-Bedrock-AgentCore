# AgentCore A2A Demo

This repo now contains only the A2A-style AgentCore setup:

- `supervisor_runtime.py`
- `search_runtime.py`
- `summary_runtime.py`
- shared helpers:
  - `a2a_common.py`
  - `search_service.py`
  - `summary_service.py`
  - `groq_provider.py`
- `gradio_app.py` for a simple UI
- `lauki_qna.csv` as the FAQ dataset

## Runtimes

- `supervisor_agent_2`
  - user-facing runtime
  - routes requests to worker runtimes
- `search_agent`
  - FAQ retrieval runtime
- `summary_agent`
  - answer synthesis runtime

## Setup

Use Python `3.13` and `uv`.

```powershell
uv venv
.\.venv\Scripts\Activate.ps1
uv sync
```

Create `.env.local` with:

```env
GROQ_API_KEY=your_key_here
```

## Deploy

Deploy the worker runtimes first, then the supervisor:

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONUTF8='1'
chcp 65001 > $null

.\.venv\Scripts\agentcore.exe deploy -a search_agent
.\.venv\Scripts\agentcore.exe deploy -a summary_agent
.\.venv\Scripts\agentcore.exe deploy -a supervisor_agent_2
```

## Local UI

Run:

```powershell
.\.venv\Scripts\python.exe E:\AgentCore\gradio_app.py
```

The Gradio UI is pinned to `supervisor_agent_2`.

## Notes

- For direct CLI invokes against these runtimes, include `--user-id`.
- The toolkit reads provider API keys from `.env.local`, not `.env`.
- Search and summary are worker runtimes; the UI should talk only to `supervisor_agent_2`.
