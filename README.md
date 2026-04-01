# AgentCore A2A Application

This repository contains a production-style proof of concept for an Agent-to-Agent style application built on AWS Bedrock AgentCore.

The app has:

- a `supervisor` runtime that receives user requests
- a `search` runtime for FAQ retrieval
- a `summary` runtime for response synthesis
- a simple Gradio UI for local testing against the deployed supervisor runtime

The codebase is now organized into packages with class-based services, tools, and agents so it is easier to extend into a real application.

## Folder Structure

```text
E:\AgentCore
├── agentcore_a2a/
│   ├── agents/
│   │   ├── db_agent.py
│   │   ├── search_agent.py
│   │   ├── summary_agent.py
│   │   └── supervisor_agent.py
│   ├── services/
│   │   ├── model_factory.py
│   │   ├── request_response.py
│   │   └── runtime_invoker.py
│   ├── tools/
│   │   └── faq_search.py
│   ├── ui/
│   │   └── gradio_chat_app.py
│   ├── config.py
│   ├── container.py
│   ├── runtime_apps.py
│   └── schemas.py
│   ├── entrypoints/
│   │   ├── gradio_app.py
│   │   ├── search_runtime.py
│   │   ├── summary_runtime.py
│   │   └── supervisor_runtime.py
├── .bedrock_agentcore/
│   ├── search_agent/
│   ├── summary_agent/
│   └── supervisor_agent_2/
├── .bedrock_agentcore.yaml
├── .env.local
├── lauki_qna.csv
├── pyproject.toml
└── uv.lock
```

## Architecture

### 1. Supervisor Runtime

User requests first hit `supervisor_agent_2`.

The supervisor:

- classifies the request
- selects a route
- invokes worker runtimes
- merges results into a final response

Main class:

- `agentcore_a2a.agents.supervisor_agent.SupervisorAgent`

### 2. Search Runtime

The search runtime retrieves relevant FAQ entries from `lauki_qna.csv` using a lightweight lexical search tool and then asks the model to produce a grounded answer.

Main classes:

- `agentcore_a2a.tools.faq_search.FAQRepository`
- `agentcore_a2a.tools.faq_search.FAQSearchTool`
- `agentcore_a2a.agents.search_agent.SearchAgent`

### 3. Summary Runtime

The summary runtime takes worker outputs and synthesizes a concise user-facing answer.

Main class:

- `agentcore_a2a.agents.summary_agent.SummaryAgent`

### 4. DB Agent

The DB agent is currently a placeholder for future production integration.

Main class:

- `agentcore_a2a.agents.db_agent.DatabaseAgent`

It does not connect to a real database yet. It intentionally avoids hallucinating account-specific data.

## Class-Based Design

The application is assembled through a lightweight dependency container:

- `agentcore_a2a.container.ApplicationContainer`

That container wires:

- config loading
- model creation
- FAQ repository/tooling
- message builders
- runtime invocation
- agent classes

This makes the code easier to test and easier to extend with real database clients, monitoring, retries, or external APIs.

## Runtime Entry Points

The runtime entrypoints now live under:

- [agentcore_a2a/entrypoints/search_runtime.py](/e:/AgentCore/agentcore_a2a/entrypoints/search_runtime.py)
- [agentcore_a2a/entrypoints/summary_runtime.py](/e:/AgentCore/agentcore_a2a/entrypoints/summary_runtime.py)
- [agentcore_a2a/entrypoints/supervisor_runtime.py](/e:/AgentCore/agentcore_a2a/entrypoints/supervisor_runtime.py)

They delegate into:

- `agentcore_a2a.runtime_apps`

## UI Entry Point

The Gradio UI entrypoint is:

- [agentcore_a2a/entrypoints/gradio_app.py](/e:/AgentCore/agentcore_a2a/entrypoints/gradio_app.py)

It delegates into:

- `agentcore_a2a.ui.gradio_chat_app.GradioChatApplication`

## Local Setup

### Prerequisites

- Python `3.13`
- `uv`
- AWS CLI configured
- access to Bedrock AgentCore in your AWS account

### Create Environment

```powershell
uv venv
.\.venv\Scripts\Activate.ps1
uv sync
```

### Create Secrets File

Create `.env.local`:

```env
GROQ_API_KEY=your_groq_api_key
```

Important:

- the toolkit reads provider API keys from `.env.local`
- not from `.env`

## Deploy Order

Deploy worker runtimes first, then deploy the supervisor.

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONUTF8='1'
chcp 65001 > $null

.\.venv\Scripts\agentcore.exe deploy -a search_agent
.\.venv\Scripts\agentcore.exe deploy -a summary_agent
.\.venv\Scripts\agentcore.exe deploy -a supervisor_agent_2
```

## Run Gradio UI

```powershell
.\.venv\Scripts\python.exe E:\AgentCore\agentcore_a2a\entrypoints\gradio_app.py
```

The UI is pinned to:

- `supervisor_agent_2`

## Direct Invoke

When invoking deployed runtimes directly, include `--user-id`:

```powershell
$payload = @{
  prompt    = "What is roaming activation?"
  actor_id  = "test-user"
  thread_id = "thread-1234567890abcdef1234567890abcdef"
} | ConvertTo-Json -Compress

.\.venv\Scripts\agentcore.exe invoke --agent supervisor_agent_2 --user-id test-user $payload
```

## Example Test Prompts

### Search

- `What is roaming activation?`
- `How do I activate international roaming?`

### Search + Summary

- `Summarize how roaming activation works.`
- `Explain roaming charges in simple terms.`

### DB Placeholder

- `What is my current plan?`
- `What is my billing status?`

### Mixed

- `What is my current plan and does it include roaming?`
- `I am traveling tomorrow. Do I have roaming active and what are the charges?`

## Notes

- `search_agent` and `summary_agent` should remain stateless workers.
- `supervisor_agent_2` is the only runtime your frontend should call directly.
- The current DB agent is intentionally safe and non-connected.
- If you modify supervisor logic, redeploy `supervisor_agent_2`.
- If you modify retrieval or synthesis logic, redeploy the corresponding worker runtime too.

## Architecture

