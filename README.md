# AWS Bedrock AgentCore Multi-Agent Demo

This project is a lightweight FAQ assistant built on AWS Bedrock AgentCore.

It includes:

- a deployed AgentCore runtime entrypoint
- a supervisor-style multi-agent flow inside the runtime
- internal worker agents for search, summary, and DB-style queries
- AgentCore memory support with `actor_id` and `thread_id`
- a local Gradio UI that can call the deployed AgentCore runtime

The codebase is intentionally kept lightweight so it can fit within Bedrock AgentCore free-tier style image constraints.

## Architecture

The main deployed runtime is:

- [02_agentcore_memory.py](/e:/AgentCore/02_agentcore_memory.py)

That file currently hosts:

- `Supervisor Agent`
- `Search Agent`
- `Summary Agent`
- `DB Agent` placeholder
- AgentCore memory integration
- the AgentCore runtime entrypoint

Other important files:

- [00_langgraph_agent.py](/e:/AgentCore/00_langgraph_agent.py): simple local FAQ agent example
- [01_agentcore_runtime.py](/e:/AgentCore/01_agentcore_runtime.py): lightweight single-agent AgentCore runtime
- [gradio_app.py](/e:/AgentCore/gradio_app.py): local Gradio chat UI for the deployed runtime
- [lauki_qna.csv](/e:/AgentCore/lauki_qna.csv): FAQ knowledge source
- [pyproject.toml](/e:/AgentCore/pyproject.toml): project dependencies
- [.bedrock_agentcore.yaml](/e:/AgentCore/.bedrock_agentcore.yaml): AgentCore runtime configuration

## Prerequisites

Install these before starting:

- Python `3.13`
- `uv`
- AWS CLI
- access to AWS Bedrock AgentCore in your AWS account

Optional but helpful:

- PowerShell on Windows
- Docker or Podman if you plan to build/deploy runtimes locally through the toolkit

## 1. Clone And Enter The Project

```powershell
git clone <your-repo-url>
cd E:\AgentCore
```

## 2. Create The Environment

If you do not already have a `.venv`, create one:

```powershell
uv venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

## 3. Install Dependencies

Install everything from the lockfile:

```powershell
uv sync
```

If the lockfile needs to be refreshed first:

```powershell
uv lock
uv sync
```

Why `uv sync` matters:

- it installs the exact project dependencies from [uv.lock](/e:/AgentCore/uv.lock)
- it avoids dependency drift across machines
- it ensures the same packages are used for local testing and deployment packaging

## 4. Configure Environment Variables

Create or update [.env](/e:/AgentCore/.env).

At minimum, set:

```env
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_optional_huggingface_token_here
```

Notes:

- `GROQ_API_KEY` is required for the LLM calls used by the runtime.
- `HF_TOKEN` is optional in the current lightweight version.
- Do not commit real secrets to source control.

## 5. Configure AWS Access

Make sure AWS CLI is configured for the same account where your AgentCore runtime exists:

```powershell
aws configure
```

Then verify identity:

```powershell
aws sts get-caller-identity
```

You should see the correct AWS account ID.

Important:

- launch the app from the same terminal where AWS credentials work
- if you use SSO or temporary credentials, refresh them before starting the UI
- if credentials expire while the Gradio app is open, restart the app after re-authenticating

## 6. Verify Local Python

Always run this project from the local `.venv`.

Check the interpreter:

```powershell
.\.venv\Scripts\python.exe -c "import sys; print(sys.executable)"
```

It should point to:

```text
E:\AgentCore\.venv\Scripts\python.exe
```

## 7. Run The Gradio UI

Start the local UI:

```powershell
.\.venv\Scripts\python.exe gradio_app.py
```

Then open the URL shown in the terminal, usually:

```text
http://127.0.0.1:7860
```

The Gradio UI:

- reads [.bedrock_agentcore.yaml](/e:/AgentCore/.bedrock_agentcore.yaml)
- finds the `default_agent`
- calls the deployed AgentCore runtime over AWS
- keeps `actor_id` and `thread_id` per chat
- logs every request and response

## 8. Log Files

The Gradio UI writes logs to:

- [logs/gradio_agentcore_ui.log](/e:/AgentCore/logs/gradio_agentcore_ui.log)
- [logs/gradio_agentcore_ui.jsonl](/e:/AgentCore/logs/gradio_agentcore_ui.jsonl)

Use these to debug:

- runtime invocation errors
- route selection
- payload/response behavior
- session issues

## 9. How The Multi-Agent Runtime Works

Inside [02_agentcore_memory.py](/e:/AgentCore/02_agentcore_memory.py):

- `Supervisor Agent` decides which worker(s) to use
- `Search Agent` handles FAQ retrieval from the CSV knowledge base
- `Summary Agent` combines findings into a user-facing answer
- `DB Agent` currently acts as a safe placeholder and does not query a live database

Typical routes:

- `search_only`
- `db_only`
- `search_then_summary`
- `db_then_summary`
- `search_and_db_then_summary`

## 10. Suggested Test Questions

Search-focused questions:

- `What is roaming activation?`
- `Are international roaming packs available?`
- `How do I activate international roaming?`
- `What is the billing cycle for postpaid?`

DB-style questions:

- `What is my current plan?`
- `What is my billing status?`
- `Do I have any unpaid invoices?`

Mixed routing questions:

- `I am traveling tomorrow. Do I have roaming active and what are the charges?`
- `What is my current plan, and does it include roaming?`
- `Summarize my billing status and explain roaming charges.`

Expected behavior:

- Search questions should hit Search or Search + Summary
- DB questions should hit the DB route
- mixed questions should hit multiple workers

## 11. Deploying To Bedrock AgentCore

This repo uses [.bedrock_agentcore.yaml](/e:/AgentCore/.bedrock_agentcore.yaml) as the AgentCore runtime config file.

The current default runtime is:

- `agentcore_multiagent`

Its entrypoint is:

- [02_agentcore_memory.py](/e:/AgentCore/02_agentcore_memory.py)

Typical workflow:

1. Update code
2. Refresh dependencies if needed:

```powershell
uv lock
uv sync
```

3. Deploy using the AgentCore toolkit/CLI you already configured
4. Confirm the runtime status
5. Test invocation

If your local toolkit is installed, the common commands are typically along these lines:

```powershell
agentcore status
agentcore invoke '{"prompt":"What is roaming activation?"}'
```

Use the runtime already defined in [.bedrock_agentcore.yaml](/e:/AgentCore/.bedrock_agentcore.yaml) rather than creating a brand-new one unless you intend to.

## 12. Important Note About This Repo Config

The checked-in [.bedrock_agentcore.yaml](/e:/AgentCore/.bedrock_agentcore.yaml) currently contains runtime IDs, ARNs, ECR repositories, and AWS-account-specific values from an existing environment.

A new developer should treat those values as environment-specific.

Before deploying from a new AWS account, update:

- AWS account ID
- region
- runtime IDs and ARNs
- execution roles
- ECR repositories
- memory IDs

Do not assume these values will work in another AWS account unchanged.

## 13. Common Problems

### `AccessDeniedException` with signature mismatch

This usually means one of these:

- AWS credentials expired
- wrong AWS profile or terminal session
- Gradio app started in a shell that does not match your working AWS identity
- stale app process still running

What to do:

1. Open a new terminal
2. Activate `.venv`
3. Run:

```powershell
aws sts get-caller-identity
```

4. Start the app again with:

```powershell
.\.venv\Scripts\python.exe gradio_app.py
```

### `ServiceQuotaExceededException` during deploy

This project previously hit AgentCore image size limits.

To reduce image size, the repo was intentionally changed to remove heavyweight ML dependencies like:

- `torch`
- `sentence-transformers`
- `faiss-cpu`
- Hugging Face embedding/vector-store packages

The current retrieval path uses lightweight lexical FAQ search instead.

### `python` command uses the wrong interpreter

Always prefer:

```powershell
.\.venv\Scripts\python.exe
```

rather than relying on system `python`.

## 14. Project Dependency Notes

The current project intentionally keeps dependencies small:

- `bedrock-agentcore`
- `bedrock-agentcore-starter-toolkit`
- `gradio`
- `langchain-groq`
- `langchain[aws]`
- `langgraph`
- `langgraph-checkpoint-aws`

If you add large ML packages again, the AgentCore deployment image may become too large.

## 15. Recommended First-Time Setup Checklist

For a new developer, the safest sequence is:

1. Install Python `3.13`
2. Install `uv`
3. Install AWS CLI
4. Clone the repo
5. Run `uv venv`
6. Activate `.venv`
7. Run `uv sync`
8. Create `.env`
9. Run `aws sts get-caller-identity`
10. Launch `gradio_app.py`
11. Test a few prompts
12. Only then attempt AgentCore deployment

## 16. Useful Commands

Refresh dependencies:

```powershell
uv lock
uv sync
```

Compile-check key files:

```powershell
python -m py_compile 00_langgraph_agent.py
python -m py_compile 01_agentcore_runtime.py
python -m py_compile 02_agentcore_memory.py
python -m py_compile gradio_app.py
```

Run the Gradio UI:

```powershell
.\.venv\Scripts\python.exe gradio_app.py
```

Check AWS identity:

```powershell
aws sts get-caller-identity
```

## 17. Next Improvements

Possible follow-up improvements:

- move supervisor/worker logic out of [02_agentcore_memory.py](/e:/AgentCore/02_agentcore_memory.py) into separate modules
- replace the DB placeholder agent with a real read-only database integration
- add log viewing directly inside the Gradio UI
- clean up the runtime response shape so the UI does not need fallback parsing

