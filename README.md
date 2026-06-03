# Sage

**AI assistant with intelligent tool calling — knows when to think, calculate, or search.**

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)
![LangChain](https://img.shields.io/badge/LangChain-agent-yellow)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?logo=openai)

---

## Overview

Sage is an AI assistant built on a LangChain ReAct agent powered by GPT-4o-mini. It autonomously decides — for every message — whether to answer directly from its training data, delegate a calculation to a safe math engine, or search the web for real-time information. Responses are delivered via Server-Sent Events (SSE), enabling true streaming without polling.

The core design challenge is routing accuracy: making the agent reach for a tool only when needed, and pick the *right* tool when it does. That requires carefully crafted tool descriptions, explicit routing rules in the system prompt, and a clear separation of concerns between the three paths.

---

## How It Works — Decision Logic

```
User Question
     |
LangChain Agent (GPT-4o-mini)
     | analyzes intent
     |-- General knowledge --> Direct LLM response
     |-- Mathematical      --> Calculator tool (numexpr)
     |-- Current/real-time --> Web Search (DuckDuckGo)
     |
SSE Streaming Response
```

The agent uses a ReAct (Reason + Act) loop. On each turn it produces a `Thought`, decides on an `Action`, observes the result, and iterates until it produces a `Final Answer`. This loop is capped at 3 iterations to prevent runaway execution.

**Why the routing is reliable:**

- The system prompt contains explicit, prioritized routing rules that map question categories to paths.
- Each tool's description is written to *exclude* the other tools' domains — not just describe what it does.
- The boundary cases are explicit. "Who was Einstein?" routes direct (stable knowledge) while "Latest AI news?" routes to search (temporal). The distinction is encoded in the tool descriptions, not in code.

Adding a new tool requires no routing code changes — just a new `@tool` function with a well-written description and the agent adapts.

---

## Tech Stack

| Component  | Technology        | Why                                                                 |
|------------|-------------------|---------------------------------------------------------------------|
| LLM        | GPT-4o-mini       | Cost-effective, fast, strong tool-calling and instruction-following |
| Framework  | LangChain         | Industry-standard agent orchestration with ReAct support            |
| Calculator | numexpr           | Safe math evaluation — no `eval()`, sandboxed to numeric ops only  |
| Web Search | DuckDuckGo        | No API key required, zero setup friction for evaluators             |
| API        | FastAPI           | Async-native, auto-docs at `/docs`, first-class Pydantic integration|
| Streaming  | SSE               | Real-time token delivery, standard protocol, works with plain curl  |

---

## Getting Started

### Prerequisites

- Python 3.11+
- OpenAI API key

### Installation

```bash
git clone https://github.com/guantunes17/sage.git
cd sage
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### Running

```bash
python -m app.main
# Server starts at http://localhost:8000
# Interactive API docs at http://localhost:8000/docs
```

---

## API Endpoints

### `POST /chat` — Chat with Sage (SSE streaming)

```bash
# Direct response (general knowledge)
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Quem foi Albert Einstein?"}'

# Calculator
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Quanto é 128 vezes 46?"}'

# Web search
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Quais são as últimas notícias sobre inteligência artificial?"}'
```

Each response is a stream of SSE events:

```
event: token
data: {"event":"token","content":"Albert Einstein was...","tool_used":null,"finish_reason":null}

event: done
data: {"event":"done","content":null,"tool_used":null,"finish_reason":"stop"}
```

### `GET /health` — Health check

```bash
curl http://localhost:8000/health
# {"status":"healthy","app_name":"Sage","version":"1.0.0"}
```

### `GET /tools` — List available tools

```bash
curl http://localhost:8000/tools
# {"tools":[{"name":"calculator","description":"..."},{"name":"web_search","description":"..."}]}
```

---

## Architecture

```
sage/
├── app/
│   ├── config.py     # Environment loading, constants, logging setup
│   ├── schemas.py    # Pydantic v2 request/response models
│   ├── tools.py      # Tool definitions: calculator (numexpr) + DuckDuckGo search
│   ├── agent.py      # LangChain ReAct agent, system prompt, streaming generator
│   └── main.py       # FastAPI app: endpoints, SSE streaming, error handlers
├── .env.example
├── requirements.txt
└── README.md
```

- **`config.py`** — Loads `.env`, validates `OPENAI_API_KEY` at startup (fails fast), configures logging format and level, exposes constants.
- **`schemas.py`** — All Pydantic v2 models including `ChatEvent` (the SSE payload), `ChatRequest` (with whitespace-stripping validator), and error/health/tools response types.
- **`tools.py`** — `@tool`-decorated functions. The calculator uses `numexpr.evaluate()` for safe math. The web search wraps `DuckDuckGoSearchRun` with error handling. `get_tools()` and `get_tools_info()` are the only public exports.
- **`agent.py`** — Builds the `AgentExecutor` with `create_react_agent`, the system prompt, temperature=0, and `max_iterations=3`. `run_agent_stream()` is an async generator that yields `(event_type, content, tool_used)` tuples consumed by the SSE endpoint.
- **`main.py`** — Three endpoints (`/chat`, `/health`, `/tools`), CORS middleware, startup logging, and exception handlers for validation errors, HTTP errors, and unhandled exceptions.

---

## Design Decisions

**1. `numexpr` over `eval()`**
`eval()` executes arbitrary Python code — a serious security risk in a web-facing API. `numexpr` is restricted to numeric expressions and cannot import modules, call functions outside its scope, or access the filesystem. This is the correct choice even in a prototype.

**2. DuckDuckGo over paid alternatives (Tavily, Serper, etc.)**
Sage needs exactly one API key to run: OpenAI's. Eliminating search API keys removes a setup barrier for anyone cloning the repo. DuckDuckGo's LangChain integration handles rate limits and parsing cleanly, making it the right tool for this scope.

**3. SSE over WebSocket**
The `/chat` interaction is unidirectional: the client sends one message and receives a stream of tokens. SSE maps directly to this pattern. It requires no handshake, works over HTTP/1.1, survives proxies and firewalls, and is natively testable with `curl -N`. WebSockets would add bidirectional complexity with no benefit here.

**4. Structured SSE events with metadata**
Each SSE frame carries a `ChatEvent` JSON payload that includes `tool_used` and `finish_reason` — not just raw text. This gives clients full observability: they know when a tool was used, which one, and why the stream ended, without parsing the text content.

**5. System prompt routing over code-level routing**
Tool selection is entirely driven by the LLM reading its system prompt and tool descriptions — there are no `if/elif` routing blocks in the code. This scales linearly: adding a third tool means writing one `@tool` function and one description. The routing logic lives in the prompt, where it belongs.

---

## What I'd Do Differently (With More Time)

- **Docker + docker-compose**: A single `docker compose up` command for full reproducibility across environments, with no local Python setup required.
- **Tests with pytest**: Unit tests for each tool function (including error paths), integration tests for the API endpoints against a mocked agent, and routing tests to validate the system prompt's decision-making with representative inputs.
- **Conversation memory**: `ConversationBufferWindowMemory` for multi-turn context within a session — so Sage can answer follow-up questions that reference previous turns.
- **Observability with LangSmith**: Trace every agent run — which tools were considered, which were invoked, token counts, and latency per step. Essential for debugging routing failures in production.
- **Rate limiting**: Per-IP throttling on `/chat` to prevent API key abuse, using a library like `slowapi`.
- **Frontend**: A Streamlit or lightweight React interface for a visual demo that shows the streaming response and the tool routing decision in real time.
- **Additional tools**: The architecture supports adding tools without touching routing logic. Natural next candidates: a URL reader/summarizer, a translation tool, and a code execution sandbox.

---

## What I Learned

Working on Sage reinforced how central tool descriptions are to agent behavior. The LLM cannot read intent from code — it reads natural language. Getting the calculator to activate for "What is 15% of 300?" but not for "What is the meaning of life?" required writing tool descriptions that actively exclude the wrong cases, not just describe the right ones. The routing is not a function of the framework; it's a function of the prompts.

Implementing SSE streaming with FastAPI's async generators was more nuanced than expected. The key insight is that the generator must never raise an uncaught exception mid-stream — the HTTP response has already started with a 200 status, so there's no way to send a 500 afterward. Error handling has to happen *inside* the generator, formatted as an error SSE event, so the client always receives a graceful termination signal.

The `numexpr` vs `eval()` decision was the clearest example of production awareness mattering at the prototype stage. It would be easy to ship `eval()` in a challenge and no one would notice. But the habit of reaching for the safe alternative by default — not when explicitly required — is what separates code written for evaluation from code written for production.

---

## License

MIT — see [LICENSE](LICENSE).
