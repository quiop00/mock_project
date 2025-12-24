# Internal RAG Chatbot (LangGraph)

Minimal internal chatbot that always retrieves from your internal PDF/DOCX docs before answering. Multi-turn context is tracked per session. If no info is found, the bot replies with the required fallback.

## Features
- RAG-first: retrieve before every answer; no hallucinated info.
- Multi-turn: conversation history kept via LangGraph state.
- Vector store: FAISS (default) or Chroma; embeddings: OpenAI.
- Chunking: tiktoken-based, default 600/80 tokens chunk/overlap.
- LangSmith tracing: `@traceable` on nodes; default run metadata/tags when config not supplied.
- Fallback message enforced when retrieval has no context.

## Project Layout
- `src/internal_chatbot/`
  - `internal_chatbot.py` — LangGraph wiring for `chatbot_graph`
  - `internal_chatbot_state.py` — state schema and doc formatting
  - `internal_chatbot_config.py` — env/config defaults & system prompt
  - `internal_chatbot_retriever.py` — load PDF/DOCX, chunk, embed, vector store, retriever
  - `internal_chatbot_prompts.py` — answer prompt template
  - `internal_chatbot_nodes.py` — LangGraph nodes + LangSmith trace decorators
- `langgraph.json` — exposes `internal_chatbot` graph for `langgraph dev`

## Prerequisites
- Python 3.11–3.13
- OpenAI API key (for embeddings + chat model)
- (Optional) LangSmith account for tracing (`LANGSMITH_API_KEY`)

## Setup
1) Create and activate a virtual env (or use your own):
```bash
python3 -m venv .venv
source .venv/bin/activate
```

```CMD
.venv\Scripts\Activate.ps1
```

2) Install deps (requires network to fetch wheels):
```bash
python3 -m pip install -e .
```

3) Prepare `.env` in repo root:
```env
OPENAI_API_KEY=sk-...
# Optional: LangSmith tracing
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_PROJECT=internal-chatbot
# LANGSMITH_API_KEY=...
```

4) Add internal documents under `data/internal_docs/`:
- Place your `.pdf` and `.docx` files there. They will be loaded, chunked, embedded, and indexed at startup.

## Run with LangGraph Dev
```bash
langgraph dev
```
If There have error No module found, please run 'export PYTHONPATH=./src' first, then install the lib and run 'langgraph dev' again.

Then open the LangGraph Studio URL shown in the console. Typing the question

## Fallback Behavior
If retrieval returns no context, the bot replies:
```
I could not find this information in the internal documents. Would you like me to help in another way?
```

## LangSmith Tracing
- Nodes are decorated with `@traceable`; if you set `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_PROJECT=internal-chatbot` (plus `LANGSMITH_API_KEY` if needed), traces appear in LangSmith.
