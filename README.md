# Personal RAG assistant

An LLM assistant that answers questions grounded in live web, market, and
sports data, with a self-refreshing knowledge base and a periodic
fine-tuning loop — both driven by CI/CD.

## How "always current" actually works

Two different mechanisms cover two different kinds of freshness:

| Need | Mechanism | Latency |
|---|---|---|
| Stock/crypto price, live score, "what's happening right now" | Live connectors (`app/connectors/`), called at query time straight from the source API | Seconds — genuinely real-time |
| General knowledge, news, background context | ChromaDB knowledge base, refreshed by `.github/workflows/ingest.yml` on a cron | Minutes (default: every 15 min) |

No system indexes "the entire web" every second — not even Google does
that. The working version of "every second" is: hit live APIs directly
for anything time-sensitive, and keep a background knowledge base fresh
enough that it's never seriously stale.

## How the model "learns" over time

1. `ingest.yml` (frequent) pulls fresh content into the vector store.
2. `train.yml` (weekly, GPU required) turns recently-ingested chunks into
   Q&A training examples (`dataset_builder.py`), fine-tunes a LoRA
   adapter on top of the base model (`train_lora.py` — QLoRA, so only a
   few MB gets trained, not the full 7B parameters), and evaluates it
   against a fixed regression set (`evaluate.py`).
3. Only if evaluation passes does `promote_adapter.py` flip the "active
   adapter" pointer that `llm_engine.py` reads — so a bad fine-tune never
   reaches production, and rolling back is just re-pointing a file.

Worth repeating from the chat: retrieval is already how the assistant
learns new *facts*. Fine-tuning here mostly teaches the model how to use
retrieved context well — tone, reasoning habits, domain conventions — not
new knowledge. Both are wired up since that's what was asked for, but
they're doing different jobs.

## Architecture

Streamlit (frontend) calls FastAPI (backend), which merges live connector
results with ChromaDB retrieval results into one context block, then
hands that to the LLM (Mistral-7B-Instruct + whichever LoRA adapter is
currently active) to generate a grounded, cited answer.

## Requirements and honest costs

- Serving works on CPU (slow) or GPU (fast); roughly 6GB VRAM is enough
  for 4-bit inference of the 7B base model.
- Fine-tuning (`train_lora.py`) needs an actual CUDA GPU — GitHub's free
  hosted runners don't have one. Point `train.yml` at a self-hosted GPU
  runner, or swap that job for a trigger to a cloud GPU box (RunPod,
  Lambda, or a scheduled Colab job).
- Free API keys worth getting: [Tavily](https://tavily.com) for web
  search. TheSportsDB and CoinGecko work with no key. yfinance needs no
  key either.
- CI installs the full `backend/requirements.txt` (including torch) even
  for lightweight tests, to keep the setup simple — expect the first CI
  run to take a few minutes. Split a leaner `requirements-test.txt` later
  if that gets annoying.

## Setup

```bash
git clone <your-repo> && cd ai-assistant
cp .env.example .env   # add your TAVILY_API_KEY

pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt

# Terminal 1 — backend (run from repo root)
uvicorn app.main:app --reload --app-dir backend

# Terminal 2 — frontend
streamlit run frontend/streamlit_app.py
```

Or via Docker: `docker compose up --build` (CPU only by default — see the
commented GPU block in `docker-compose.yml`).

Useful one-off commands (run from the repo root):

```bash
# Refresh the knowledge base manually
PYTHONPATH=backend python -m app.ingestion.crawler

# Build a fine-tuning dataset from what's been ingested
PYTHONPATH=backend python -m app.training.dataset_builder

# Fine-tune (needs a CUDA GPU)
PYTHONPATH=backend python -m app.training.train_lora

# Evaluate, then promote if it passes
PYTHONPATH=backend python -m app.training.evaluate --adapter models/adapters/candidate
PYTHONPATH=backend python -m app.training.promote_adapter
```

## Extending it

- Add topics to `backend/app/ingestion/watchlist.json` to widen what the
  background knowledge base tracks.
- Add a new connector under `backend/app/connectors/` and wire it into
  `router.py`'s keyword rules (or replace those rules with an LLM-based
  intent classifier once a fine-tuned model is available to spare the
  extra call).
- Swap `BASE_MODEL_ID` in `.env` for any Hugging Face causal LM that fits
  your GPU — Qwen2.5-7B-Instruct is a solid ungated alternative to
  Mistral.
- If you decide you don't actually want in-house fine-tuning, swap
  `llm_engine.py` to call a local Ollama server instead — much simpler,
  but then `train.yml` has nowhere to plug in (Ollama serves GGUF files,
  not PEFT adapters directly).
- Responses are non-streaming for now; adding token streaming from
  `llm_engine.py` through FastAPI to Streamlit is a natural next step.

## Project structure

```
ai-assistant/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app + router registration
│   │   ├── config.py            # env-driven settings
│   │   ├── api/
│   │   │   ├── routes_chat.py   # POST /chat
│   │   │   └── routes_health.py # GET /health
│   │   ├── core/
│   │   │   ├── rag_pipeline.py  # orchestrates retrieval + generation
│   │   │   ├── llm_engine.py    # loads base model + active LoRA adapter
│   │   │   ├── embeddings.py    # sentence-transformers wrapper
│   │   │   └── vector_store.py  # ChromaDB wrapper
│   │   ├── connectors/
│   │   │   ├── web_search.py    # Tavily live search
│   │   │   ├── finance.py       # yfinance + CoinGecko
│   │   │   ├── sports.py        # TheSportsDB
│   │   │   └── router.py        # picks which connector(s) to call
│   │   ├── ingestion/
│   │   │   ├── crawler.py       # scheduled KB refresh job
│   │   │   ├── chunker.py       # text splitting
│   │   │   ├── indexer.py       # embed + upsert into Chroma
│   │   │   └── watchlist.json   # topics the KB stays current on
│   │   └── training/
│   │       ├── dataset_builder.py  # fresh chunks -> Q&A training pairs
│   │       ├── train_lora.py       # QLoRA fine-tuning
│   │       ├── evaluate.py         # pass/fail gate for a candidate adapter
│   │       └── promote_adapter.py  # flips the "active adapter" pointer
│   ├── tests/
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── streamlit_app.py
│   ├── requirements.txt
│   └── Dockerfile
├── data/                  # vector store + generated training sets
├── models/adapters/       # versioned LoRA adapters
├── .github/workflows/
│   ├── ingest.yml          # cron: refresh knowledge base
│   ├── train.yml           # cron: fine-tune -> evaluate -> promote
│   └── ci.yml               # tests on every push/PR
├── docker-compose.yml
├── .env.example
└── README.md
```
