# ChefMind AI 🍴

> **Culinary RAG Agent powered by IBM watsonx**
>
> Interprets your recipe library, adapts recipes to your dietary constraints,
> and delivers step-by-step cooking instructions — all grounded in your actual documents.

---

## Architecture

```
data/recipes/          ← your recipe .txt / .md files
      │
      ▼
  ingest.py            ← chunk → embed (ibm/slate-125m-english-rtrvr) → Chroma DB
      │
      ▼
  retriever.py         ← semantic search: query → top-K recipe chunks
      │
      ▼
  prompt_builder.py    ← inject chunks + user constraints into ChefMind system prompt
      │
      ▼
  chefmind.py          ← call IBM Granite (ibm/granite-13b-chat-v2) via watsonx.ai SDK
      │
      ▼
  app.py               ← CLI entrypoint (interactive or argument-based)
```

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | — |
| IBM Cloud account | Free tier works |
| watsonx.ai project | Create one at [cloud.ibm.com](https://cloud.ibm.com) |
| IBM Cloud API key | IAM → API Keys |

---

## Quick Start

### 1. Clone & install dependencies

```bash
git clone <your-repo>
cd chefmind_ai
pip install -r requirements.txt
```

### 2. Configure credentials

```bash
cp .env.example .env
# Edit .env — fill in WATSONX_API_KEY, WATSONX_URL, WATSONX_PROJECT_ID
```

### 3. Add your recipe documents

Drop any `.txt` or `.md` recipe files into `data/recipes/`.
Two sample recipes (chocolate chip cookies, bolognese) are included.

### 4. Ingest (build the vector index)

```bash
python ingest.py
```

To wipe and rebuild:
```bash
python ingest.py --reset
```

### 5. Run ChefMind AI

**Interactive mode** (guided prompts):
```bash
python app.py
```

**One-shot CLI mode:**
```bash
python app.py \
  --question "Make the chocolate chip cookies keto and sugar-free" \
  --ingredients "almond flour, eggs, butter, erythritol, dark chocolate 85%, vanilla" \
  --restrictions "keto, sugar-free, gluten-free" \
  --constraints "45 minutes, standard oven"
```

---

## File Reference

| File | Purpose |
|---|---|
| [`config.py`](config.py) | Central settings — credentials, model IDs, chunk sizes |
| [`ingest.py`](ingest.py) | Load → chunk → embed → store documents |
| [`retriever.py`](retriever.py) | Semantic search over the Chroma vector store |
| [`prompt_builder.py`](prompt_builder.py) | ChefMind system prompt template + injection |
| [`chefmind.py`](chefmind.py) | Pipeline orchestrator (retrieve → prompt → generate) |
| [`app.py`](app.py) | CLI entrypoint |
| [`data/recipes/`](data/recipes/) | Your recipe knowledge base |
| [`.env.example`](.env.example) | Credentials template |

---

## Configuration Reference (`config.py`)

| Variable | Default | Description |
|---|---|---|
| `WATSONX_MODEL_ID` | `ibm/granite-13b-chat-v2` | Foundation model for generation |
| `EMBEDDING_MODEL_ID` | `ibm/slate-125m-english-rtrvr` | Embedding model for retrieval |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Local vector store path |
| `CHUNK_SIZE` | `512` | Characters per text chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `TOP_K_RESULTS` | `4` | Recipe chunks retrieved per query |

---

## Notes

- **Text generation API deprecation:** As of February 2026, IBM deprecated the
  `/ml/v1/text/generation` endpoint. `chefmind.py` uses the **chat API**
  (`model.chat()`) which is the recommended path going forward.
- **Adding more recipes:** Just drop files in `data/recipes/` and re-run `ingest.py`.
  You do not need to touch any other code.
- **Switching models:** Set `WATSONX_MODEL_ID` in your `.env` to any
  [supported watsonx foundation model](https://cloud.ibm.com/docs/watsonx-ai).
