# Production-Grade Local RAG System

A complete, clean, reliable, and fact-grounded Retrieval-Augmented Generation (RAG) system built in Python using LangChain, ChromaDB, local Sentence-Transformers embeddings, and Ollama.

---

## Architecture

The project is structured with strict separation of concerns following clean software engineering principles:

```
rag_system/
│
├── app/
│   ├── ingestion/
│   │   ├── loader.py        # Scans directory and loads PDF/Markdown/Text files
│   │   ├── splitter.py      # Splits documents into semantic chunks with headings preserved
│   │   └── embedder.py      # Wraps sentence-transformers models locally
│   │
│   ├── retrieval/
│   │   └── retriever.py     # Performs similarity search and L2 distance relevance gating
│   │
│   ├── database/
│   │   └── chroma.py        # Manages the persistent disk-backed ChromaDB store
│   │
│   ├── llm/
│   │   └── client.py        # Swappable wrapper for LLM model communication
│   │
│   ├── prompts/
│   │   └── grounding_prompt.py # Strict system grounding instructions
│   │
│   └── pipeline.py          # Orchestrates Ingestion and Query pipelines
│
├── data/                    # Raw source documents to ingest (Markdown/PDF/Txt)
├── chroma_db/               # Persisted vector database folder
├── tests/                   # Complete pytest suite
├── .env.example             # Template for configurations
├── requirements.txt         # Package dependencies
└── main.py                  # CLI application entrypoint
```

---

## Anti-Hallucination Guardrails

To meet the strict **no-hallucination** requirement, the system implements a multi-layer guardrail:

1. **Relevance Gating**: Before sending anything to the LLM, the system computes the L2 distance of the retrieved chunks. If the best match distance is greater than the configured `RELEVANCE_THRESHOLD` (e.g. `1.2`), the system immediately refuses to answer *without* invoking the LLM.
2. **Context Isolation**: The user prompt and documents are separated using clear delimiters (`=== END OF CONTEXT ===`) to prevent instruction-leakage or prompt injection from retrieved documents.
3. **Strict Grounding System Prompt**: The system prompt forces the LLM to only answer from context, use the exact values, and respond with a specific refusal string if information is missing or incomplete.
4. **Programmatic Attribution**: Even if the LLM fails to list sources, the application appends validated source filenames and page numbers (if any) directly from ChromaDB's metadata to the footer of the response.

---

## Chunking & Embedding Justifications

- **Recursive Markdown Splitting**: Uses markdown-aware separators (`\n## `, `\n### `, `\n#### `, paragraph break, line break, word) so headings are the preferred split points and sections stay intact.
- **Chunk Size (512)**: Ideal size to keep complete technical steps/bullet points together in one context window while avoiding noisy metadata overlap.
- **Local Embedding (`all-MiniLM-L6-v2`)**: Extremely fast, runs entirely on your CPU/GPU, doesn't leak data to external APIs, and creates high-quality 384-dimension vector spaces.

---

## Setup Instructions

### 1. Install System Requirements
Make sure Python 3.9+ is installed.
Install [Ollama](https://ollama.ai/) and run it on your machine.
Pull the model you wish to use:
```bash
ollama pull llama3.2
```

### 2. Install Project Dependencies
Create a virtual environment and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure the Environment
Copy the example configuration file:
```bash
cp .env.example .env
```
Open `.env` and adjust the variables if needed. By default, it is configured for the local Ollama instance and local embeddings.

### Configuration Reference

All settings are read from environment variables (or from `.env`):

| Variable | Default | Description |
|---|---|---|
| `DATA_DIR` | `./data` | Folder of documents to ingest |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Where the vector database is stored |
| `CHROMA_COLLECTION` | `rag_knowledge_base` | ChromaDB collection name |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model |
| `CHUNK_SIZE` | `512` | Maximum characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between adjacent chunks |
| `TOP_K` | `5` | Chunks retrieved per query |
| `RELEVANCE_THRESHOLD` | `1.2` | Maximum L2 distance to accept a chunk |
| `LLM_PROVIDER` | `ollama` | `ollama` or `mock` |
| `OLLAMA_MODEL` | `llama3.2` | Ollama model tag |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |

### Running Without Ollama

Set `LLM_PROVIDER=mock` to run the full pipeline with a canned LLM. This is
useful for trying the ingestion and retrieval flow; embeddings still run locally.

---

## Usage Instructions

### 1. Ingest Data
To index all documents in the `data/` folder:
```bash
python main.py ingest
```

To wipe the existing database and re-ingest all documents from scratch:
```bash
python main.py ingest --reset
```

### 2. Query the Knowledge Base

**Interactive Mode (Query Loop)**
Start an interactive command-line session:
```bash
python main.py query
```

**Single-Shot Query**
Run a query directly from the shell:
```bash
python main.py query "How do I upgrade my subscription plan?"
```

---

## Testing

Run all unit and integration tests from inside the `rag_system/` directory:
```bash
python -m pytest tests/ -v
```
