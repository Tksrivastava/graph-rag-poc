# Graph RAG POC

A **Proof-of-Concept** for Graph Retrieval-Augmented Generation (Graph RAG) applied to the **LME (London Metal Exchange) aluminum supply chain** domain. This project demonstrates how to extract a knowledge graph from raw news articles, persist it in Neo4j, and query it using natural language — all powered by LangGraph, Ollama, and Groq.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
  - [Stage 1 — Chunking the Corpus](#stage-1--chunking-the-corpus)
  - [Stage 2 — Knowledge Graph Extraction](#stage-2--knowledge-graph-extraction)
  - [Stage 3 — Populating Neo4j](#stage-3--populating-neo4j)
  - [Stage 4 — Graph Visualization](#stage-4--graph-visualization)
  - [Stage 5 — Graph RAG Query Pipeline](#stage-5--graph-rag-query-pipeline)
- [LangGraph Workflow](#langgraph-workflow)
- [Core Module Reference](#core-module-reference)
- [Dataset](#dataset)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Pipeline](#running-the-pipeline)
- [Interactive Graph](#interactive-graph)
- [Dependencies](#dependencies)
- [Author](#author)
- [License](#license)

---

## Overview

Traditional RAG systems retrieve document chunks using vector similarity. **Graph RAG** instead builds a **knowledge graph** from the source corpus — capturing entities and the relationships between them — and uses the graph structure as the retrieval backbone.

This POC applies that idea to LME aluminum market news:

1. Raw news articles are chunked into small text fragments.
2. A local LLM (Ollama / `qwen2.5:3b`) extracts structured entities and relationships from each chunk, producing a knowledge graph in JSONL format.
3. The graph is ingested into **Neo4j AuraDB** (cloud-hosted).
4. At query time, a **LangGraph** pipeline converts a natural-language question into a **Cypher query**, runs it against Neo4j to find relevant chunk IDs, retrieves those chunks, and passes them to **Groq** (`llama-3.3-70b-versatile`) for a final grounded answer.

---

## Architecture

```
Raw News Articles (JSONL)
        │
        ▼
[ extract-chunk-corpus.py ]
  RecursiveCharacterTextSplitter
        │
        ▼
   chunks.jsonl  (1,428 chunks)
        │
        ▼
[ extract-kg.py ]
  Ollama (qwen2.5:3b) — local LLM
  Structured KG extraction (nodes + relationships)
        │
        ▼
  extracted-kg.jsonl  (1,428 KG records)
        │
        ▼
[ create-graph-db.py ]
  Neo4j AuraDB ingestion
        │
        ▼
    Neo4j Graph DB
        │
        ▼
[ GraphRAGPOC — LangGraph workflow ]
  1. Natural language query
  2. Cypher query generation (Groq)
  3. Neo4j execution → chunk IDs
  4. Chunk retrieval from chunks.jsonl
  5. Final answer generation (Groq)
        │
        ▼
     Answer
```

---

## Project Structure

```
graph-rag-poc/
│
├── core/                          # Reusable Python package
│   ├── __init__.py
│   ├── graph.py                   # Main GraphRAGPOC class + LangGraph workflow
│   ├── logging.py                 # Centralized LoggerFactory
│   ├── prompt.py                  # All LLM prompt templates
│   └── utils.py                   # DocumentProcess & Neo4jOps helpers
│
├── dataset/                       # Data files (tracked in git)
│   ├── news-articles-raw.jsonl    # 65 raw LME aluminum news articles
│   ├── chunks.jsonl               # 1,428 text chunks (derived from raw articles)
│   └── extracted-kg.jsonl         # 1,428 KG records (nodes + relationships per chunk)
│
├── development/                   # One-off pipeline scripts
│   ├── extract-chunk-corpus.py    # Stage 1: Chunk raw articles
│   ├── extract-kg.py              # Stage 2: Extract KG from chunks via Ollama
│   ├── create-graph-db.py         # Stage 3: Ingest KG into Neo4j
│   └── viz-graph.py               # Stage 4: Generate interactive HTML graph
│
├── interactive-graph.html         # Pre-generated PyVis graph visualization
├── main.ipynb                     # End-to-end Jupyter notebook demo
├── .env.poc.example               # Environment variable template
├── .gitignore
├── pyproject.toml                 # Package metadata (PEP 517)
└── requirements.txt               # Pinned dependencies
```

---

## How It Works

### Stage 1 — Chunking the Corpus

**Script:** `development/extract-chunk-corpus.py`

Reads `dataset/news-articles-raw.jsonl` (65 raw articles) and splits each article into smaller text chunks using LangChain's `RecursiveCharacterTextSplitter`.

- Chunk size and overlap are controlled via `.env.poc` (`CHUNK_SIZE`, `CHUNK_OVERLAP`).
- Default configuration: `CHUNK_SIZE=200`, `CHUNK_OVERLAP=0`.
- Output: `dataset/chunks.jsonl` — **1,428 chunks**, each stored as a LangChain `Document` (with `page_content` and `metadata` including article title, publish date, and source URL).

### Stage 2 — Knowledge Graph Extraction

**Script:** `development/extract-kg.py`

Iterates over all 1,428 chunks and uses a **locally running Ollama model** (`qwen2.5:3b` by default) with `format="json"` to extract a structured knowledge graph from each chunk.

The LLM is guided by two prompt templates (defined in `core/prompt.py`):

- **`SystemPrompt`** — Instructs the model to extract aluminum supply chain KGs with strict vocabulary and normalisation rules (e.g., `"lme"` → `"london metal exchange"`, `"aluminium"` → `"aluminum"`).
- **`UserPrompt`** — Provides the chunk text and a JSON schema to fill in with nodes and relationships.

Each extracted record is validated against Pydantic models (`Node`, `Relationship`, `GraphResponse`) and appended to `dataset/extracted-kg.jsonl`.

**KG Schema:**
```json
{
  "chunk_id": 0,
  "nodes": [
    {"id": "aluminum_price", "type": "financial_metric", "properties": {"value": 2985.5, "unit": "usd/t"}}
  ],
  "relationships": [
    {"source": "london_metal_exchange", "target": "aluminum_price", "type": "reports", "properties": {}}
  ]
}
```

**Entity types used:** `exchange`, `financial_metric`, `commodity`, `location`, `event`, `policy`, etc.  
**Relationship verbs used:** `reports`, `trades_on`, `affects`, `produces`, `exports_to`, `disrupts`, etc.

A 5-second delay (`time.sleep(5)`) is inserted between chunks to avoid overloading the local Ollama server.

### Stage 3 — Populating Neo4j

**Script:** `development/create-graph-db.py`

Reads `dataset/extracted-kg.jsonl` and pushes all nodes and relationships into a **Neo4j AuraDB** instance using the `Neo4jOps` helper class.

Key implementation details:

- All entities are stored under a single node label: `Entity`.
- Nested property dictionaries are **flattened** (e.g., `{"value": 100, "unit": "usd/t"}` becomes `value_value=100`, `value_unit="usd/t"`) for Neo4j compatibility.
- The `chunk_id` is attached to every node and relationship as a property, enabling the RAG pipeline to trace graph hits back to source text chunks.
- Nodes are created with `MERGE` to avoid duplicates.

### Stage 4 — Graph Visualization

**Script:** `development/viz-graph.py`

Queries all edges from Neo4j (`MATCH (a)-[r]->(b) RETURN a.id, b.id`) and builds an interactive HTML graph using **NetworkX** and **PyVis**.

- Physics simulation is enabled (ForceAtlas2 layout).
- Full interactivity: drag nodes, zoom, hover, adjust physics/nodes/edges via control panel.
- Output: `interactive-graph.html` (self-contained, opens in browser automatically).

### Stage 5 — Graph RAG Query Pipeline

**Class:** `core/graph.py → GraphRAGPOC`

The main inference class. Instantiated with a Groq API key, Neo4j driver, Groq model name, and path to the chunks file. Internally builds and compiles a **LangGraph** state machine.

---

## LangGraph Workflow

The query pipeline is a four-node directed acyclic graph (DAG) with no conditional edges:

```
[get_cypher_query] → [get_valid_chunks] → [get_context] → [get_final_response]
```

**State object (`GraphRAGState`):**

| Field | Type | Description |
|---|---|---|
| `natural_query` | `str` | The original user question |
| `cypher_query` | `Optional[str]` | Generated Cypher query |
| `valid_chunks` | `Optional[List[int]]` | Chunk IDs returned from Neo4j |
| `context` | `Optional[List[str]]` | Text of relevant chunks |
| `final_response` | `Optional[str]` | Final LLM answer |

**Node descriptions:**

**`get_cypher_query`**  
Introspects the live Neo4j schema (node labels, relationship types, sample properties) and feeds it — along with the user's question — to the `CypherPrompt`. Groq (`llama-3.3-70b-versatile`) generates a single valid Cypher query. The prompt enforces strict output rules: no Markdown, no explanation, output must start with `MATCH`.

**`get_valid_chunks`**  
Executes the Cypher query against the Neo4j session. Parses the result DataFrame to find all `chunk_id` columns, coerces them to integers, and deduplicates them. This produces the list of source chunk IDs that are semantically relevant to the query.

**`get_context`**  
Maps each chunk ID back to its `Document` object loaded from `chunks.jsonl`, assembling the context passages to be used by the final LLM call.

**`get_final_response`**  
Passes the original question and retrieved context chunks to the `FinalLLMPrompt`. Groq generates a concise, grounded analytical answer. The system persona is an LME aluminum industry analyst; the model is instructed to respond only from the provided context and say "Insufficient information" if the context is insufficient.

---

## Core Module Reference

### `core/graph.py`

| Class / Method | Description |
|---|---|
| `GraphRAGState` | Pydantic state model for the LangGraph workflow |
| `GraphRAGPOC.__init__()` | Initialises LLM, Neo4j session, loads chunks, builds workflow |
| `GraphRAGPOC.get_response(query)` | Public entry point — runs the full workflow and returns the answer |
| `_workflow_get_cypher_query()` | LangGraph node 1: natural language → Cypher |
| `_workflow_get_valid_chunks()` | LangGraph node 2: Cypher → chunk IDs |
| `_workflow_get_context()` | LangGraph node 3: chunk IDs → text passages |
| `_workflow_get_final_response()` | LangGraph node 4: passages → final answer |

### `core/utils.py`

| Class / Method | Description |
|---|---|
| `DocumentProcess.save_docs_jsonl()` | Serialise a list of LangChain `Document` objects to a JSONL file |
| `DocumentProcess.load_docs_jsonl()` | Deserialise a JSONL file back into a list of `Document` objects |
| `Neo4jOps.create_node()` | `MERGE` an entity node into Neo4j |
| `Neo4jOps.create_relationship()` | `MERGE` a typed relationship between two nodes |
| `Neo4jOps.extract_node_schema_info()` | Fetch node labels and property key counts from Neo4j |
| `Neo4jOps.extract_relationship_info()` | Fetch relationship types, source/target labels, and usage counts |
| `Neo4jOps.extract_node_property_info()` | Fetch sample property keys for each node label |

### `core/prompt.py`

| Class | Purpose |
|---|---|
| `SystemPrompt` | Static system prompt for KG extraction — vocabulary and normalisation rules |
| `UserPrompt` | Per-chunk user prompt for KG extraction — provides the text and JSON schema |
| `CypherPrompt` | Multi-step Cypher generation prompt — includes schema, normalisation, expansion, and filtering instructions |
| `FinalLLMPrompt` | Final answer prompt — enforces grounded, context-only responses as an LME analyst |

### `core/logging.py`

| Class | Description |
|---|---|
| `LoggerFactory` | Centralised logger factory. Produces loggers with a consistent format (`timestamp \| level \| name \| message`), console handler, and an optional rotating file handler (`logs/app.log`, 10 MB max, 3 backups). |

---

## Dataset

| File | Records | Description |
|---|---|---|
| `dataset/news-articles-raw.jsonl` | 65 articles | Raw LME aluminum market news scraped from AL Circle (Jan 2026). Each record contains `page_content`, `article_title`, `article_publish_date`, and `article_link`. |
| `dataset/chunks.jsonl` | 1,428 chunks | Articles split into 200-token chunks with no overlap. |
| `dataset/extracted-kg.jsonl` | 1,428 KG records | One KG (nodes + relationships) extracted per chunk by Ollama. |

The dataset covers topics including LME cash and futures prices, Chinese aluminum production and caps, CBAM (EU Carbon Border Adjustment Mechanism), SHFE (Shanghai Futures Exchange) movements, Indonesia supply growth, recycling investments, and tariff impacts.

---

## Prerequisites

- **Python 3.11** (the project requires `>=3.11, <3.12`)
- **Ollama** installed and running locally with `qwen2.5:3b` pulled — required only for Stage 2 (KG extraction)
- A **Neo4j AuraDB** free instance (or any Neo4j 5.x instance) with credentials
- A **Groq API key** — used for Cypher generation and final answer generation at query time

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/Tksrivastava/graph-rag-poc.git
cd graph-rag-poc

# 2. Create a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Install as a local package for development
pip install -e .
```

---

## Configuration

Copy the example environment file and fill in your credentials:

```bash
cp .env.poc.example .env.poc
```

Edit `.env.poc`:

```env
# Chunking
CHUNK_SIZE = 200
CHUNK_OVERLAP = 0

# Ollama (local LLM for KG extraction)
OLLAMA_MODEL_NAME = "qwen2.5:3b"

# Neo4j AuraDB
NEO4J_URI = "neo4j+s://<your-instance>.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "<your-password>"
NEO4J_DATABASE = "neo4j"
AURA_INSTANCEID = "<your-instance-id>"
AURA_INSTANCENAME = "<your-instance-name>"

# Groq (cloud LLM for querying)
GROQ_LLM_MODEL = "llama-3.3-70b-versatile"
GROQ_API_KEY = "<your-groq-api-key>"
```

> **Note:** `.env.poc` is listed in `.gitignore` and will never be committed. Never commit real credentials.

---

## Running the Pipeline

The four development stages are meant to be run **in order** the first time. The derived dataset files (`chunks.jsonl`, `extracted-kg.jsonl`) are already committed to the repository, so you can skip to Stage 3 or Stage 5 if you don't need to regenerate them.

### Stage 1 — Chunk the corpus
```bash
python development/extract-chunk-corpus.py
```
Reads `dataset/news-articles-raw.jsonl` → writes `dataset/chunks.jsonl`.

### Stage 2 — Extract the knowledge graph
```bash
# Ensure Ollama is running and qwen2.5:3b is available
ollama pull qwen2.5:3b
ollama serve

python development/extract-kg.py
```
Reads `dataset/chunks.jsonl` → writes `dataset/extracted-kg.jsonl`.  
This step is slow (one API call per chunk with a 5-second delay). Resume support is built in — edit the `chunk_id >= 136` threshold in the script to restart from a specific chunk.

### Stage 3 — Populate Neo4j
```bash
python development/create-graph-db.py
```
Reads `dataset/extracted-kg.jsonl` → pushes all nodes and relationships to Neo4j AuraDB.

### Stage 4 — Visualise the graph
```bash
python development/viz-graph.py
```
Queries Neo4j → renders `interactive-graph.html` and opens it in your default browser.

### Stage 5 — Query the graph (Jupyter)

Open `main.ipynb` in Jupyter and run the cells. The notebook demonstrates end-to-end querying using `GraphRAGPOC`:

```python
from neo4j import GraphDatabase
from core.graph import GraphRAGPOC

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

rag = GraphRAGPOC(
    groq_model=GROQ_LLM_MODEL,
    groq_api=GROQ_API_KEY,
    neo4j_driver=driver,
    chunks_path="dataset/chunks.jsonl",
)

answer = rag.get_response("What is happening with Chinese aluminum production?")
print(answer)
```

---

## Interactive Graph

The file `interactive-graph.html` is a self-contained interactive visualisation of the full knowledge graph stored in Neo4j. Open it directly in any modern browser:

```bash
open interactive-graph.html       # macOS
xdg-open interactive-graph.html   # Linux
start interactive-graph.html      # Windows
```

Features:
- Drag and reposition nodes
- Zoom in/out
- Hover to inspect node IDs
- Adjust physics simulation, node styling, and edge styling via the built-in control panel
- Directed edges showing relationship direction

---

## Dependencies

| Package | Version | Role |
|---|---|---|
| `langchain` | `>=0.2.0,<0.3.0` | Core RAG framework, document utilities, text splitter |
| `langgraph` | `>=0.2.0,<0.3.0` | Graph-based LLM workflow orchestration |
| `langchain-community` | `>=0.2.0,<0.3.0` | Community integrations |
| `langchain-ollama` | `0.1.3` | Ollama LLM integration (KG extraction) |
| `langchain-groq` | `0.1.9` | Groq LLM integration (query pipeline) |
| `neo4j` | `6.1.0` | Neo4j Python driver |
| `networkx` | `3.6.1` | Graph data structure for visualisation |
| `pyvis` | `0.3.2` | Interactive HTML graph rendering |
| `pypdf` | `4.2.0` | PDF parsing (if raw data includes PDFs) |
| `pydantic` | (transitive) | Data validation for KG models |
| `python-dotenv` | `1.0.1` | `.env` file loading |
| `pandas` | `2.2.2` | DataFrame operations on Neo4j query results |
| `pandas-toon` | `0.1.0` | DataFrame-to-string conversion for LLM prompts |
| `numpy` | `1.26.4` | Numerical utilities |
| `tqdm` | `4.67.3` | Progress bars for batch processing |
| `urllib3` | `1.26.18` | HTTP utilities |

---

## Author

**Tanul Kumar Srivastava**

---

## License

This project is licensed under the **MIT License**.
