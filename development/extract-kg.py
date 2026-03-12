import os
import json
import pickle
import subprocess
from tqdm import tqdm
from typing import Final
from pathlib import Path
from core.prompt import *
from dotenv import load_dotenv
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from core.logging import LoggerFactory
from langchain_ollama import ChatOllama

logger = LoggerFactory().get_logger(__name__)

BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent
ENV_PATH: Final[Path] = BASE_DIR / ".env.poc"
CHUNK_SAVE_PATH: Final[Path] = BASE_DIR / "dataset" / "chunks.pkl"
KG_SAVE_PATH: Final[Path] = BASE_DIR / "dataset" / "extracted-kg.jsonl"

load_dotenv(dotenv_path=ENV_PATH)
logger.info("Environment variables loaded from .env.poc")


class Node(BaseModel):
    id: str = Field(..., description="Unique identifier of the entity")
    type: str = Field(..., description="Type of the entity")
    properties: Dict[str, Any] = Field(default_factory=dict)


class Relationship(BaseModel):
    source: str = Field(..., description="Source node id")
    target: str = Field(..., description="Target node id")
    type: str = Field(..., description="Relationship type")
    properties: Dict[str, Any] = Field(default_factory=dict)


class GraphResponse(BaseModel):
    nodes: List[Node]
    relationships: List[Relationship]


class ServeOllama:
    def __init__(self):
        self._start_ollama_serve()

        self.model = os.getenv("OLLAMA_MODEL_NAME")
        self.llm = ChatOllama(model=self.model, temperature=0, format="json")
        logger.info(f"{self.model} Model initialized with JSON output format")

    def _start_ollama_serve(self):
        logger.info("Starting Ollama serve")
        self.process = subprocess.Popen(["ollama", "serve"])

    def _kill_ollama_serve(self):
        self.process.kill()
        logger.info("Killing Ollama serve")

    def get_kg(self, content_chunk: str = None, chunk_id: int = None, previous_chunks: List[str] = None):
        prompt = f"""
                    {SystemPrompt.system_prompt}
                    {UserPrompt(chunk=content_chunk, previous_chunks=previous_chunks).get_prompt()}"""

        response = self.llm.invoke(prompt)

        logger.info(f"Response generated for chunk_id={chunk_id}")

        return GraphResponse(
            **json.loads(response.content if hasattr(response, "content") else response)
        )


if __name__ == "__main__":
    logger.info("Initiating Knowledge Graph extraction on chunks")

    logger.info("Loading Tesla report chunks")
    chunks = pickle.load(open(CHUNK_SAVE_PATH, "rb"))

    logger.info("Initializing Ollama")
    llm = ServeOllama()

    logger.info("Processing chunks sequentially")

    for chunk_id, chunk in tqdm(enumerate(chunks), total=len(chunks)):

        logger.info(f"Processing chunk_id - {chunk_id}")

        prev_chunks = []
        if chunk_id - 1 >= 0: prev_chunks.append(chunks[chunk_id - 1])
        if chunk_id - 2 >= 0: prev_chunks.append(chunks[chunk_id - 2])
        if len(prev_chunks) == 2: logger.info("Two chunks extracted for context")
        elif len(prev_chunks) == 1: logger.info("Single chunk extracted for context")
        else: logger.info("No previous chunks for context")

        response = llm.get_kg(chunk_id=chunk_id, content_chunk=chunk, previous_chunks=prev_chunks)

        record = {
            "chunk_id": chunk_id,
            "nodes": response.nodes,
            "relationships": response.relationships,
        }

        with open(KG_SAVE_PATH, "a") as f:
            f.write(json.dumps(record, default=lambda x: x.model_dump()) + "\n")

    logger.info("Knowledge Graph extracted")

    llm._kill_ollama_serve()