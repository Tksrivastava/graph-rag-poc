import os
import json
import pickle
import threading
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
from concurrent.futures import ThreadPoolExecutor

logger = LoggerFactory().get_logger(__name__)

BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent
ENV_PATH: Final[Path] = BASE_DIR / ".env.poc"
CHUNK_SAVE_PATH: Final[Path] = BASE_DIR / "dataset" / "chunks.pkl"
KG_SAVE_PATH: Final[Path] = BASE_DIR / "dataset" / "extracted-kg.jsonl"

load_dotenv(dotenv_path=ENV_PATH)
logger.info("Environment variables loaded from .env.poc")

# Creating the Pydantic class for structure output
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


# Initializing LLM and setting up Ollama server
class ServeOllama:
    def __init__(self):
        self._start_ollama_serve()

        self.model = os.getenv("OLLAMA_MODEL_NAME")
        self.llm = ChatOllama(model=self.model, temperature=0, format="json")
        logger.info(f"{self.model} Model initialized with JSON output format")

    def _start_ollama_serve(self):
        logger.info("Starting Ollama serve")
        self.process = subprocess.Popen(["ollama", "serve"], env=os.environ.copy())

    def _kill_ollama_serve(self):
        self.process.kill()
        logger.info("Killing Ollama serve")

    def get_kg(self, content_chunk: str = None, chunk_id: int = None):
        prompt = f"""
                    {SystemPrompt.system_prompt}
                    {UserPrompt(chunk=content_chunk).get_prompt()}"""
        response = self.llm.invoke(prompt)
        logger.info(f"Response generated for chunk_id={chunk_id}")

        return GraphResponse(
            **json.loads(response.content if hasattr(response, "content") else response)
        )


# Thread lock for safe file writing
write_lock = threading.Lock()


def process_chunk(args):
    chunk_id, chunk, llm = args

    logger.info(f"Processing chunk_id - {chunk_id}")

    response = llm.get_kg(chunk_id=chunk_id, content_chunk=chunk)

    record = {
        "chunk_id": chunk_id,
        "nodes": response.nodes,
        "relationships": response.relationships,
    }

    with write_lock:
        with open(KG_SAVE_PATH, "a") as f:
            f.write(json.dumps(record, default=lambda x: x.model_dump()) + "\n")


# Main
if __name__ == "__main__":
    logger.info("Initiating Knowledge Graph extraction on chunks")

    logger.info("Loading Tesla report chunks")
    chunks = pickle.load(open(CHUNK_SAVE_PATH, "rb"))

    logger.info("Initializing Ollama")
    llm = ServeOllama()

    logger.info("Processing chunks in parallel")

    tasks = [(i, chunk, llm) for i, chunk in enumerate(chunks)]

    with ThreadPoolExecutor(max_workers=3) as executor:
        list(tqdm(executor.map(process_chunk, tasks), total=len(tasks)))

    logger.info("Knowledge Graph extracted")

    llm._kill_ollama_serve()