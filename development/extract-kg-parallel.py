import os
import json
import subprocess
from tqdm import tqdm
from typing import Final
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from core.prompt import *
from core.logging import LoggerFactory
from core.utils import DocumentProcess
from langchain_ollama import ChatOllama

logger = LoggerFactory().get_logger(__name__)

BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent
ENV_PATH: Final[Path] = BASE_DIR / ".env.poc"
CHUNK_SAVE_PATH: Final[Path] = BASE_DIR / "dataset" / "chunks.jsonl"
KG_SAVE_PATH: Final[Path] = BASE_DIR / "dataset" / "extracted-kg.jsonl"

load_dotenv(dotenv_path=ENV_PATH)
logger.info("Environment variables loaded from .env.poc")

write_lock = threading.Lock()


class Node(BaseModel):
    id: str
    type: str
    properties: Dict[str, Any] = Field(default_factory=dict)


class Relationship(BaseModel):
    source: str
    target: str
    type: str
    properties: Dict[str, Any] = Field(default_factory=dict)


class GraphResponse(BaseModel):
    nodes: List[Node]
    relationships: List[Relationship]


class OllamaServer:
    def __init__(self, port: int):
        self.port = port
        self._start()

    def _start(self):
        logger.info(f"Starting Ollama server on port {self.port}")

        env = os.environ.copy()
        env["OLLAMA_HOST"] = f"http://127.0.0.1:{self.port}"

        self.process = subprocess.Popen(
            ["ollama", "serve"],
            env=env,
        )

    def stop(self):
        logger.info(f"Stopping Ollama server {self.port}")
        self.process.kill()


class OllamaClient:
    def __init__(self, port: int, model: str):
        self.port = port
        os.environ["OLLAMA_HOST"] = f"http://127.0.0.1:{port}"

        self.llm = ChatOllama(
            model=model,
            temperature=0,
            format="json"
        )

    def get_kg(self, content_chunk: str, chunk_id: int):

        prompt = f"""
{SystemPrompt.system_prompt}
{UserPrompt(chunk=content_chunk).get_prompt()}
"""

        response = self.llm.invoke(prompt)

        logger.info(f"Response generated for chunk_id={chunk_id}")

        return GraphResponse(
            **json.loads(response.content if hasattr(response, "content") else response)
        )


def process_chunk(chunk_id, doc, client: OllamaClient):

    response = client.get_kg(
        chunk_id=chunk_id,
        content_chunk=doc
    )

    record = {
        "chunk_id": chunk_id,
        "nodes": response.nodes,
        "relationships": response.relationships,
    }

    with write_lock:
        with open(KG_SAVE_PATH, "a") as f:
            f.write(json.dumps(record, default=lambda x: x.model_dump()) + "\n")


if __name__ == "__main__":

    logger.info("Initiating Knowledge Graph extraction on chunks")

    chunks = DocumentProcess.load_docs_jsonl(path=CHUNK_SAVE_PATH)

    model = os.getenv("OLLAMA_MODEL_NAME")

    server1 = OllamaServer(port=11434)
    server2 = OllamaServer(port=11435)

    client1 = OllamaClient(port=11434, model=model)
    client2 = OllamaClient(port=11435, model=model)

    clients = [client1, client2]

    logger.info("Starting threaded processing")

    futures = []

    with ThreadPoolExecutor(max_workers=3) as executor:

        for chunk_id, doc in enumerate(chunks):

            client = clients[chunk_id % 2]

            futures.append(
                executor.submit(process_chunk, chunk_id, doc, client)
            )

        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass

    logger.info("Knowledge Graph extracted")

    server1.stop()
    server2.stop()