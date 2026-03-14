import os
import json
import time
from tqdm import tqdm
from typing import Final
from pathlib import Path
from core.prompt import *
from dotenv import load_dotenv
from typing import List, Dict, Any
from pydantic import BaseModel, Field
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


if __name__ == "__main__":
    logger.info("Initiating Knowledge Graph extraction on chunks")

    logger.info("Loading Tesla report chunks")
    chunks = DocumentProcess.load_docs_jsonl(path=CHUNK_SAVE_PATH)

    logger.info("Initializing LMM")
    model = os.getenv("OLLAMA_MODEL_NAME")
    llm = ChatOllama(model=model, temperature=0, format="json")
    logger.info(f"{model} Model initialized with JSON output format")

    logger.info("Processing chunks sequentially")

    for chunk_id, doc in enumerate(tqdm(chunks, total=len(chunks), desc="Processing chunks")):

        if chunk_id >= 136:

            try:
                logger.info(f"Processing chunk_id - {chunk_id}")

                prompt = f"""
                        {SystemPrompt.system_prompt}
                        {UserPrompt(chunk=doc).get_prompt()}"""
                response = llm.invoke(prompt)
                logger.info(f"Response generated")
                
                response = GraphResponse(**json.loads(response.content if hasattr(response, "content") else response))
                logger.info("Pydantic parsed")

                time.sleep(5)

                record = {
                "chunk_id": chunk_id,
                "nodes": response.nodes,
                "relationships": response.relationships,
                }

                with open(KG_SAVE_PATH, "a") as f:
                    f.write(json.dumps(record, default=lambda x: x.model_dump()) + "\n")

            except Exception as e:
                logger.info(e)
        
        else : pass

    logger.info("Knowledge Graph extracted")