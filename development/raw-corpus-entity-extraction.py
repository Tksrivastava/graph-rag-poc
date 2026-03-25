import os
import json
import time
from tqdm import tqdm
from typing import List
from typing import Final
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from core.logging import LoggerFactory
from core.utils import DocumentProcess
from langchain_ollama import ChatOllama
from core.prompt import EntityExtractionPrompt

logger = LoggerFactory().get_logger(__name__)

BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent
ENV_PATH: Final[Path] = BASE_DIR / ".env.poc"
CHUNK_SAVE_PATH: Final[Path] = BASE_DIR / "dataset" / "raw-article-paragraph-chunks.jsonl"
ENTITY_SAVE_PATH: Final[Path] = BASE_DIR / "dataset" / "extracted-entities.jsonl"

load_dotenv(dotenv_path=ENV_PATH)
logger.info("Environment variables loaded from .env.poc")

class ExtractEntity(BaseModel):
    entities: List[str] = Field(..., description="Unique entities identified based on the input context")

if __name__ == "__main__":
    logger.info("Initiating entity extraction on chunks")

    logger.info("Loading LME Aluminum report chunks")
    chunks = DocumentProcess.load_docs_jsonl(path=CHUNK_SAVE_PATH)

    logger.info("Initializing LMM")
    model = os.getenv("OLLAMA_MODEL_NAME")
    llm = ChatOllama(model=model, temperature=0, format="json")
    logger.info(f"{model} Model initialized with JSON output format")

    logger.info("Processing chunks sequentially")

    for chunk_id, doc in enumerate(tqdm(chunks, total=len(chunks), desc="Processing chunks")):

        try:
            logger.info(f"Processing chunk_id - {chunk_id}")
            response = llm.invoke(EntityExtractionPrompt(chunk=doc).get_prompt())
            logger.info(f"Response generated")
            
            response = ExtractEntity(**json.loads(response.content if hasattr(response, "content") else response))
            logger.info("Pydantic parsed")

            time.sleep(5)

            record = {
            "chunk_id": chunk_id,
            "entities": response.entities,
            "metadata": doc.metadata
            }

            with open(ENTITY_SAVE_PATH, "a") as f:
                f.write(json.dumps(record, default=lambda x: x.model_dump()) + "\n")

        except Exception as e:
            logger.info(e)

    logger.info("Entity extracted")