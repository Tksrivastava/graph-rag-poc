import os
from typing import Final
from pathlib import Path
from pypdf import PdfReader
from dotenv import load_dotenv
from core.utils import DocumentProcess
from core.logging import LoggerFactory
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = LoggerFactory().get_logger(__name__)

BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent
ENV_PATH: Final[Path] = BASE_DIR / ".env.poc"
REPORT_PATH: Final[Path] = BASE_DIR / "dataset" / "news-articles-raw.jsonl"
CHUNK_SAVE_PATH: Final[Path] = BASE_DIR / "dataset" / "chunks.jsonl"

load_dotenv(dotenv_path=ENV_PATH)
logger.info("Environment variables loaded from .env.poc")


if __name__ == "__main__":
    logger.info("Initiating text extraction and chunking")

    logger.info("Loading News report")
    report = DocumentProcess.load_docs_jsonl(path=REPORT_PATH)

    logger.info("Chunking report")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(os.getenv("CHUNK_SIZE")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP")),
    )
    chunks = splitter.split_documents(report)
    logger.info(f"{len(chunks)} chunks prepared")

    logger.info("Saving chunks")
    DocumentProcess.save_docs_jsonl(docs=chunks, path=CHUNK_SAVE_PATH)
