import os
import itertools
from typing import Final
from pathlib import Path
from pypdf import PdfReader
from dotenv import load_dotenv
from langchain.schema import Document
from core.utils import DocumentProcess
from core.logging import LoggerFactory

logger = LoggerFactory().get_logger(__name__)

BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent
ENV_PATH: Final[Path] = BASE_DIR / ".env.poc"
REPORT_PATH: Final[Path] = BASE_DIR / "dataset" / "news-articles-raw.jsonl"
CHUNK_SAVE_PATH: Final[Path] = BASE_DIR / "dataset" / "raw-article-paragraph-chunks.jsonl"

load_dotenv(dotenv_path=ENV_PATH)
logger.info("Environment variables loaded from .env.poc")


if __name__ == "__main__":
    logger.info("Initiating text extraction and chunking")

    logger.info("Loading News report")
    report = DocumentProcess.load_docs_jsonl(path=REPORT_PATH)

    logger.info("Chunking news articles based on paragraph")
    chunks = [Document(page_content=chunk, metadata=news_content.metadata)
              for news_content in report for chunk in news_content.page_content.split("\n")]
    logger.info(f"{len(chunks)} chunks prepared")

    logger.info("Saving chunks")
    DocumentProcess.save_docs_jsonl(docs=chunks, path=CHUNK_SAVE_PATH)
