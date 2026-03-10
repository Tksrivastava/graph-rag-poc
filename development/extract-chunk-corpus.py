import os
import pickle
from typing import Final
from pathlib import Path
from pypdf import PdfReader
from dotenv import load_dotenv
from core.logging import LoggerFactory
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = LoggerFactory().get_logger(__name__)

BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent
ENV_PATH: Final[Path] = BASE_DIR / ".env.poc"
REPORT_PATH: Final[Path] = BASE_DIR / "dataset" / "tesla-report-10k.pdf"
CHUNK_SAVE_PATH: Final[Path] = BASE_DIR / "dataset" / "chunks.pkl"

load_dotenv(dotenv_path=ENV_PATH)
logger.info("Environment variables loaded from .env.poc")

if __name__ == "__main__":
    logger.info("Initiating text extraction and chunking")

    logger.info("Loading Tesla report")
    reader = PdfReader(REPORT_PATH)

    logger.info("Extracting data")
    report = ""
    for page in reader.pages:
        report += page.extract_text()
    logger.info(f"Report size - {len(report)}")

    logger.info("Chunking report")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(os.getenv("CHUNK_SIZE")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP")),
    )
    chunks = splitter.split_text(report)
    logger.info(f"{len(chunks)} chunks prepared")

    logger.info("Saving chunks")
    pickle.dump(chunks, open(CHUNK_SAVE_PATH, "wb"))
