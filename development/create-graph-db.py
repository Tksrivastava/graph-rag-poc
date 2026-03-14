import os
import json
from tqdm import tqdm
from typing import Final
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase
from core.utils import Neo4jOps
from core.logging import LoggerFactory

logger = LoggerFactory().get_logger(__name__)

BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent
ENV_PATH: Final[Path] = BASE_DIR / ".env.poc"
KG_SAVE_PATH: Final[Path] = BASE_DIR / "dataset" / "extracted-kg.jsonl"

load_dotenv(dotenv_path=ENV_PATH)
logger.info("Environment variables loaded from .env.poc")


def flatten_properties(props: dict) -> dict:
    """Flatten nested dictionaries for Neo4j compatibility."""
    flat = {}

    for k, v in props.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                flat[f"{k}_{sub_k}"] = sub_v
        else:
            flat[k] = v

    return flat


# Neo4j connection
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

neo = Neo4jOps(driver=driver)

logger.info("Connected to Neo4j AuraDB")


if __name__ == "__main__":

    logger.info("Loading Knowledge Graph")

    with KG_SAVE_PATH.open("r", encoding="utf-8") as f:

        for line in tqdm(f, desc="Creating Graph DB"):

            graph = json.loads(line)

            chunk_id = graph["chunk_id"]
            nodes = graph["nodes"]
            relationships = graph["relationships"]

            # Create nodes
            for node in nodes:

                node_props = flatten_properties(node.get("properties", {}))
                node_props["chunk_id"] = chunk_id

                node["properties"] = node_props

                neo.create_node(node=node)

            # Create relationships
            for relationship in relationships:

                rel_props = flatten_properties(relationship.get("properties", {}))
                rel_props["chunk_id"] = chunk_id

                relationship["properties"] = rel_props

                neo.create_relationship(relationship=relationship)

    logger.info("Graph DB created")

    driver.close()

    logger.info("Neo4j Aura DB connection closed")