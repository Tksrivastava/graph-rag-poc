import json
import neo4j
from pathlib import Path
from langchain.schema import Document
from core.logging import LoggerFactory
from typing import Final, Iterable, List, Union, Dict

logger = LoggerFactory().get_logger(__name__)

class DocumentProcess:

    @staticmethod
    def save_docs_jsonl(
        docs: Iterable[Document],
        path: Final[Union[str, Path]]
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            for doc in docs:
                row = {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                }
                f.write(json.dumps(row, default=str) + "\n")

    @staticmethod
    def load_docs_jsonl(
        path: Final[Union[str, Path]]
    ) -> List[Document]:
        """
        Load LangChain Documents from a JSONL file.
        """
        path = Path(path)
        docs: List[Document] = []

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                docs.append(
                    Document(
                        page_content=row["page_content"],
                        metadata=row.get("metadata", {}),
                    )
                )

        return docs

class Neo4jOps:
    def __init__(self, driver: neo4j._sync.driver.Neo4jDriver):
        self.session = driver.session()
        logger.info("Session created")

    def create_node(self, node: Dict):
        query = """
        MERGE (n:Entity {id: $id})
        SET n.type = $type
        SET n += $props"""
        
        self.session.run(query, id=node["id"], type=node["type"], props=node["properties"])
        logger.info("Node pushed")
    
    def create_relationship(self, relationship: Dict):
        query = """
        MATCH (a:Entity {id: $source})
        MATCH (b:Entity {id: $target})
        MERGE (a)-[r:%s]->(b)
        SET r += $props
        """ % relationship["type"]
        
        self.session.run(query, source=relationship["source"], target=relationship["target"], props=relationship["properties"])
        logger.info("Relationship pushed")