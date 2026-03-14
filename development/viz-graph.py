import os
import webbrowser
import networkx as nx
from typing import Final
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase
from pyvis.network import Network
from core.logging import LoggerFactory

logger = LoggerFactory().get_logger(__name__)

BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent
ENV_PATH: Final[Path] = BASE_DIR / ".env.poc"
GRAPH_VIZ_PATH: Final[Path] = BASE_DIR / "interactive-graph.html"

load_dotenv(dotenv_path=ENV_PATH)
logger.info("Environment variables loaded from .env.poc")

# Neo4j connection
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)
logger.info("Connected to Neo4j AuraDB")

if __name__ == "__main__":
    G = nx.Graph()

    with driver.session() as session:
        result = session.run("""
        MATCH (a)-[r]->(b)
        RETURN a.id AS source, b.id AS target
        """)

        for record in result:
            G.add_edge(record["source"], record["target"])

    driver.close()
    logger.info("Neo4j Aura DB connection closed")

    # Create interactive PyVis network
    nt = Network(
        height="750px", 
        width="100%",
        directed=True,
        notebook=False,  # For .py files
        cdn_resources='in_line'  # Fixes warning
    )

    nt.from_nx(G)

    # Add maximum interactivity
    nt.show_buttons(['physics', 'nodes', 'edges', 'scale'])
    nt.set_options('''
    {
    "physics": {
        "enabled": true,
        "forceAtlas2Based": {
        "springLength": 100,
        "avoidOverlap": 0.1
        },
        "minVelocity": 0.75
    },
    "interaction": {
        "dragNodes": true,
        "zoomView": true,
        "selectNode": true,
        "hover": true
    },
    "configure": {
        "enabled": true,
        "filter": "physics,interaction"
    }
    }
    ''')

    # Generate and auto-open
    nt.write_html(str(GRAPH_VIZ_PATH))
    logger.info("Interactive graph saved as 'interactive_graph.html'")

    webbrowser.open(str(GRAPH_VIZ_PATH))
    logger.info(f"Interactive graph saved as '{GRAPH_VIZ_PATH}' and opened in browser")