import neo4j
import pandas as pd
from pathlib import Path
from pydantic import BaseModel
from langchain_groq import ChatGroq
from core.logging import LoggerFactory
from typing import Final, Optional, List
from langgraph.graph import StateGraph, END
from core.utils import Neo4jOps, DocumentProcess
from core.prompt import CypherPrompt, FinalLLMPrompt

logger = LoggerFactory().get_logger(__name__)


class GraphRAGState(BaseModel):
    natural_query: str
    cypher_query: Optional[str] = None
    valid_chunks: Optional[List[int]] = None
    context: Optional[List[str]] = None
    final_response: Optional[str] = None


class GraphRAGPOC:

    def __init__(
        self,
        groq_model: str = None,
        groq_api: str = None,
        neo4j_driver: neo4j._sync.driver.Neo4jDriver = None,
        chunks_path: Final[Path] = None,
    ):
        self.groq_model = groq_model
        self.groq_api = groq_api
        self.neo4j_driver = neo4j_driver
        self.neo = Neo4jOps(driver=self.neo4j_driver)
        self.chunks_path = chunks_path

        self.workflow = StateGraph(GraphRAGState)

        self._init_llm()
        self._init_neo4j_session()
        self._init_load_chunks()
        self._init_build_workflow()

    def _init_llm(self):
        if not self.groq_model or not self.groq_api:
            raise ValueError("Please provide groq_model and groq_api")

        self.llm = ChatGroq(
            model=self.groq_model,
            api_key=self.groq_api,
            temperature=0,
        )

        logger.info("LLM initialized | model=%s", self.groq_model)

    def _init_neo4j_session(self):
        if not self.neo4j_driver:
            raise ValueError("Please provide neo4j driver")

        self.session = self.neo4j_driver.session()
        logger.info("Neo4j session created successfully")

    def _init_load_chunks(self):
        if not self.chunks_path:
            raise ValueError("Please provide chunks")

        self.documents = DocumentProcess.load_docs_jsonl(self.chunks_path)
        logger.info("Documents loaded | total_chunks=%s", len(self.documents))

    def _workflow_get_cypher_query(self, state: GraphRAGState):
        logger.info("Node started | get_cypher_query")

        node_schema = self.neo.extract_node_schema_info()
        rel_schema = self.neo.extract_relationship_info()
        prop_schema = self.neo.extract_node_property_info()

        prompt = CypherPrompt(
            node_info=node_schema,
            relationship_info=rel_schema,
            sample_props=prop_schema,
        ).get_prompt(query=state.natural_query)

        logger.debug("Cypher generation prompt prepared")

        cypher_query = self.llm.invoke(prompt).content.strip()
        state.cypher_query = cypher_query

        logger.info("Cypher query generated")
        logger.debug("Generated Cypher: %s", cypher_query)

        return state

    def _workflow_get_valid_chunks(self, state: GraphRAGState):
        logger.info("Node started | get_valid_chunks")

        result = self.session.run(state.cypher_query).to_df()
        logger.info("Cypher query executed | rows=%s cols=%s", result.shape[0], result.shape[1])

        cols = result.columns.tolist()
        chunk_cols = [col for col in cols if "chunk_id" in col]

        logger.info("Chunk ID columns detected | columns=%s", chunk_cols)

        result[chunk_cols] = result[chunk_cols].apply(pd.to_numeric, errors="coerce")

        valid_chunks = (
            result[chunk_cols]
            .stack()
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        state.valid_chunks = valid_chunks

        logger.info(
            "Valid chunks extracted | count=%s",
            len(valid_chunks),
        )
        logger.debug("Valid chunk IDs: %s", valid_chunks)

        return state

    def _workflow_get_context(self, state: GraphRAGState):
        logger.info("Node started | get_context")

        context = [str(self.documents[i]) for i in state.valid_chunks]
        state.context = context

        logger.info("Context assembled | chunks_used=%s", len(context))

        return state

    def _workflow_get_final_response(self, state: GraphRAGState):
        logger.info("Node started | get_final_response")

        prompt = FinalLLMPrompt(
            query=state.natural_query,
            context=state.context,
        ).get_prompt()

        final_response = self.llm.invoke(prompt).content.strip()
        state.final_response = final_response

        logger.info("Final response generated")

        return state

    def _init_build_workflow(self):
        logger.info("Building LangGraph workflow")

        self.workflow.add_node("get_cypher_query", self._workflow_get_cypher_query)
        self.workflow.add_node("get_valid_chunks", self._workflow_get_valid_chunks)
        self.workflow.add_node("get_context", self._workflow_get_context)
        self.workflow.add_node("get_final_response", self._workflow_get_final_response)

        self.workflow.set_entry_point("get_cypher_query")
        logger.info("Workflow entry point set")

        self.workflow.add_edge("get_cypher_query", "get_valid_chunks")
        self.workflow.add_edge("get_valid_chunks", "get_context")
        self.workflow.add_edge("get_context", "get_final_response")

        logger.info("Workflow edges configured")

        self.app = self.workflow.compile()
        logger.info("LangGraph workflow compiled successfully")

    def get_response(self, query: str = None):
        if not query:
            raise ValueError("Please provide a valid query")

        logger.info("Workflow execution started | query=%s", query)

        self.response = self.app.invoke({"natural_query": query})

        logger.info("Workflow execution completed")

        return self.response["final_response"]