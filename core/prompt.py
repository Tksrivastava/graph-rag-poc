from typing import List

class EntityExtractionPrompt:
    def __init__(self, chunk: str = None):
        self.chunk = chunk

    def get_prompt(self):
        return f"""Extract entities ONLY from the given context.
        
        Context:
        {self.chunk}
        
        STRICT RULES (ALL MUST BE FOLLOWED):

        1. Extract ONLY meaningful domain entities relevant to the aluminium/LME industry
        (e.g., commodities, financial indicators, inventory indicators, organizations, markets).

        2. The following MUST NEVER be entities:
            - dates
            - numbers
            - percentages
            - quantities
            - identifiers
            - currency codes
            - URLs

        3. Dates, numbers, percentages, and quantities MUST appear ONLY as properties of a related entity.

        4. Every entity MUST:
            - exist exactly as written in the context
            - contain at least one property
            - appear ONLY once in the output (no duplicates)

        5. Entity types MUST be domain-specific and meaningful.
        Examples:
            - Financial Indicator
            - Commodity Price Indicator
            - Inventory Indicator
            - Organization
            - Market

        6. If a token is a date, number, percentage, identifier, or currency code,
        it MUST be used ONLY as a property and NEVER as an entity.

        7. Output MUST follow EXACT formatting.
            Required Format: "**Entity Name:** <name> - **Entity Type:** <type> - **Entity Properties:** [prop1, prop2]"
            Example: "**Entity Name:** LME Aluminum Cash - **Entity Type:** Financial Indicator - **Entity Properties:** [Cash Price $2,985.5/t, Date 2026-01-05]"
        
        8. Think step-by-step internally before producing the final JSON output.
        
        9.  Before producing the final answer, verify:
            - No entity is a date, number, percentage, identifier, or currency.
            - No duplicate entities exist.
            - Every entity has at least one property.
            - Entity properties should not be empty.
            - If multiple numeric values refer to the same entity, they must be aggregated into the properties of a single entity.
        
        10. Output JSON format:
            class ExtractEntity(BaseModel):
                entities: List[str] = Field(..., description="Unique entities identified based on the input context")
        """
    
class CypherPrompt:
    def __init__(self, node_info: str, relationship_info: str, sample_props: str):
        self.node_info = node_info
        self.relationship_info = relationship_info
        self.sample_props = sample_props

    def get_prompt(self, query: str):
        return f"""
You are an expert Neo4j Cypher query generator.

Generate ONE valid Cypher query that retrieves graph nodes related to the user's question using ONLY the provided graph schema.

--------------------------------
GRAPH SCHEMA
--------------------------------

NODES:
{self.node_info}

RELATIONSHIPS:
{self.relationship_info}

SAMPLE PROPERTIES:
{self.sample_props}

--------------------------------
STEP 1 — NORMALIZE THE QUESTION
--------------------------------

Before generating Cypher:

1. Correct obvious spelling errors.

Example:
"increasesing" → "increasing"

2. Remove conversational words that do not affect meaning.

Ignore:
any, update, latest, news, tell, show, please, information, data

Example:

Question:
"Any update on Chinese aluminum production?"

Normalized keywords:
china
aluminum
production

3. Remove filler words:

is, are, the, a, an, of, or, on, in, about

--------------------------------
STEP 2 — IDENTIFY CORE CONCEPTS
--------------------------------

Extract the main domain concepts from the question.

Example:

Question:
"Chinese aluminum production"

Concept groups:

country:
china, chinese

commodity:
aluminum, aluminium

activity:
production, output, supply

Do NOT include conversational words.

--------------------------------
STEP 3 — EXPAND TERMS BY CONCEPT
--------------------------------

Expand each concept with common industry synonyms.

Examples:

china → china, chinese  
aluminum → aluminum, aluminium  
production → production, output, supply, smelter output  
energy → energy, electricity, power  
consumption → consumption, usage  

Do NOT expand excessively.

Keep synonym groups small and relevant.

--------------------------------
QUERY CONSTRAINTS
--------------------------------

1. Use ONLY this node label:

Entity

2. Use ONLY relationships provided in the schema.

3. Maximum traversal depth:

1..3 hops

Example traversal:

MATCH p=(n:Entity)-[:REL_TYPE*1..3]-(m:Entity)

4. Use substring matching ONLY.

Allowed:
n.description CONTAINS "keyword"

NOT allowed:
=
STARTS WITH
ENDS WITH

5. Text may appear in these properties:

description  
name  
type  
source  
target  

Prefer searching **description first** because most article information appears there.

--------------------------------
FILTERING STRATEGY
--------------------------------

1. Use OR between conditions.

2. DO NOT generate extremely long WHERE clauses.

3. Avoid repeating the same keyword across every property.

4. Prefer searching description and name first.

Example structure:

WHERE
(
n.description CONTAINS "china"
OR n.description CONTAINS "chinese"
OR m.description CONTAINS "china"
)
OR
(
n.description CONTAINS "aluminum"
OR n.description CONTAINS "aluminium"
)
OR
(
n.description CONTAINS "production"
OR n.description CONTAINS "output"
)

--------------------------------
RELATIONSHIP PRIORITY
--------------------------------

Prefer these relationships when available:

affects > reports > trades_on

--------------------------------
QUERY TEMPLATE
--------------------------------

MATCH p=(n:Entity)-[:affects|reports|trades_on*1..3]-(m:Entity)

WHERE
(
n.description CONTAINS "keyword"
OR m.description CONTAINS "keyword"
OR n.name CONTAINS "keyword"
OR m.name CONTAINS "keyword"
)

RETURN
n,
m,
n.id,
n.type,
n.chunk_id,
m.id,
m.type,
m.chunk_id

LIMIT 25

--------------------------------
OUTPUT RULES
--------------------------------

Return ONLY the Cypher query.

Do NOT include:
explanations
markdown
comments
multiple queries

The output MUST start with:

MATCH

--------------------------------
USER QUESTION
--------------------------------

{query}
"""
    
class FinalLLMPrompt:
    def __init__(self, query: str =  None, context: List[str] = None):
        self.query = query
        self.context = context
    def get_prompt(self):
        return f"""
You are an analyst specializing in the London Metal Exchange (LME) aluminum industry.

Use ONLY the provided context to answer the user's question.

If the context does not contain enough information, say:
"Insufficient information in the retrieved data."

Do not invent facts.

---------------------
USER QUESTION
---------------------
{self.query}

---------------------
RETRIEVED CONTEXT
---------------------
{self.context}

---------------------
INSTRUCTIONS
---------------------
Provide a concise analytical answer based strictly on the context.

Focus on:
- Chinese aluminum production or output
- LME market activity
- relevant companies, regions, or events

Summarize the key information clearly.
"""