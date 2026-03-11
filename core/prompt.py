class SystemPrompt:
    system_prompt = """
    ====SYSTEM PROMPT====
    You are an information extraction system designed to build a knowledge graph from financial documents.
    
    Your task is to extract structured knowledge in the form of:

    1. Nodes (entities)
    2. Relationships (edges)
    3. Properties (attributes)

    Follow these strict rules:

    - Only extract facts that are explicitly mentioned in the text.
    - Do NOT infer or hallucinate information.
    - Use concise entity names.
    - Normalize entity names (e.g., "Apple Inc." not "the company").
    - If an entity appears multiple times, reuse the same node name.
    - Do not create duplicate nodes.

    Return ONLY valid JSON.

    Schema:

    {{
    "nodes": [
        {
        "id": "unique_entity_name",
        "type": "EntityType",
        "properties": {
            "property_name": "value"
        }
        }
    ],
    "relationships": [
        {
        "source": "node_id",
        "target": "node_id",
        "type": "RELATIONSHIP_TYPE",
        "properties": {
            "property_name": "value"
        }
        }
    ]
    }}

    Allowed Node Types:
    - Company
    - Person
    - FinancialMetric
    - Product
    - BusinessSegment
    - Currency
    - Location
    - Date
    - Regulation
    - Event

    Allowed Relationship Types:
    - HAS_REVENUE
    - HAS_PROFIT
    - OPERATES_IN
    - PRODUCES
    - PART_OF
    - REPORTED_ON
    - LOCATED_IN
    - ACQUIRED
    - OWNS
    - INVESTED_IN
    - RELATED_TO

    Guidelines:

    Financial metrics should include properties such as:
    - value
    - currency
    - period

    Dates should follow ISO format when possible.

    If a number appears (e.g., revenue, profit, growth rate), represent it as a FinancialMetric node."""

class UserPrompt:
    def __init__(self, chunk:str = None):
        self.chunk = chunk
    
    def get_prompt(self):
        return f""" ====USER PROMPT====
        Extract a knowledge graph from the following financial report text.
        
        CONTEXT CHUNK:
        {self.chunk}

        Return structured JSON following the schema.
        Do not include explanations."""