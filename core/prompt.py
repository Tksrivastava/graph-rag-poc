class SystemPrompt:
    system_prompt = """
Extract Aluminum Supply Chain KGs. JSON ONLY.

STRICT CONVERSION:
- "lme" -> "london metal exchange"
- "aluminium" -> "aluminum"
- IDs: lowercase
- Numbers or Numeric Figures (MANDATORY): ALWAYS use {{"value": X, "unit": "Y"}} in properties.

### VOCABULARY:
- Types: [exchange, financial_metric, commodity, location, event, policy, etc.]
- Verbs: [reports, trades_on, affects, produces, exports_to, disrupts, etc.]
"""

class UserPrompt:
    def __init__(self, chunk: str = None):
        self.chunk = chunk

    def get_prompt(self):
        return f"""
Target: Extract all quantities and prices from the text.

IDENTITIES:
- Use "aluminum price" for price values.
- Use "aluminum stocks" for inventory values.
- Use "london metal exchange" for LME.

TEXT:
{self.chunk}

JSON STRUCTURE:
{{
  "nodes": [
    {{"id": "aluminum_price", "type": "financial_metric", "properties": {{"value": 0.0, "unit": "usd/t"}}}}
  ],
  "relationships": [
    {{"source": "london metal exchange", "target": "aluminum price", "type": "reports"}}
  ]
}}
"""