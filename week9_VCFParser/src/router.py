import os
from dotenv import load_dotenv 
import anthropic

# Load .env from project root (handles running from notebooks)
load_dotenv(os.path.join(os.path.dirname(__file__), '../../.env'))

client = anthropic.Anthropic()

MODEL="claude-opus-4-5"
SYSTEM_PROMPT = f"""
You are a variant routing agent. Based on the variant profile provided, 
call the appropriate annotation tool(s). Do not interpret or summarise — 
only call tools.
"""

def build_content(variant, ensembl_info, is_pharmacogene):
    return f"""
Variant: {variant.key}
Gene: {ensembl_info.get('gene_symbol')}
Consequence: {ensembl_info.get('consequence')}
Impact: {ensembl_info.get('impact')}
Allele Frequency: {variant.INFO.get('AF')}
Is pharmacogene: {is_pharmacogene}

Based on this profile, call the appropriate tool(s) to retrieve annotation data.
- If AF < 0.01 and is_pharmacogene: call query_pharmgkb
- If AF >= 0.01 or not a pharmacogene: call query_clinvar
- If both conditions apply: call both tools
"""

def LLMRouter(max_tokens: int = 1024):
    content = build_content(variant, ensembl_info, is_pharmacogene)
    response = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": content}
        ]
    )
    return response