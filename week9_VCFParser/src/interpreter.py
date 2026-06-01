import os
import re
import json
from dotenv import load_dotenv
import anthropic

# Load .env from project root (handles running from notebooks)
load_dotenv(os.path.join(os.path.dirname(__file__), '../../.env'))

client = anthropic.Anthropic()

MODEL="claude-opus-4-5"
SYSTEM_PROMPT = f"""
You are a research variant interpretation tool. Your sole function is to 
interpret genetic variants using only the evidence provided to you. 

You must:
- Base interpretations strictly on provided consequence, frequency, 
  and database annotations
- Explicitly state when evidence is absent or insufficient
- Never infer clinical significance beyond what the data supports

You must never:
- Hallucinate database entries or literature references
- Make treatment recommendations
- State conclusions not supported by the provided evidence
"""

def build_content(variant, ensembl_info, pharmgkb_info):
    return f"""
Variant: {variant.key}
Gene: {ensembl_info.get('gene_symbol')}
Consequence: {ensembl_info.get('consequence')}
Impact: {ensembl_info.get('impact')}
Amino Acid Change: {ensembl_info.get('amino_acids')}
Allele Frequency: {variant.INFO.get('AF')}
PharmGKB annotations: {pharmgkb_info}

Provide your response in two parts:
1. A markdown clinical interpretation report
2. A JSON block using exactly this schema:
{{
    "variant_key": "",
    "chrom": "",
    "pos": "",
    "ref": "",
    "alt": "",
    "rsid": "",
    "gene": "",
    "af": "",
    "consequence": "",
    "impact": "",
    "biological_function": "",
    "associated_disease": "",
    "associated_medication": "",
    "medication_effect": "",
    "clinical_risk": "",
    "confidence": "",
    "evidence_source": [],
    "interpretation": ""
}}
Wrap the JSON in ```json``` tags.
"""

def parse_response(response_text):
    json_match = re.search(r'```json(.*?)```', response_text, re.DOTALL)
    structured = json.loads(json_match.group(1).strip()) if json_match else {}
    report = re.sub(r'```json.*?```', '', response_text, flags=re.DOTALL).strip()
    return report, structured

def LLMInterpreter(content: str, max_tokens: int = 1024):
    response = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": content}
        ]
    )
    raw = response.content[0].text
    print("Raw LLM response:")
    print(raw)
    report, structured = parse_response(raw)
    return report, structured