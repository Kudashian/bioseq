import os
from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic()
api_key = os.getenv("ANTHROPIC_API_KEY")
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
def LLMInterpreter(content: str, max_tokens: int = 1024):
    call = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content}
        ]
    )
    return 