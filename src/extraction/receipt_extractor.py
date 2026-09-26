import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

from schemas import ReceiptSchema

load_dotenv()

OLLAMA_BASE_URL="https://clever-spoons-punch.loca.lt"
OLLAMA_MODEL="llama3"


SYSTEM_PROMPT = """You are a data extraction system for German supermarket receipts (Kassenbon).
Read the raw OCR text below and return it in the exact JSON structure defined by the given schema.
If you are not sure about a value, use None instead of guessing.
Normalize abbreviated product names (e.g., H-MILCH -> Vollmilch) into the normalized_name field,
but KEEP the original text unchanged in the raw_name field.
Do not invent products that are not present in the original text."""


def extract_receipt(raw_text: str) -> ReceiptSchema | None:
    llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0,
    client_kwargs={"timeout": 120},
    )
    structured_llm = llm.with_structured_output(ReceiptSchema).with_retry(stop_after_attempt=3)

    try:
        return structured_llm.invoke(
            [
                ("system", SYSTEM_PROMPT),
                ("human", raw_text),
            ]
        )
    except Exception as e:
        print(f"Failed to connect with LLM: {e}")
        return None