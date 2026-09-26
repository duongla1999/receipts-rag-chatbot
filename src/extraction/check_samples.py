import json
from pathlib import Path

from validator import parse_and_validate

SAMPLE_PATH = Path(__file__).resolve().parents[2] / "data" / "samples" / "sample_receipts.json"

with open(SAMPLE_PATH, encoding="utf-8") as f:
    raw_list = json.load(f)

for raw in raw_list:
    receipt, errors = parse_and_validate(raw)
    if errors:
        print(f"LOI ({raw.get('receipt_id', '?')}):")
        for err in errors:
            print(f"  - {err}")
    else:
        print(f"OK: {receipt.receipt_id} - {receipt.store_name} - {len(receipt.items)} items")