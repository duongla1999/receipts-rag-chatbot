from sqlalchemy import create_engine, select
from models import Base, Receipt, ReceiptItem
from pathlib import Path
import hashlib
import json
from datetime import date, time
from sqlalchemy.orm import Session

DB_DIR = Path(__file__).resolve().parents[2] / "data" / "database"
DB_DIR.mkdir(parents=True, exist_ok=True)
engine = create_engine(f"sqlite:///{DB_DIR / 'receipts.db'}")
Base.metadata.create_all(engine)

def content_hash_of(data: dict) -> str:
    raw = f"{data['store_name']} | {data['purchase_date']} | {data['total']}"
    return hashlib.sha256(raw.encode()).hexdigest()

def insert_receipt(session: Session, data: dict) -> Receipt | None:
    h = content_hash_of(data)
    existing = session.scalar(select(Receipt).where(Receipt.content_hash == h))

    if existing:
        print(f"Already Exists: {data['receipt_id]']}")
        return None

    receipt = Receipt(
        store_name = data['store_name'],
        store_address = data['store_address'],
        purchase_date = date.fromisoformat(data["purchase_date"]),
        purchase_time = time.fromisoformat(data['purchase_time']) if data.get("purchase_time") else None,
        subtotal = data.get("subtotal"),
        tax = data.get("tax"),
        total = data['total'],
        currency = data.get("currency", "EUR"),
        payment_method = data.get("payment_method"),
        content_hash = h,
    )

    for item in data["items"]:
        receipt.items.append(
            ReceiptItem(
                raw_name = item['raw_name'],
                normalized_name = item.get("normalized_name"),
                category = item.get("category"),
                quantity = item.get("quantity", 1),
                unit = item.get("unit"),
                unit_price = item.get("unit_price"),
                total_price = item["total_price"],
            )
        )
    session.add(receipt)
    return receipt

if __name__ == "__main__":
    SAMPLE_PATH = Path(__file__).resolve().parents[2] / "data" / "samples" / "sample_receipts.json"
    with open(SAMPLE_PATH, encoding="utf-8") as f:
        receipts_data = json.load(f)

    with Session(engine) as session:
        for data in receipts_data:
            insert_receipt(session, data)
        session.commit()
        print(f"Da nap {len(receipts_data)} hoa don (bo qua neu trung).")
    