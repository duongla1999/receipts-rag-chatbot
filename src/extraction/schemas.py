from datetime import date, time
from pydantic import BaseModel

class ReceiptItemSchema(BaseModel):
    raw_name: str
    normalized_name: str | None = None
    category: str | None = None
    quantity: float = 1
    unit: str | None = None
    unit_price: float | None = None
    total_price: float

class ReceiptSchema(BaseModel):
    receipt_id: str
    store_name: str
    store_address: str | None = None
    purchase_date: date
    purchase_time: time | None = None
    currency: str = "EUR"
    items: list[ReceiptItemSchema]
    subtotal: float | None = None
    tax: float | None = None
    total: float
    payment_method: str | None = None
    