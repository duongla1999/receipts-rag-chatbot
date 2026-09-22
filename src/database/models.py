from datetime import date, time
from sqlalchemy import ForeignKey, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass

class Receipt(Base):
    __tablename__ = "receipts"
    __table_args__ = (UniqueConstraint("content_hash", name="uq_receipts_content_hash"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    store_name: Mapped[str]
    store_address: Mapped[str | None]
    purchase_date: Mapped[date]
    purchase_time: Mapped[time |None]
    subtotal: Mapped[float | None]
    tax: Mapped[float | None]
    total: Mapped[float]
    currency: Mapped[str] = mapped_column(default="EUR")
    payment_method: Mapped[str | None]
    image_path: Mapped[str | None]
    raw_text: Mapped[str | None]
    content_hash: Mapped[str]

    items: Mapped[list["ReceiptItem"]] = relationship(back_populates="receipt", cascade="all, delete-orphan")
    documents: Mapped[list["ReceiptDocument"]] = relationship(back_populates="receipt", cascade="all, delete-orphan")

class ReceiptItem(Base):
    __tablename__ = "receipts_items"

    id: Mapped[int] = mapped_column(primary_key=True)
    receipts_id: Mapped[int] = mapped_column(ForeignKey("receipts.id"))
    raw_name: Mapped[str]
    normalized_name: Mapped[str | None]
    category: Mapped[str | None]
    quantity: Mapped[float] = mapped_column(default=1)
    unit: Mapped[str | None]
    unit_price: Mapped[float | None]
    total_price: Mapped[float]

    receipt: Mapped["Receipt"] = relationship(back_populates="items")

class ReceiptDocument(Base):
    __tablename__ = "receipt_documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    receipt_id: Mapped[int] = mapped_column(ForeignKey("receipts.id"))
    text: Mapped[str]
    embedding_id: Mapped[str | None]

    receipt: Mapped["Receipt"] = relationship(back_populates="documents")