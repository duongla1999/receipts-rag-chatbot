from pathlib import Path
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from src.database.models import Base, Receipt, ReceiptItem

DB_DIR = Path(__file__).resolve().parents[2] / "data" / "database"
engine = create_engine(f"sqlite:///{DB_DIR / 'receipts.db'}")

def total_spending_by_month(session: Session) -> dict[str, float]:
    stmt = (
        select(
            func.strftime("%Y-%m", Receipt.purchase_date).label("month"),
            func.sum(Receipt.total).label("total"),
        ).group_by("month").order_by("month")
    )
    return {row.month: round(row.total,2) for row in session.execute(stmt)}

def total_spending_by_store(session: Session) -> dict[str, float]:
    stmt = (
        select (
            Receipt.store_name, func.sum(Receipt.total)
        ).group_by(Receipt.store_name).order_by(func.sum(Receipt.total).desc())
    )
    return {row[0]: round(row[1], 2) for row in session.execute(stmt)}

def top_products(session: Session, limit: int = 5) -> list[tuple[str, int]]:
    stmt = (
        select(ReceiptItem.normalized_name, func.count(ReceiptItem.id).label("times"))
        .group_by(ReceiptItem.normalized_name)
        .order_by(func.count(ReceiptItem.id)
                  .desc()).limit(limit)
    )
    return [(row[0], row[1]) for row in session.execute(stmt)]

if __name__ == "__main__":
    with Session(engine) as session:
        print("Chi tieu theo thang:", total_spending_by_month(session))
        print("Chi tieu theo cua hang:", total_spending_by_store(session))
        print("San pham mua nhieu nhat:", top_products(session))