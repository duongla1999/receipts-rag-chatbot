from sqlalchemy import create_engine
from models import Base
from pathlib import Path

DB_DIR = Path(__file__).resolve().parents[2] / "data" / "database"
DB_DIR.mkdir(parents=True, exist_ok=True)
engine = create_engine(f"sqlite:///{DB_DIR / 'receipts.db'}")
Base.metadata.create_all(engine)