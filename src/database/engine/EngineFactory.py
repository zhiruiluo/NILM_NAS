from __future__ import annotations

from src.database.engine.SQLiteEngine import SQLiteEngine


class EngineFactory:
    def __init__(self):
        pass

    def get_sqlite_engine(
        self, db_name: str, db_dir: str, create_db: bool,
    ) -> SQLiteEngine:
        return SQLiteEngine(db_name, db_dir, create_db)
