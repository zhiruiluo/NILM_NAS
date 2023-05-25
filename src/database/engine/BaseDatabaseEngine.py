import os
from pathlib import Path


class BaseDatabaseEngine:
    def __init__(self, db_name, db_dir):
        self.db_name = db_name
        self.db_dir = Path(os.path.realpath(db_dir))
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.db_dir.joinpath(db_name)

    def engine_name(self) -> str:
        return "base"
