import logging
import os

import sqlalchemy
from sqlalchemy import and_, create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from src.database.engine.BaseDatabaseEngine import BaseDatabaseEngine
from src.database.SQLiteMapper import Results, mapper_registry

logger = logging.getLogger(__name__)


def lockretry(func):
    def wrapper(*args, **kwargs):
        retry_times = 0
        max_retry = 3
        while retry_times <= max_retry:
            if retry_times != 0:
                logger.info(f"[LockRetry] retry {func.__name__} {retry_times} times!")
                print(f"[LockRetry] retry {func.__name__} {retry_times} times!")
            try:
                ret = func(*args, **kwargs)
            except sqlalchemy.exc.OperationalError as e:
                logger.info(f"[LockRetry] {e} retry {func.__name__}!")
                print(f"error message {e}")
                print(f"[LockRetry] retry {func.__name__} {retry_times} times!")
                retry_times += 1
            else:
                break
        else:
            exit()
        return ret

    return wrapper


class SQLiteEngine(BaseDatabaseEngine):
    def __init__(self, db_name: str, db_dir: str =__file__, create_db: str = True):
        super().__init__(db_name, db_dir)
        self.timeout = 15
        logger.info(f"[SQLiteEngine] db_file path {self.db_path}")
        if create_db:
            self._create_tables()
        else:
            if not os.path.isfile(self.db_path):
                raise FileNotFoundError(f"DB not found at path {self.db_path}")

    def engine_name(self):
        return "SQLite"

    @property
    def engine(self):
        if not hasattr(self, "_engine"):
            self._engine = create_engine(
                f"sqlite+pysqlite:///{self.db_path}",
                echo=False,
                future=True,
                connect_args={"timeout": self.timeout},
            )
        return self._engine

    @property
    def session(self) -> Session:
        if not hasattr(self, "_session"):
            Session = sessionmaker(bind=self.engine, autoflush=False)
            self._session = Session
        return self._session

    def query(self, *args, **kwargs):
        with self.session.begin() as session:
            q = session.query(*args, **kwargs) 
        return q

    @lockretry
    def insert(self, x):
        with self.session.begin() as session:
            session.add(x)

    def get_statement_exact_match(self, model: Results):
        clauses = []
        conditions = [
            k for k, v in vars(model).items() if k != "_sa_instance_state" and v is not None
        ]
        for k in conditions:
            clause = getattr(model.__table__.columns, k) == getattr(model, k)
            clauses.append(clause)

        statm = select(model.__table__).where(and_(*clauses))
        return statm

    @lockretry
    def get_all(self, statm):
        with self.session() as session:
            objs = session.execute(statm).all()
        return objs

    # def _create_tables(self):
    #     if not os.path.isfile(self.db_path):
    #         Base.metadata.create_all(self.engine)

    def _create_tables(self):
        if not os.path.isfile(self.db_path):
            mapper_registry.metadata.create_all(self.engine)
            logger.info(f"[SQLite Engine] initializing database {self.db_path}")
