from __future__ import annotations

import logging

from src.base_module.configs import ExpResults
from src.config_options.option_def import MyProgramArgs

from .engine.EngineFactory import EngineFactory
from .ResultsDao import ResultDao, ResultDaoFactory

logger = logging.getLogger(__name__)


class Persistence:
    def __init__(self, result_dao: ResultDao):
        self.result_dao = result_dao

    def check_results(self):
        pass

    def save_expresults(self, result: ExpResults, args: MyProgramArgs):
        self.result_dao.save_result(result, args)

    # def save_results(self, results: dict):
    #     train_metrics = Metrics(acc, accmacro, f1macro, f1micro)
    #     val_metrics = Metrics(acc, accmacro, f1macro, f1micro)
    #     test_metrics = Metrics(acc, accmacro, f1macro, f1micro)
    #     exp_results = ExpResults(train_metrics, val_metrics, test_metrics)
    #     self.result_dao.save_result(exp_results, self.args)

    def get_best_entry(self, metrics: str | list[str] = "val_acc") -> list:
        results = self.result_dao.get_best_entry_by_metrics(metrics)
        for r in results:
            logger.info(r)
        return results

    def get_all_results(self) -> list:
        results = self.result_dao.get_all_results()
        return results

    def to_csv(self, fn: str):
        ...

    def search(self):
        pass


class PersistenceFactory:
    def __init__(self, db_name: str, db_dir: str, create_db: bool = True) -> None:
        self.db_name = db_name
        self.db_dir = db_dir
        self.create_db = create_db

    def get_persistence(self) -> Persistence:
        databaseEngine = EngineFactory().get_sqlite_engine(
            self.db_name,
            self.db_dir,
            self.create_db,
        )
        rdao = ResultDaoFactory(databaseEngine).get_dao()
        return Persistence(rdao)
