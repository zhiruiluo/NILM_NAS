from __future__ import annotations

import logging

from sqlalchemy import func

from src.base_module.configs import ExpResults
from src.config_options.option_def import MyProgramArgs
from src.database.engine.SQLiteEngine import SQLiteEngine
from src.database.SQLiteMapper import MultilabelResults

logger = logging.getLogger(__name__)


class MultilabelResultDao:
    def __init__(self):
        pass

    def save_result(self, result: ExpResults, args: MyProgramArgs) -> None:
        pass

    def get_result(self, model, dataset) -> list[object]:
        pass

    def get_all_results(self) -> list[object]:
        pass

    def get_best_entry_by_metrics(self, metrics: str | list[str]):
        pass


class MultilabelResultSqliteDao(MultilabelResultDao):
    def __init__(self, sqlite_engine: SQLiteEngine):
        self.sqlite_engine = sqlite_engine

    def save_result(self, result: ExpResults, args: MyProgramArgs):
        result_ = MultilabelResults(
            model=args.expOption.model,
            model_params=args.modelConfig.dumps_json(),
            dataset=args.expOption.dataset,
            data_params=args.datasetConfig.dumps_json(),
            train_accmacro=result.train_metrics.acc,
            train_f1macro=result.train_metrics.f1macro,
            val_accmacro=result.val_metrics.acc,
            val_f1macro=result.val_metrics.f1macro,
            test_accmacro=result.test_metrics.acc,
            test_f1macro=result.test_metrics.f1macro,
            start_time=result.start_time,
            training_time=result.training_time,
            macs=result.macs,
            flops=result.flops,
            params=result.params.dumps_json(),
            nas_params=result.nas_params,
        )
        logger.info(f"[ResultSqlitDao] save results to {self.sqlite_engine.db_path}")
        logger.info(f"[ResultSqlitDao]  *********\n {result_} \n *********")
        self.sqlite_engine.insert(result_)

    # def get_best_config(self, model, dataset):

    def get_result(self, model: str, dataset: str):
        results = MultilabelResults(model=model, dataset=dataset)
        statm = self.sqlite_engine.get_statement_exact_match(results)
        objs = self.sqlite_engine.get_all(statm)
        # o: Results
        # for o in objs:
        #     print(o.params)
        logger.info(objs)
        return objs

    def get_all_results(self) -> list:
        results = MultilabelResults()
        statm = self.sqlite_engine.get_statement_exact_match(results)
        objs = self.sqlite_engine.get_all(statm)
        return objs

    def get_best_entry_by_metrics(self, metrics: list[str] | str):
        logger.info(metrics)
        if isinstance(metrics, list):
            statm = [MultilabelResults.id]
            for m in metrics:
                statm.append(func.max(getattr(MultilabelResults, m)))
            q = self.sqlite_engine.query(*statm)
        else:
            statm = [
                MultilabelResults.id,
                func.max(getattr(MultilabelResults, metrics)),
            ]
            q = self.sqlite_engine.query(*statm)
        statm = MultilabelResults.__table__.select().where(
            MultilabelResults.__table__.columns.id == q[0].id,
        )
        q = self.sqlite_engine.get_all(statm)
        return q


class ResultDaoFactory:
    def __init__(self, engine):
        self._engine = engine

    def get_dao(self):
        if self._engine.engine_name() == "SQLite":
            return MultilabelResultSqliteDao(self._engine)
