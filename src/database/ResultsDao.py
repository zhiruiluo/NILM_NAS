from __future__ import annotations

import logging

from sqlalchemy import func

from src.base_module.configs import ExpResults
from src.config_options.option_def import MyProgramArgs
from src.database.engine.SQLiteEngine import SQLiteEngine
from src.database.SQLiteMapper import Results

logger = logging.getLogger(__name__)


class ResultDao:
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


# class EnhancedJSONEncoder(json.JSONEncoder):
#     def default(self, o):
#         if dataclasses.is_dataclass(o):
#             return dataclasses.asdict(o)
#         elif isinstance(o, Namespace):
#             return vars(o)
#         return super().default(o)


class ResultSqliteDao(ResultDao):
    def __init__(self, sqlite_engine: SQLiteEngine):
        self.sqlite_engine = sqlite_engine

    def save_result(self, result: ExpResults, args: MyProgramArgs):
        result_ = Results(
            model=args.expOption.model,
            model_params=args.modelConfig.dumps_json(),
            dataset=args.expOption.dataset,
            data_params=args.datasetConfig.dumps_json(),
            train_acc=result.train_metrics.acc,
            train_f1macro=result.train_metrics.f1macro,
            train_f1micro=result.train_metrics.f1micro,
            val_acc=result.val_metrics.acc,
            val_f1macro=result.val_metrics.f1macro,
            val_f1micro=result.val_metrics.f1micro,
            test_acc=result.test_metrics.acc,
            test_f1macro=result.test_metrics.f1macro,
            test_f1micro=result.test_metrics.f1micro,
            start_time=result.start_time,
            training_time=result.training_time,
            flops=result.flops,
            params=result.params.dumps_json(),
            nas_params=result.nas_params,
        )
        logger.info(f"[ResultSqlitDao] save results to {self.sqlite_engine.db_path}")
        logger.info(f"[ResultSqlitDao]  *********\n {result_} \n *********")
        self.sqlite_engine.insert(result_)

    # def get_best_config(self, model, dataset):

    def get_result(self, model: str, dataset: str):
        results = Results(model=model, dataset=dataset)
        statm = self.sqlite_engine.get_statement_exact_match(results)
        objs = self.sqlite_engine.get_all(statm)
        # o: Results
        # for o in objs:
        #     print(o.params)
        logger.info(objs)
        return objs

    def get_all_results(self) -> list:
        results = Results()
        statm = self.sqlite_engine.get_statement_exact_match(results)
        objs = self.sqlite_engine.get_all(statm)
        return objs

    def get_best_entry_by_metrics(self, metrics: list[str] | str):
        logger.info(metrics)
        if isinstance(metrics, list):
            statm = [Results.id]
            for m in metrics:
                statm.append(func.max(getattr(Results, m)))
            q = self.sqlite_engine.query(*statm)
        else:
            statm = [
                Results.id,
                func.max(getattr(Results, metrics)),
            ]
            q = self.sqlite_engine.query(*statm)
        statm = Results.__table__.select().where(
            Results.__table__.columns.id == q[0].id,
        )
        q = self.sqlite_engine.get_all(statm)
        return q


class ResultDaoFactory:
    def __init__(self, engine):
        self._engine = engine

    def get_dao(self):
        if self._engine.engine_name() == "SQLite":
            return ResultSqliteDao(self._engine)
