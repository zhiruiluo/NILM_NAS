import sys
sys.path.append('.')
# sys.path.append(os.path.abspath('.'))
import os
import time
from pathlib import Path
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from distributed import Client
from pymoo.core.problem import DaskParallelization
from pymoo.optimize import minimize

from src.config_options import MyProgramArgs, OptionManager
from src.database.Persistence import PersistenceFactory
from src.project_logging import LoggerManager
from src.search.nas_problem import get_tsnet_problem
from src.cluster_deploy import dask_launch
# import logging
# logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)

class DaskParallelization:
    def __init__(self, client) -> None:
        super().__init__()
        self.client = client

    def __call__(self, f, X):
        jobs = [self.client.submit(f, x) for x in X]
        return [job.result() for job in jobs]

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("client", None)
        return state

@dask_launch(
    exp_name='DASK_GE',
    num_nodes=1,
    num_gpus=2,
    partition='epscor',
    load_env="conda activate p39c116\n"
    + "export OMP_NUM_THREADS=10\n"
    + "export PL_DISABLE_FORK=1",
    command_suffix="--address='auto' --exp_name={{EXP_NAME}}",
)
def main():
    opt = OptionManager()
    args: MyProgramArgs = opt.replace_params({
        "datasetConfig": "REDD_multilabel",
        # "datasetConfig.win_size": 600,
        # "datasetConfig.stride": 1,
        "modelConfig": "TSNet",
        "modelConfig.n_phases": 3,
        "modelConfig.n_ops": 3,
        "nasOption.enable": True,
        "nasOption.num_cpus": 8,
        "nasOption.num_gpus": 1,
        "nasOption.backend": "no_report",
        "modelBaseConfig.label_mode": "multilabel",
        # "trainerOption.limit_train_batches": 0.1,
        # "trainerOption.limit_val_batches": 0.1,
        # "trainerOption.limit_test_batches": 0.1,
    })
    
    persistance = PersistenceFactory(
        db_name=args.systemOption.db_name,
        db_dir=args.systemOption.db_dir,
    ).get_persistence()

    del persistance

    root_logger = LoggerManager.get_main_logger(args, "dask_parallel")
    root_logger.info(args)
    
    if os.environ.get("ip_head", None):
        client = Client(address=os.environ.get("ip_head", None))
    else:
        client = Client()
        
    runner = DaskParallelization(client)
    problem, algorithm = get_tsnet_problem(args, pop_size=20, elementwise_runner=runner)
    res = minimize(problem, algorithm, ('n_gen', 10), seed=1)
    print("Threads:", res.exec_time, "res", res.F)
    client.close()
    print("DASK SHUTDOWN")

if __name__ == '__main__':
    main()