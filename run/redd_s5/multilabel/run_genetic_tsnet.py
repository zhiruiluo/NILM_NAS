import sys

sys.path.append('.')
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
from dask.distributed import Client, LocalCluster
from src.search.nas_problem import get_tsnet_problem
import logging
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)


def cli():
    exp_name = "DASK"
    t = time.strftime("%m%d_%H%M", time.localtime())
    log_root = Path("logging/")
    log_directory = log_root.joinpath(
        "{}_{}".format(exp_name, t),
    )
    opt = OptionManager()
    opt.args.systemOption.exp_name = exp_name + "_" + t
    
    cluster = SLURMCluster(
        queue="epscor",
        # account='zluo_epscor',
        # n_workers=8,
        memory="8g",
        cores=8,
        # processes=8,
        walltime="24:00:00",
        log_directory=log_directory,
        job_directives_skip=["--mem"],
        job_extra_directives=["--gpus-per-task 1", "--mem-per-cpu 2G"],
        death_timeout=300,
        # worker_extra_args=["--death-timeout 300"]
    )

    cluster.scale(jobs=2)
    client = Client(cluster)
    print(client)
    main(client, opt)

def cli_local():
    exp_name = "DASK"
    t = time.strftime("%m%d_%H%M", time.localtime())
    log_root = Path("logging/")
    log_directory = log_root.joinpath(
        "{}_{}".format(exp_name, t),
    )
    opt = OptionManager()
    opt.args.systemOption.exp_name = exp_name + "_" + t
    cluster = LocalCluster()
    client = Client(cluster.scheduler.address)
    main(client, opt)
    

def main(client, opt):
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
    
    runner = DaskParallelization(client)
    problem, algorithm = get_tsnet_problem(args, pop_size=20, elementwise_runner=runner)
    res = minimize(problem, algorithm, ('n_gen', 10), seed=1)
    print("Threads:", res.exec_time, "res", res.F)
    client.close()
    print("DASK SHUTDOWN")

if __name__ == '__main__':
    cli_local()