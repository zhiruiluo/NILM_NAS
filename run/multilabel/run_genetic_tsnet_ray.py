import sys
sys.path.append('.')
# sys.path.append(os.path.abspath('.'))
import os
import time
from pathlib import Path
import ray
from pymoo.core.problem import DaskParallelization
from pymoo.optimize import minimize

from src.config_options import MyProgramArgs, OptionManager
from src.database.Persistence import PersistenceFactory
from src.project_logging import LoggerManager
from src.search.nas_problem import get_tsnet_problem
from src.cluster_deploy import dask_launch, slurm_launch
# import logging
# logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)

class RayParallelization:
    def __init__(self, job_resources={}) -> None:
        super().__init__()
        self.job_resources = job_resources

    def __call__(self, f, X):
        runnable = ray.remote(f.__call__.__func__)
        runnable = runnable.options(**self.job_resources)
        futures = [runnable.remote(f,x) for x in X]
        return ray.get(futures)

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

@slurm_launch(
    exp_name='RAY_GE',
    num_nodes=4,
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
        "datasetConfig.win_size": 300,
        "datasetConfig.stride": 60,
        "datasetConfig.combine_mains": False,
        "modelConfig":"TSNet",
        "modelConfig.n_phases": 3,
        "modelConfig.n_ops": 3,
        "modelConfig.in_channels": 2,
        "modelConfig.out_channels": 32,
        "nasOption.enable": True,
        "nasOption.num_cpus": 8,
        "nasOption.num_gpus": 1,
        "nasOption.backend": "no_report",
        "modelBaseConfig.epochs": 30,
        "modelBaseConfig.patience": 30,
        "modelBaseConfig.label_mode": "multilabel",
        "modelBaseConfig.batch_size": 64,
        "modelBaseConfig.val_batch_size": 256,
        "modelBaseConfig.test_batch_size": 256,
        # "trainerOption.limit_train_batches": 0.1,
        # "trainerOption.limit_val_batches": 0.1,
        # "trainerOption.limit_test_batches": 0.1,
    })
    
    persistance = PersistenceFactory(
        db_name=args.systemOption.db_name,
        db_dir=args.systemOption.db_dir,
    ).get_persistence()

    del persistance

    root_logger = LoggerManager.get_main_logger(args, "ray_parallel")
    root_logger.info(args)
    
    if os.environ.get("head_node_ip", None):
        ray.init(
            address=args.systemOption.address,
            _node_ip_address=os.environ["head_node_ip"],
        )
    else:
        ray.init()
        
    runner = RayParallelization(job_resources={
        'num_cpus': args.nasOption.num_cpus,
        'num_gpus': args.nasOption.num_gpus,
    })
    problem, algorithm = get_tsnet_problem(args, pop_size=20, elementwise_runner=runner)
    res = minimize(problem, algorithm, ('n_gen', 5), seed=1)
    print("Threads:", res.exec_time, "res", res.F)

if __name__ == '__main__':
    main()