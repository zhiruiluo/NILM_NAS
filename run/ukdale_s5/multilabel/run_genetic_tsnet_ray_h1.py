import os
import sys

sys.path.append('.')
import ray
from pymoo.optimize import minimize

from src.cluster_deploy import slurm_launch
from src.config_options import MyProgramArgs, OptionManager
from src.database.Persistence import PersistenceFactory
from src.project_logging import LoggerManager
from src.search.nas_problem import do_every_generations, get_tsnet_problem

POP_SIZE = 20
N_GEN = 10


class RayParallelization:
    def __init__(self, job_resources={}) -> None:
        super().__init__()
        self.job_resources = job_resources

    def __call__(self, f, X):
        runnable = ray.remote(f.__call__.__func__)
        runnable = runnable.options(**self.job_resources)
        futures = [runnable.remote(f,x) for x in X]
        try:
            return ray.get(futures)
        except ray.exceptions.RayTaskError as e:
            print(e)

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

def loop(args: MyProgramArgs, root_logger):
    root_logger.info(f"ray init")
    if os.environ.get("head_node_ip", None):
        ray.init(
            address=args.systemOption.address,
            _node_ip_address=os.environ["head_node_ip"],
            _temp_dir=args.systemOption.exp_dir,
        )
    else:
        ray.init()
        
    runner = RayParallelization(job_resources={
        'num_cpus': args.nasOption.num_cpus,
        'num_gpus': args.nasOption.num_gpus,
    })
    problem, algorithm = get_tsnet_problem(
        args, pop_size=POP_SIZE, monitor='val_f1macro', mode='max', elementwise_runner=runner, debug=False)
    res = minimize(problem, algorithm, ('n_gen', N_GEN), seed=1,
                   verbose=True, save_history=True, callback=do_every_generations)
    print("Threads:", res.exec_time, "res", res.F)
    root_logger.info(f"Threads: {res.exec_time} res {res.F}")
    print("history:", res.history)
    root_logger.info(f"history: {res.history}")
    ray.shutdown()
    root_logger.info(f"ray shutdown")
    
    
@slurm_launch(
    exp_name='GEh1_new',
    num_nodes=4,
    num_gpus=2,
    partition='epscor',
    log_dir='logging/UKDALE/',
    load_env="conda activate p39c116\n"
    + "export OMP_NUM_THREADS=10\n"
    + "export PL_DISABLE_FORK=1",
    command_suffix="--address='auto' --exp_name={{EXP_NAME}}",
)
def main():
    opt = OptionManager()
    args: MyProgramArgs = opt.replace_params({
        "trainerOption.verbose": False,
        "datasetConfig": "UKDALE_multilabel",
        "datasetConfig.stride": 5,
        "datasetConfig.imbalance_sampler": False,
        "datasetConfig.random_sampler": True,
        "datasetConfig.random_sampler_ratio": 0.3,
        "modelConfig": "TSNet",
        "modelConfig.n_phases": 4,
        "modelConfig.n_ops": 5,
        "modelConfig.in_channels": 1,
        "modelConfig.out_channels": 16,
        "modelConfig.head_type": "ASL",
        "nasOption.enable": True,
        "nasOption.num_cpus": 16,
        "nasOption.num_gpus": 1,
        "nasOption.backend": "no_report",
        "modelBaseConfig.epochs": 20,
        "modelBaseConfig.patience": 20,
        "modelBaseConfig.label_mode": "multilabel",
        "modelBaseConfig.batch_size": 128,
        "modelBaseConfig.val_batch_size": 512,
        "modelBaseConfig.test_batch_size": 512,
        "modelBaseConfig.lr": 1e-3,
        "modelBaseConfig.weight_decay": 1e-3,
        "modelBaseConfig.lr_scheduler": 'none',
        # "trainerOption.limit_train_batches": 0.1,
        # "trainerOption.limit_val_batches": 0.1,
        # "trainerOption.limit_test_batches": 0.1,
        # "trainerOption.precision": 16,
    })
    
    persistance = PersistenceFactory(
        db_name=args.systemOption.db_name,
        db_dir=args.systemOption.db_dir,
    ).get_persistence()

    del persistance

    root_logger = LoggerManager.get_main_logger(args, "ray_parallel")
    root_logger.info(args)
    
    for win_size in [150]:
        args.datasetConfig.win_size = win_size
        for house_no in [1]:
            args.datasetConfig.house_no = house_no
            loop(args, root_logger)

if __name__ == '__main__':
    main()