from __future__ import annotations
import os
import sys
from typing import Any

sys.path.append('.')
import ray
from pymoo.optimize import minimize

from src.cluster_deploy import slurm_launch
from src.config_options import MyProgramArgs, OptionManager
from src.database.Persistence import PersistenceFactory
from src.project_logging import LoggerManager
from src.search.nas_problem import do_every_generations, get_tsnet_problem
from sklearn.model_selection import ParameterGrid
import copy
import dill
from pathlib import Path
import logging


logger = logging.getLogger(__name__)

POP_SIZE = 30
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

# class RayParallelization:
#     def __init__(self, job_resources={}, max_pending_task=2) -> None:
#         super().__init__()
#         self.job_resources = job_resources
#         self.result_refs = []
#         self.ready_results = []
#         self.max_num_pending_tasks = max_pending_task

#     def __call__(self, f, X):
#         runnable = ray.remote(f.__call__.__func__)
#         runnable = runnable.options(**self.job_resources)
#         for x in X:
#             if len(self.result_refs) > self.max_num_pending_tasks:
#                 ready_refs, self.result_refs = ray.wait(self.result_refs,
#                                                         num_returns=1)
#                 self.ready_results.append(ray.get(ready_refs))
#             self.result_refs.append(runnable.remote(f,x))
        
#         self.ready_results.append(ray.get(self.result_refs))
#         return self.ready_results

#     def __getstate__(self):
#         state = self.__dict__.copy()
#         return state

class StatefulLoop():
    def __init__(self, args: MyProgramArgs, parameter_grid: dict) -> None:
        self.parameter_grid = ParameterGrid(parameter_grid)
        self.length_grid = len(self.parameter_grid)
        self.iteration_idx = 0
        self.args = args
    
    def set_ckppath(self, path):
        self.path = path
    
    def _trainable(self, param):
        args = copy.deepcopy(self.args)
        args.datasetConfig.win_size = param['win_size']
        args.datasetConfig.house_no = param['house_no']
        
        searcher = Searcher(args)
        searcher.search()
        self.save_checkpoint()
    
    def run(self):
        for i in range(self.iteration_idx, self.length_grid):
            param = self.parameter_grid[i]
            self.iteration_idx = i
            self._trainable(param)
        
    def next(self):
        param = self.parameter_grid[self.iteration_idx]
        self._trainable(param)
        self.save_checkpoint()
        
    def __getstate__(self) -> object:
        state = self.__dict__.copy()
        return state

    def save_checkpoint(self):
        with open(self.path, 'wb') as f:
            dill.dump(self, f)
    
    @staticmethod
    def load_checkpoint(path) -> StatefulLoop:
        if os.path.isfile(path):
            with open(path, 'rb') as f:
                return dill.load(f)

class Searcher():
    def __init__(self, args: MyProgramArgs) -> None:
        self.args = args
    
    def search(self):
        args = self.args
        logger.info(f"ray init")
        if os.environ.get("head_node_ip", None):
            ray.init(
                address=args.systemOption.address,
                _node_ip_address=os.environ["head_node_ip"],
                # _temp_dir=args.systemOption.exp_dir,
            )
        else:
            ray.init(resources={})
    
        runner = RayParallelization(job_resources={
            'num_cpus': self.args.nasOption.num_cpus,
            'num_gpus': self.args.nasOption.num_gpus,
            'memory': 64*1024*1024*1024
        })
        
        problem, algorithm, checkpoint = get_tsnet_problem(
            args, pop_size=POP_SIZE, monitor=args.trainerOption.monitor, mode=args.trainerOption.mode, elementwise_runner=runner, debug=True)

        if checkpoint.has_checkpoint():
            algorithm = checkpoint.load_checkpoint()
            
        res = minimize(problem, algorithm, ('n_gen', N_GEN), seed=1,
                    verbose=True, save_history=True, callback=do_every_generations)
            
        print("Threads:", res.exec_time, "res", res.F)
        logger.info(f"Threads: {res.exec_time} res {res.F}")
        print("history:", res.history)
        logger.info(f"history: {res.history}")
        ray.shutdown()
        logger.info(f"ray shutdown")

    
def Search(args: MyProgramArgs, root_logger):
    root_logger.info(f"ray init")
    if os.environ.get("head_node_ip", None):
        ray.init(
            address=args.systemOption.address,
            _node_ip_address=os.environ["head_node_ip"],
            _temp_dir=args.systemOption.exp_dir,
        )
    else:
        ray.init(resources={})
        
    runner = RayParallelization(job_resources={
        'num_cpus': args.nasOption.num_cpus,
        'num_gpus': args.nasOption.num_gpus,
        'memory': 64*1024*1024*1024
    })
    problem, algorithm, checkpoint = get_tsnet_problem(
        args, pop_size=POP_SIZE, monitor=args.trainerOption.monitor, mode=args.trainerOption.mode, elementwise_runner=runner, debug=False)

    # if checkpoint.
    res = minimize(problem, algorithm, ('n_gen', N_GEN), seed=1,
                   verbose=True, save_history=True, callback=do_every_generations)
    print("Threads:", res.exec_time, "res", res.F)
    root_logger.info(f"Threads: {res.exec_time} res {res.F}")
    print("history:", res.history)
    root_logger.info(f"history: {res.history}")
    ray.shutdown()
    root_logger.info(f"ray shutdown")


@slurm_launch(
    exp_name='GEh2',
    num_nodes=4,
    num_gpus=2,
    partition='epscor',
    log_dir='logging/UKDALE_424/',
    load_env="conda activate p39c116\n"
    + "export OMP_NUM_THREADS=10\n"
    + "export PL_DISABLE_FORK=1",
    command_suffix="--address='auto' --exp_name={{EXP_NAME}}",
)
def main():
    opt = OptionManager()
    args: MyProgramArgs = opt.replace_params({
        "trainerOption.verbose": False,
        "trainerOption.monitor": 'val_f1macro',
        "trainerOption.mode": "max",
        "datasetConfig": "UKDALE_multilabel",
        "datasetConfig.stride": 30,
        "datasetConfig.imbalance_sampler": False,
        "datasetConfig.splits": '4:2:4',
        "modelConfig": "TSNet",
        "modelConfig.n_phases": 3,
        "modelConfig.n_ops": 5,
        "modelConfig.in_channels": 1,
        "modelConfig.out_channels": 32,
        "modelConfig.head_type": "ASL",
        "nasOption.enable": True,
        "nasOption.num_cpus": 8,
        "nasOption.num_gpus": 1,
        "nasOption.backend": "no_report",
        "modelBaseConfig.epochs": 25,
        "modelBaseConfig.patience": 25,
        "modelBaseConfig.label_mode": "multilabel",
        "modelBaseConfig.batch_size": 512,
        "modelBaseConfig.val_batch_size": 512,
        "modelBaseConfig.test_batch_size": 512,
        "modelBaseConfig.lr": 0.025,
        "modelBaseConfig.weight_decay": 1e-2,
        "modelBaseConfig.lr_scheduler": 'none',
    })
    
    persistance = PersistenceFactory(
        db_name=args.systemOption.db_name,
        db_dir=args.systemOption.db_dir,
    ).get_persistence()

    del persistance

    root_logger = LoggerManager.get_main_logger(args, "ray_parallel")
    root_logger.info(args)
    
    
    parameter_grid = {
        "win_size": [150,300],
        "house_no": [2],
    }
    
    ckp_path = Path(args.systemOption.exp_dir).joinpath('exp_checkpoint')
    ckp_path.mkdir(parents=True, exist_ok=True)
    path = ckp_path.joinpath('stateful_loop.ckpt')
    if path.is_file():
        loop = StatefulLoop.load_checkpoint(path)
        loop.set_ckppath(path)
    else:
        loop = StatefulLoop(args, parameter_grid)
        loop.set_ckppath(path)
    loop.run()


if __name__ == '__main__':
    main()