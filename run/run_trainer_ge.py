from __future__ import annotations

import os
import sys

sys.path.append('.')
import copy
import logging
import time
from pathlib import Path

import dill
import ray
from pymoo.optimize import minimize
from sklearn.model_selection import ParameterGrid

from src.cluster_deploy import slurm_launch
from src.config_options import MyProgramArgs, OptionManager
from src.database.Persistence import PersistenceFactory
from src.project_logging import LoggerManager
from src.search.nas_problem_with_channel import get_tsnet_problem_with_channels
from src.search.utils import do_every_generations
from memory_profiler import profile
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
            rets = ray.get(futures)
            print(rets)
            logger.info(rets)
            return rets
        except ray.exceptions.RayTaskError as e:
            print(e)

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

class RayParallelization_:
    def __init__(self, job_resources={}, max_pending_task=2) -> None:
        super().__init__()
        self.job_resources = job_resources
        self.max_num_pending_tasks = max_pending_task
        self.max_retries = 3
        
        self.pending_refs = []
        self.ready_results = {}
        self.registered_tasks = []

    def _wait_refs(self, future_refs):
        ready_results = {}
        for i in range(self.max_retries+1):
            try:
                ready_refs, pending_refs = ray.wait(future_refs,
                                                        num_returns=1)
                ready_results[ready_refs[0]] = ray.get(ready_refs)[0]
                return ready_results, pending_refs
            except ray.exceptions.WorkerCrashedError:
                print(f'WorkerCrashedError catched! Runner is retrying for {i} time!')
                
            time.sleep(2)
        raise TimeoutError('ray.exceptions.WorkerCrashedError! Max retries reached!')
        
    def _preprocess(self):
        self.pending_refs = []
        self.ready_results = {}
    
    def _postprocess(self):
        self.registered_tasks = []
        self.pending_refs = []
        self.ready_results = {}

    def __call__(self, f, X):
        self._preprocess()
        
        runnable = ray.remote(f.__call__.__func__)
        runnable = runnable.options(**self.job_resources)
        
        if len(self.registered_tasks) == 0:
            for x in X:
                ref = runnable.remote(f,x)
                self.registered_tasks.append((ref, x))
        
        for ref, x in self.registered_tasks:
            if len(self.pending_refs) > self.max_num_pending_tasks:
                ready_results, self.pending_refs = self._wait_refs(self.pending_refs)
                self.ready_results.update(ready_results)
                print(f'[RayParallelization] pending tasks {len(self.pending_refs)} | finished tasks {len(self.ready_results)}')
            self.pending_refs.append(ref)
        
        while len(self.pending_refs) > 0:
            ready_results, self.pending_refs = self._wait_refs(self.pending_refs)
            self.ready_results.update(ready_results)
            print(f'[RayParallelization] pending tasks {len(self.pending_refs)} | finished tasks {len(self.ready_results)}')
        
        rets = []
        for ref, _ in self.registered_tasks:
            rets.append(self.ready_results[ref])
        
        self._postprocess()
        return rets

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

class StatefulLoop():
    def __init__(self, args: MyProgramArgs, parameter_grid: dict) -> None:
        self.parameter_grid = ParameterGrid(parameter_grid)
        self.length_grid = len(self.parameter_grid)
        self.iteration_idx = 0
        self.args = args
        self.path = None
    
    def set_ckppath(self, path):
        self.path = path
    
    def _trainable(self, param):
        args = copy.deepcopy(self.args)
        args.datasetConfig.win_size = param['win_size']
        args.datasetConfig.house_no = param['house_no']
        
        searcher = Searcher(args)
        searcher.search()
    
    def run(self):
        while self.iteration_idx < self.length_grid:
            param = self.parameter_grid[self.iteration_idx]
            self.iteration_idx += 1
            self._trainable(param)
            self.save_checkpoint()
        
    def next(self):
        param = self.parameter_grid[self.iteration_idx]
        self.iteration_idx += 1
        self._trainable(param)
        self.save_checkpoint()
        
    def __getstate__(self) -> object:
        state = self.__dict__.copy()
        return state

    def save_checkpoint(self):
        if self.path is None:
            return
        with open(self.path, 'wb') as f:
            dill.dump(self, f)
    
    @staticmethod
    def load_checkpoint(path, args: MyProgramArgs=None, parameter_grid: dict=None) -> StatefulLoop:
        if os.path.isfile(path):
            with open(path, 'rb') as f:
                obj = dill.load(f)
                if args: obj.args = args
                if parameter_grid: obj.parameter_grid = ParameterGrid(parameter_grid)
                return obj

class Searcher():
    def __init__(self, args: MyProgramArgs) -> None:
        self.args = args
    
    @profile
    def search(self):
        args = self.args
    
        # runner = RayParallelization(job_resources={
        #     'num_cpus': self.args.nasOption.num_cpus,
        #     'num_gpus': self.args.nasOption.num_gpus,
        #     'memory': 40*1024*1024*1024},
        #     # max_pending_task=10,
        # )
        
        problem, algorithm, checkpoint = get_tsnet_problem_with_channels(
            args, pop_size=POP_SIZE, monitor=args.trainerOption.monitor, mode=args.trainerOption.mode, debug=False)

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


def main():
    opt = OptionManager()
    args: MyProgramArgs = opt.replace_params({
        "trainerOption.verbose": True,
        "trainerOption.monitor": 'val_f1macro',
        "trainerOption.mode": "max",
        "datasetConfig": "REDD_multilabel",
        "datasetConfig.stride": 30,
        "datasetConfig.combine_mains": True,
        "datasetConfig.imbalance_sampler": False,
        "datasetConfig.splits": '4:2:4',
        "modelConfig": "TSNet",
        "modelConfig.n_phases": 3,
        "modelConfig.n_ops": 5,
        "modelConfig.in_channels": 1,
        "modelConfig.head_type": "ASL",
        "nasOption.enable": True,
        "nasOption.num_cpus": 8,
        "nasOption.num_gpus": 0,
        "nasOption.backend": "no_report",
        "modelBaseConfig.epochs": 1,
        "modelBaseConfig.patience": 1,
        "modelBaseConfig.label_mode": "multilabel",
        "modelBaseConfig.batch_size": 256,
        "modelBaseConfig.val_batch_size": 512,
        "modelBaseConfig.test_batch_size": 512,
        "modelBaseConfig.lr": 1e-3,
        "modelBaseConfig.weight_decay": 1e-3,
        "modelBaseConfig.lr_scheduler": 'none',
        "trainerOption.limit_train_batches": 0.1,
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
    
    
    parameter_grid = {
        "win_size": [60,150,300],
        "house_no": [1,3],
    }
    

    loop = StatefulLoop(args, parameter_grid)
    loop.run()


if __name__ == '__main__':
    main()