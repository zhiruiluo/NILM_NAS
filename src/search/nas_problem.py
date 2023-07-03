from __future__ import annotations

import json

import numpy as np
import os
import logging
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from src.trainer.TraningManager import trainable
from src.config_options import MyProgramArgs, OptionManager
import copy
from .utils import do_every_generations, Checkpoint
from .nsga2_algorithm import nsga2_algorithm_factory
from pathlib import Path

logger = logging.getLogger(__name__)


def get_unique_task_id(gene: np.ndarray) -> str:
    return json.dumps(gene.tolist())

def get_n_var(n_phases: int, n_ops: int) -> int:
    return n_phases*((n_ops * (n_ops-1)//2)+1 + n_ops*3)

def get_n_var_repeat(n_ops: int) -> int:
    return (n_ops * (n_ops-1)//2)+1 + n_ops*3

class NASProblem(ElementwiseProblem):
    def __init__(
        self,
        args: MyProgramArgs, 
        n_var=20,
        n_obj=1,
        n_constr=0,
        lb=None,
        ub=None,
        vtype=int,
        monitor='val_acc',
        mode='max',
        checkpoint_callback = None,
        debug=False,
        **kwargs,
    ):
        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            n_constr=n_constr,
            xl=lb,
            xu=ub,
            vtype=vtype,
            **kwargs,
        )
        self.monitor = monitor
        self.mode = mode
        self.debug = debug
        self.args = args
        self._n_evaluated = 0
        self._evaluated_tasks = {}
        self.checkpoint_callback = checkpoint_callback
        logger.info('[nasproblem] init')

    def _evaluate(self, x, out, *args, **kwargs):
        logger.info('[nasproblem] evaluate')
        os.sched_setaffinity(0, list(range(os.cpu_count())))
        objs = np.full((1, self.n_obj), np.nan)
        unique_task_id = get_unique_task_id(x)
        algorithm = kwargs.get('algorithm')
        if algorithm is not None:
            n_gen = algorithm.n_gen
            
        print('n_gen', n_gen)
        self._n_evaluated += 1
        increm_id = self._n_evaluated
        if unique_task_id not in self._evaluated_tasks:
            
            self._evaluated_tasks[unique_task_id] = objs
            config = {"args": copy.deepcopy(self.args)}
            config["args"].modelConfig.bit_string = ''.join(['1' if i else '0' for i in x])
            config['nas_params'] = {'n_gen': n_gen}
            if not self.debug:
                result = trainable(config)
            else:
                result = {}
                result[self.monitor] = 1
                result['flops'] = 100
            
            if self.mode == 'max':
                objs[:, 0] = 1 - result[self.monitor]
            elif self.mode == 'min':
                objs[:, 0] = result[self.monitor]
            objs[:, 1] = result['flops']
            print("first run objs", objs)
        else:
            objs = self._evaluated_tasks[unique_task_id]
            print("evaluated objs", objs)

        out["F"] = self._evaluated_tasks[unique_task_id]
        if self.checkpoint_callback:
            self.checkpoint_callback.save_checkpoint(algorithm)
        
        
def get_tsnet_problem(args: MyProgramArgs, pop_size: int, monitor: str, mode: str, elementwise_runner=None, debug=False):
    algorithm = nsga2_algorithm_factory(vtype="binary", pop_size=pop_size)
    n_var = get_n_var(args.modelConfig.n_phases,args.modelConfig.n_ops)
    checkpoint = Checkpoint(Path(args.systemOption.exp_dir).joinpath('exp_checkpoint').as_posix(), 
                            f'nasproblem_{args.datasetConfig.house_no}_{args.datasetConfig.win_size}_checkpoint.ckp')
    
    if elementwise_runner is None:
        problem = NASProblem(args, n_var=n_var, n_obj=2, lb=[0] * n_var, ub=[2] * n_var, monitor=monitor, mode=mode, checkpoint_callback=checkpoint, debug=debug)
    else:
        problem = NASProblem(args, n_var=n_var, n_obj=2, lb=[0] * n_var, ub=[2] * n_var, monitor=monitor, mode=mode, elementwise_runner=elementwise_runner, checkpoint_callback=checkpoint, debug=debug)
    return problem, algorithm, checkpoint


def test_nasproblem_binary():
    import pickle
    opt = OptionManager()
    args = opt.replace_params({'modelConfig': 'TSNet','datasetConfig':'REDD_multilabel'})
    algorithm = nsga2_algorithm_factory(vtype="binary")
    n_var = get_n_var(args.modelConfig.n_phases,args.modelConfig.n_ops)
    problem = NASProblem(args, n_var=n_var, n_obj=2, lb=[0] * n_var, ub=[2] * n_var, debug=True)
    a = pickle.dumps(problem)
    res = minimize(
        problem,
        algorithm,
        ("n_gen", 10),
        seed=1,
        verbose=False,
        save_history=True,
        callback=do_every_generations,
    )
    print(res.X, res.F)


def test_nasproblem_integer():
    algorithm = nsga2_algorithm_factory(vtype="int")
    n_var = 10
    problem = NASProblem(n_var=n_var, n_obj=2, lb=[0] * n_var, ub=[10] * n_var)
    res = minimize(
        problem,
        algorithm,
        ("n_gen", 10),
        seed=1,
        verbose=False,
        save_history=True,
        callback=do_every_generations,
    )
    print(res.X, res.F)
