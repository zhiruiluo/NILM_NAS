from pymoo.core.problem import DaskParallelization
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
import numpy as np
from loguru import logger
from typing import Literal
import json


def nsga2_algorithm_factory(pop_size=50, vtype: Literal['int','binary']='binary'):
    if vtype == 'int':
        algorithm = NSGA2(pop_size=pop_size,
                sampling=IntegerRandomSampling(),
                mutation=PolynomialMutation(at_least_once=True,vtype=int),
            )
    elif vtype == 'binary':
        algorithm = NSGA2(pop_size=pop_size,
                sampling=BinaryRandomSampling(),
                mutation=BitflipMutation(),
            )
    else:
        raise ValueError('vtype is not valid')
    
    return algorithm


def do_every_generations(algorithm):
    # this function will be call every generation
    # it has access to the whole algorithm class
    gen = algorithm.n_gen
    pop_var = algorithm.pop.get("X")
    pop_obj = algorithm.pop.get("F")

    # report generation info to files
    logger.info("generation = {}".format(gen))
    logger.info("population error: best = {}, mean = {}, "
                 "median = {}, worst = {}".format(np.min(pop_obj[:, 0]), np.mean(pop_obj[:, 0]),
                                                  np.median(pop_obj[:, 0]), np.max(pop_obj[:, 0])))
    logger.info("population complexity: best = {}, mean = {}, "
                 "median = {}, worst = {}".format(np.min(pop_obj[:, 1]), np.mean(pop_obj[:, 1]),
                                                  np.median(pop_obj[:, 1]), np.max(pop_obj[:, 1])))

def get_unique_task_id(gene: np.ndarray) -> str:
    return json.dumps(gene.tolist())

class NASProblem(ElementwiseProblem):
    def __init__(self, n_var=20, n_obj=1, n_constr=0, lb=None, ub=None, vtype=int, **kwargs):
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr,xl=lb, xu=ub, vtype=vtype,**kwargs)
        self._n_evaluated = 0
        self._evaluated_tasks = {}
        
    def _evaluate(self, x, out, *args, **kwargs):
        objs = np.full((1, self.n_obj), np.nan)
        unique_task_id = get_unique_task_id(x)
        
        self._n_evaluated += 1
        increm_id = self._n_evaluated
        if unique_task_id not in self._evaluated_tasks:
            objs[:,0] = np.sum(x)
            objs[:,1] = 100
            self._evaluated_tasks[unique_task_id] = objs
            print('first_run', objs)
        else:
            objs = self._evaluated_tasks[unique_task_id]
            print('evaluated objs', objs)
        
        out['F'] = self._evaluated_tasks[unique_task_id]
        
    
def test_nasproblem_binary():
    algorithm = nsga2_algorithm_factory(vtype='binary')
    n_var = 10
    problem = NASProblem(n_var=n_var, n_obj=2, lb=[0]*n_var,ub=[2]*n_var)
    res = minimize(problem, algorithm, ('n_gen',10), seed=1, verbose=False, save_history=True, callback=do_every_generations)
    print(res.X, res.F)

def test_nasproblem_integer():
    algorithm = nsga2_algorithm_factory(vtype='int')
    n_var = 10
    problem = NASProblem(n_var=n_var, n_obj=2, lb=[0]*n_var,ub=[10]*n_var)
    res = minimize(problem, algorithm, ('n_gen',10), seed=1, verbose=False, save_history=True, callback=do_every_generations)
    print(res.X, res.F)
    