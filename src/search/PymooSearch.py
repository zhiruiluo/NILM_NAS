import copy
import logging
import numpy as np
import pickle
from typing import Dict, List, Optional, Tuple, Union, Any

from ray.tune.result import DEFAULT_METRIC
from ray.tune.search.sample import (
    Categorical,
    Domain,
    Float,
    Integer,
    Quantized,
    LogUniform,
)
from ray.tune.search import (
    UNRESOLVED_SEARCH_SPACE,
    UNDEFINED_METRIC_MODE,
    UNDEFINED_SEARCH_SPACE,
    Searcher,
)
from ray.tune.search.variant_generator import parse_spec_vars
from ray.tune.utils import flatten_dict
from ray.tune.utils.util import is_nan_or_inf, unflatten_dict, validate_warmstart

from src.search.nas_problem import do_every_generations, get_tsnet_problem
from src.config_options import MyProgramArgs
from pymoo.optimize import minimize
from pymoo.core.algorithm import Algorithm

try:
    import pymoo as pm
except ImportError:
    pm = None
    
logger = logging.getLogger(__name__)

class ExampleSearch(Searcher):
    def __init__(self, args: MyProgramArgs, pop_size, n_gen, metric="mean_loss", mode="min", **kwargs):
        super(ExampleSearch, self).__init__(
            metric=metric, mode=mode, **kwargs)
        problem, algorithm: Algorithm = get_tsnet_problem(args, pop_size, metric, mode)
        
        
        algorithm.setup(problem, termination=('n_gen', n_gen), seed=1, verbose=True, save_history=True, callback=do_every_generations)
        
        self.algorithm = algorithm
        self.configurations = {}
        self.pending_tasks = []

    def spin_once(self):
        if len(self.pending_tasks) == 0:
            if not self.algorithm.has_next():
                raise StopIteration()
            else:
                infills = self.algorithm.infill()

    def suggest(self, trial_id):
        suggestion = self.spin_once()
        
        
        
        infills = self.algorithm.infill()
        if infills is None:
            self.algorithm.advance()
        
        self.configurations[trial_id] = infills
        return infills

    def on_trial_complete(self, trial_id, result, **kwargs):
        configuration = self.configurations[trial_id]
        if result and self.metric in result:
            self.optimizer.update(configuration, result[self.metric])

class PymooSearch(Searcher):
    def __init__(
        self,
        metric: str | None = None, 
        mode: str | None = None
    ):
        super().__init__(metric=metric, mode=mode, **kwargs)
        