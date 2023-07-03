from __future__ import annotations
import numpy as np
import dill
from pathlib import Path
import os
import logging

logger = logging.getLogger(__name__)

def do_every_generations(algorithm):
    # this function will be call every generation
    # it has access to the whole algorithm class
    gen = algorithm.n_gen
    pop_var = algorithm.pop.get("X")
    pop_obj = algorithm.pop.get("F")

    # report generation info to files
    logger.info(f"generation = {gen}")
    logger.info(
        "population error: best = {}, mean = {}, "
        "median = {}, worst = {}".format(
            np.min(pop_obj[:, 0]),
            np.mean(pop_obj[:, 0]),
            np.median(pop_obj[:, 0]),
            np.max(pop_obj[:, 0]),
        ),
    )
    logger.info(
        "population complexity: best = {}, mean = {}, "
        "median = {}, worst = {}".format(
            np.min(pop_obj[:, 1]),
            np.mean(pop_obj[:, 1]),
            np.median(pop_obj[:, 1]),
            np.max(pop_obj[:, 1]),
        ),
    )


class Checkpoint():
    def __init__(self, dir: str, fn: str, termination = None) -> None:
        self.dir = Path(dir)
        self.file_path = self.dir.joinpath(fn)
        self._termination = termination
    
    def save_checkpoint(self, algorithm):
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.file_path, 'wb') as f:
            dill.dump(algorithm, f)
    
    def has_checkpoint(self) -> bool:
        if os.path.isfile(self.file_path):
            return True
        return False
    
    def load_checkpoint(self):
        with open(self.file_path, 'rb') as f:
            checkpoint = dill.load(f)
        print(f"Loaded Checkpint from {self.file_path}:", checkpoint)
        if self._termination:
            checkpoint.termination = self._termination
        return checkpoint