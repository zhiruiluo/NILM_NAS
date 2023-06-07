from __future__ import annotations

import dataclasses

from simple_parsing import Serializable

@dataclasses.dataclass
class RayOptions(Serializable):
    backend: str = "ray_tune"
    num_gpus: int = 1
    num_cpus: int = 8


@dataclasses.dataclass
class DaskOptions(Serializable):
    backend: str = 'dask'
    num_gpus: int = 1
    num_cpus: int = 8
    

@dataclasses.dataclass
class BO(Serializable):
    utility_kind: str = "ucb"
    kappa: float = 20
    kappa_decay: float = 0.98
    xi: float = 0.0
    patience: int = 5


@dataclasses.dataclass
class NASOption(Serializable):
    enable: bool = False
    backend: str = "ray_tune"  # choices: 'ray_tune'
    search_strategy: str = "random_search"
    num_samples: int = 100
    num_gpus: int = 1
    num_cpus: int = 8


@dataclasses.dataclass
class Genetic_NSGA2(Serializable):
    n_gen: int = 10
    n_eval: int = 150
    pop_size: int = 20
    seed: int = 1