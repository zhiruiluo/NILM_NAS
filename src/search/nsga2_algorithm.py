from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem, Problem
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.crossover.pntx import PointCrossover
from pymoo.operators.sampling.rnd import BinaryRandomSampling, IntegerRandomSampling
from typing import Literal

def nsga2_algorithm_factory(pop_size=50, vtype: Literal["int", "binary"] = "binary"):
    if vtype == "int":
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=IntegerRandomSampling(),
            mutation=PolynomialMutation(at_least_once=True, vtype=int),
        )
    elif vtype == "binary":
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=BinaryRandomSampling(),
            mutation=BitflipMutation(prob_var=0.02),
            # crossover=SimulatedBinaryCrossover(prob_var=0.9)
            crossover = PointCrossover(n_points=2),
        )
    else:
        raise ValueError("vtype is not valid")

    return algorithm

