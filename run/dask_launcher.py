from dask_jobqueue import SLURMCluster
from distributed import Client
from dask import delayed, compute
from pathlib import Path
import time
import dask
import torch
import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.core.problem import DaskParallelization
from pymoo.core.problem import Problem
from dask.distributed import Client
from pymoo.config import Config
Config.warnings['not_compiled'] = False

import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

def cli():
    exp_name = 'DASK'
    log_root = Path('logging/')
    log_directory = log_root.joinpath('{}_{}'.format( exp_name, time.strftime("%m%d_%H%M", time.localtime())))

    cluster = SLURMCluster(
        queue='epscor',
        # account='zluo_epscor',
        # n_workers=8,
        memory='8g',
        cores=8,
        # processes=8,
        walltime='24:00:00',
        log_directory=log_directory,
        job_directives_skip=['--mem'],
        job_extra_directives = ["--gpus-per-task 1", "--mem-per-cpu 2G"],
        # worker_extra_args=["--resources GPU=1"]
    )

    cluster.scale(jobs=2)
    # cluster.adapt(minimum_jobs=1, maximum_jobs=6)
    # print(cluster.job_script())
    # exit()
    client = Client(cluster)
    
    main(client)
    
def main(client):
    print("DASK STARTED")

    # initialize the thread pool and create the runner
    runner = DaskParallelization(client)

    class MyProblem(Problem):

        def __init__(self, **kwargs):
            super().__init__(n_var=10, n_obj=1, n_ieq_constr=0, xl=-5, xu=5, **kwargs)

        def _evaluate(self, x, out, *args, **kwargs):
            # print(x, out)
            # time.sleep(1)
            
            out["F"] = np.sum(x ** 2, axis=1)

    print(torch.cuda.is_available())
    problem = MyProblem(elementwise_runner=runner)
    res = minimize(problem, GA(), termination=("n_gen", 200), seed=1)
    print('Threads:', res.exec_time, "res", res.F)


    client.close()
    print("DASK SHUTDOWN")
    
cli()