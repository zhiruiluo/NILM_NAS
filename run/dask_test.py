from __future__ import annotations

import sys

from dask import compute
from dask import delayed
from dask.distributed import Client
from dask_jobqueue import SLURMCluster

# Input arguments
# project_name = sys.argv[1]
# num_of_worker_jobs = int(sys.argv[2])

# Create the SLURMCluster and define what resources to ask for each of the worker job.
# Notice the local_directory and python, python path must be adjusted to used module.
# To find out Python path, run:
# module load xx
# which python

cluster = SLURMCluster(
    queue='small',
    # project = project_name,
    cores=2,
    memory='8GB',
    walltime='00:10:00',
    interface='ib0',
    # local_directory = "/scratch/<YOUR-PROJECT>/temp",
    job_extra_directives=['--partition epscor'],
    # python = "/appl/soft/ai/cont_conda/python-data-2022-04-ubi8.5/bin/python"
)

# This launches the cluster (submits the worker jobs)
cluster.scale(1)
print(cluster.job_script())
client = Client(cluster)

list_of_delayed_functions = []
datasets = ['/data/dataset1', '/data/dataset2',
            '/data/dataset3', '/data/dataset4']


def processDataset(dataset):
    # Do something to the dataset
    results = dataset
    return results


for dataset in datasets:
    list_of_delayed_functions.append(delayed(processDataset)(dataset))

# This starts the execution with the resources available
compute(list_of_delayed_functions)
