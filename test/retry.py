import numpy as np
import os
import ray
import time

ray.init(ignore_reinit_error=True)

@ray.remote
def potentially_fail(failure_probability):
    time.sleep(0.2)
    if np.random.random() < failure_probability:
        print('worker died')
        os._exit(0)
    return 0


ray.get(potentially_fail.remote(1))
# for _ in range(1):
#     try:
#         # If this task crashes, Ray will retry it up to one additional
#         # time. If either of the attempts succeeds, the call to ray.get
#         # below will return normally. Otherwise, it will raise an
#         # exception.
#         ray.get(potentially_fail.remote(1))
#         print('SUCCESS')
#     except ray.exceptions.WorkerCrashedError as e:
#         print('FAILURE')
#         print(e)