from typing import Any
import ray
import functools

def evaluate():
    ...
    
class Evaluate():
    def __init__(self) -> None:
        pass
    
    def __call__(self) -> Any:
        ...

def test_evaluate0():
    ray.remote(evaluate)
    
def test_evaluate1():
    e = Evaluate()
    
    ray.remote(e.__call__.__func__)