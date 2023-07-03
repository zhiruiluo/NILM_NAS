import dill
from sklearn.model_selection import ParameterGrid

class StateFullLoop():
    def __init__(self, parameter_grid: dict) -> None:
        self.parameter_grid = ParameterGrid(parameter_grid)
        self.length_grid = len(self.parameter_grid)
        self.iteration_idx = 0
    
    def run(self):
        for i in range(self.iteration_idx, self.length_grid):
            param = self.parameter_grid[i]
            print(param)
            self.iteration_idx = i
        
    def next(self):
        obj = self.parameter_grid[self.iteration_idx]
        self.iteration_idx += 1
        print(obj)
        
    def __getstate__(self) -> object:
        state = self.__dict__.copy()
        return state
    
    def save_checkpoint(self):
        return dill.dumps(self)
            
    @staticmethod
    def load_checkpoint(bits):
        return dill.loads(bits)


def test_resume_loop():
    loop = StateFullLoop({'a':[1,2], 'b': [3,4]})
    loop.next()
    loop.next()
    bits = loop.save_checkpoint()
    print('save')
    loop2 = StateFullLoop.load_checkpoint(bits)
    print('load')
    loop2.run()