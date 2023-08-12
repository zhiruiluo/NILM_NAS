from __future__ import annotations

import json

import numpy as np
import os
import logging
from pymoo.config import Config
Config.warnings['not_compiled'] = False
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from src.trainer.TraningManager import trainable
from src.config_options import MyProgramArgs, OptionManager
import copy
from .utils import do_every_generations, Checkpoint
from .nsga2_algorithm import nsga2_algorithm_factory
from pathlib import Path
import gc

logger = logging.getLogger(__name__)

CHANNEL_BITS = 1
CHANNEL_MAP = {
    '0': 32,
    '1': 64
}
LSTM_BITS = 2
LSTM_MAP = {
    '0': 32,
    '1': 64
}

def get_unique_task_id(gene: np.ndarray) -> str:
    return json.dumps(gene.tolist())

def get_n_var_with_channels(n_phases: int, n_ops: int) -> int:
    return LSTM_BITS + CHANNEL_BITS*n_phases + n_phases*((n_ops * (n_ops-1)//2)+1 + n_ops*3)

def get_n_var_repeat_with_channels(n_phases: int, n_ops: int) -> int:
    return LSTM_BITS + CHANNEL_BITS*n_phases + (n_ops * (n_ops-1)//2)+1 + n_ops*3

def decode_channels(bit_string: str, n_phases: int):
    lstm_hidden = LSTM_MAP[bit_string[0]]
    lstm_out = LSTM_MAP[bit_string[1]]
    channel_bits = bit_string[LSTM_BITS: n_phases+LSTM_BITS]
    assert len(channel_bits) % n_phases == 0
    channels: list[int] = []
    for c in range(0, len(channel_bits), CHANNEL_BITS):
        channels.append(CHANNEL_MAP[channel_bits[c: c+CHANNEL_BITS]])
    rest_bit_string = bit_string[n_phases+2:]
    return [lstm_hidden, lstm_out], channels, rest_bit_string

class NASProblem(ElementwiseProblem):
    def __init__(
        self,
        args: MyProgramArgs, 
        n_var=20,
        n_obj=1,
        n_constr=0,
        lb=None,
        ub=None,
        vtype=int,
        monitor='val_acc',
        mode='max',
        checkpoint_callback = None,
        debug=False,
        **kwargs,
    ):
        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            n_constr=n_constr,
            xl=lb,
            xu=ub,
            vtype=vtype,
            **kwargs,
        )
        self.monitor = monitor
        self.mode = mode
        self.debug = debug
        self.args = args
        self.checkpoint_callback = checkpoint_callback
        logger.info('[nasproblem] init')

    def _evaluate(self, x, out, *args, **kwargs):
        logger.info('[nasproblem] evaluate')
        os.sched_setaffinity(0, list(range(os.cpu_count())))
        objs = np.full((1, self.n_obj), np.nan)
        unique_task_id = get_unique_task_id(x)
        algorithm = kwargs.get('algorithm')
        if algorithm is not None:
            n_gen = algorithm.n_gen
            
        logger.info(f'n_gen {n_gen}')
            
        config = {"args": copy.deepcopy(self.args)}
        lstm_params, out_channels, bit_string = decode_channels(''.join(['1' if i else '0' for i in x]), config['args'].modelConfig.n_phases)
        config["args"].modelConfig.out_channels = out_channels
        config["args"].modelConfig.bit_string = bit_string
        config['args'].modelConfig.lstm_hidden_features = lstm_params[0]
        config['args'].modelConfig.lstm_out_features = lstm_params[1]
        config['nas_params'] = {'n_gen': n_gen}
        if not self.debug:
            result = trainable(config)
        else:
            result = {}
            result[self.monitor] = 1
            result['flops'] = 1
        
        if self.mode == 'max':
            objs[:, 0] = 1 - result[self.monitor]
        elif self.mode == 'min':
            objs[:, 0] = result[self.monitor]
        objs[:, 1] = result['flops']
        print("first run objs", objs)
        

        out["F"] = objs
        if self.checkpoint_callback:
            self.checkpoint_callback.save_checkpoint(algorithm)
        
        del result
        del config
        gc.collect()

def get_tsnet_problem_with_channels_lstm(args: MyProgramArgs, pop_size: int, monitor: str, mode: str, elementwise_runner=None, checkpoint_flag: bool= True, debug=False):
    '''
    args: MyProgramArgs
    pop_size: int
    monitor: str
    mode: str
    elementwise_runner = None
    checkpoint_flag: bool = True
    debug: bool = False
    '''
    algorithm = nsga2_algorithm_factory(vtype="binary", pop_size=pop_size)
    n_var = get_n_var_with_channels(args.modelConfig.n_phases,args.modelConfig.n_ops)
    if checkpoint_flag:
        checkpoint = Checkpoint(Path(args.systemOption.exp_dir).joinpath('exp_checkpoint').as_posix(), 
                                f'nasproblem_{args.datasetConfig.house_no}_{args.datasetConfig.win_size}_checkpoint.ckp')
    else:
        checkpoint = None
    
    if elementwise_runner is None:
        problem = NASProblem(args, n_var=n_var, n_obj=2, lb=[0] * n_var, ub=[2] * n_var, monitor=monitor, mode=mode, checkpoint_callback=checkpoint, debug=debug)
    else:
        problem = NASProblem(args, n_var=n_var, n_obj=2, lb=[0] * n_var, ub=[2] * n_var, monitor=monitor, mode=mode, elementwise_runner=elementwise_runner, checkpoint_callback=checkpoint, debug=debug)
    return problem, algorithm, checkpoint


def test_nasproblem_binary():
    opt = OptionManager()
    args = opt.replace_params({'modelConfig': 'TSNet','datasetConfig':'REDD_multilabel'})
    problem, algorithm, checkpoint  = get_tsnet_problem_with_channels_lstm(args, pop_size=20, monitor='val_f1macro', mode='max', checkpoint_flag=False, debug=True)
    
    res = minimize(
        problem,
        algorithm,
        ("n_gen", 10),
        seed=1,
        verbose=False,
        save_history=True,
        callback=do_every_generations,
    )
    print(res.X, res.F)
