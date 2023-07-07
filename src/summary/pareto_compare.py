import sys

sys.path.append('.')
import json
import os
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
import simple_parsing as sp

from src.config_options import MyProgramArgs
from src.database.SQLiteMapper import Results
from src.summary.utils import read_all_results
import matplotlib.pyplot as plt
from pymoo.indicators.hv import Hypervolume

@dataclass
class Config:
    exp_dir: str
    main_model: str
    other_model: str
    x_axis: str = 'flops'
    y_axis: str = 'val_f1macro'

def get_pareto_front(Xs, Ys, maxX=True, maxY=True):
    sorted_list = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxX)
    pareto_front = [sorted_list[0]]
    for pair in sorted_list[1:]:
        if maxY:
            if pair[1] >= pareto_front[-1][1]:
                pareto_front.append(pair)
        else:
            if pair[1] <= pareto_front[-1][1]:
                pareto_front.append(pair)
    return pareto_front

def plot_pareto_frontier(Xs, Ys, maxX=True, maxY=True, ax=None):
    pareto_front = get_pareto_front(Xs, Ys, maxX, maxY)

    if ax == None:
        ax = plt.subplot()
    ax.scatter(Xs,Ys, s=4)
    pf_X = [pair[0] for pair in pareto_front]
    pf_Y = [pair[1] for pair in pareto_front]
    ax.plot(pf_X, pf_Y, 'r')
    ax.margins(0.2,0.1)
    ax.grid(visible=True, which='major', axis='both')
    ax.set_xscale('log')
    
    return ax

class ParetoCompare():
    def __init__(self, args: Config) -> None:
        self.args: Config = args
        self.df_all = self._load_all_results()
        self.plot_dir = Path('results/').joinpath(args.exp_dir.replace('/','_'))
    
    def _load_all_results(self):
        exp_dir = Path(self.args.exp_dir)
        results = read_all_results('nas_results.db', exp_dir.as_posix())
        dc: Dict[str, list] = defaultdict(list)
        for result in results:
            p: MyProgramArgs = result['params']
            r: Results = result['result']
            data_params = json.loads(r.data_params)
            nas_params = json.loads(r.nas_params)
            dc['exp'].append(exp_dir.name)
            dc['val_acc'].append(r.val_acc)
            dc['val_f1macro'].append(r.val_f1macro)
            dc['val_loss'].append(r.__getattribute__('val_loss'))
            dc['test_acc'].append(r.test_acc)
            dc['test_f1macro'].append(r.test_f1macro)
            dc['flops'].append(r.flops)
            dc['dataset'].append(r.dataset)
            dc['model'].append(r.model)
            dc['modelConfig'].append(p.modelConfig.dumps_json())
            dc['house_no'].append(data_params['house_no'])
            dc['win_size'].append(data_params['win_size'])
            dc['stride'].append(data_params['stride'])
            dc['training_time'].append(r.training_time)
            dc['n_gen'].append(nas_params.get('n_gen'))
            
        df = pd.DataFrame.from_dict(dc).reset_index(drop=True)

        return df
    
    def plot(self):
        for keys, df in self.df_all.groupby(by=['house_no','win_size']):
            Xs = df[self.args.x_axis].to_list()
            Ys = df[self.args.y_axis].to_list()