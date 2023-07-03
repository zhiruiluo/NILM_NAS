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
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scienceplots
plt.style.use(["science","no-latex"])

@dataclass
class Config:
    exp_dir: str
    exp_names: List[str]
    select_size: int = 3
    hosue_no: int = 2
    win_size: int = 150
    monitor: str = 'f1macro'
    random_select: bool = False


def parse_random_rank_csv():
    with open('./run/ukdale/multilabel/random_rank.csv', mode='r') as fp:
        df_rank = pd.read_csv(fp)
    return df_rank
    

class TopRanking():
    def __init__(self, args: Config) -> None:
        self.args: Config = args
        self.df_all = self._load_all_results()
        self.plot_dir = Path('results/').joinpath(args.exp_dir.replace('/','_'))
    
    def _load_all_results(self):
        df_all = []
        exp_dir = Path(self.args.exp_dir)
        for fn in self.args.exp_names:
            results = read_all_results('nas_results.db', exp_dir.joinpath(fn).as_posix())
            dc: Dict[str, list] = defaultdict(list)
            for result in results:
                p: MyProgramArgs = result['params']
                r: Results = result['result']
                data_params = json.loads(r.data_params)
                nas_params = json.loads(r.nas_params)
                dc['exp'].append(fn)
                dc['train_acc'].append(r.train_acc)
                dc['train_f1macro'].append(r.train_f1macro)
                dc['train_loss'].append(r.train_loss)
                dc['val_acc'].append(r.val_acc)
                dc['val_f1macro'].append(r.val_f1macro)
                dc['val_loss'].append(r.val_loss)
                dc['test_acc'].append(r.test_acc)
                dc['test_f1macro'].append(r.test_f1macro)
                dc['test_loss'].append(r.test_loss)
                dc['flops'].append(r.flops)
                dc['dataset'].append(r.dataset)
                dc['model'].append(r.model)
                dc['modelConfig'].append(p.modelConfig.dumps_json())
                dc['out_channels'].append(p.modelConfig.out_channels)
                dc['bit_string'].append(p.modelConfig.bit_string)
                dc['house_no'].append(data_params['house_no'])
                dc['win_size'].append(data_params['win_size'])
                dc['stride'].append(data_params['stride'])
                dc['training_time'].append(r.training_time)
                dc['n_gen'].append(nas_params.get('n_gen'))
                dc['lr'].append(p.modelBaseConfig.lr)
                dc['batch_size'].append(p.modelBaseConfig.batch_size)
                
            df = pd.DataFrame.from_dict(dc)
            df_all.append(df)
        
        df_all = pd.concat(df_all).reset_index(drop=True)

        return df_all
    
    def get_top_rank(self):
        df = self.df_all
        df = df[df['win_size'] == self.args.win_size]
        df_ordered = df.sort_values(by=['val_'+self.args.monitor], ascending=False).reset_index(drop=True)
        if self.args.random_select:
            selection = np.sort(np.random.choice(df_ordered.shape[0], self.args.select_size))
        else:
            imax = 0
            imin = df_ordered.shape[0] - 1
            def idx(min,max, perc):
                return round((max-min)*perc + min)
            selection = np.array((imax, idx(imax, imin, 0.1), idx(imax, imin, 0.2), round(np.mean((imax,imin))), idx(imax, imin, 0.7), idx(imax, imin, 0.8), imin))
            print(selection)
        df_1 = df_ordered.iloc[selection,:].reset_index(drop=True)
        print(df_1)
        df_1.to_csv(Path(self.args.exp_dir).joinpath(self.args.exp_names[0]).joinpath('random_rank.csv'))
        df_2 = df_1.sort_values(by=[f'test_{self.args.monitor}'], ascending=False)
        print(np.array(df_2.index))
        tau, p_value = stats.kendalltau(np.arange(df_1.shape[0]), np.array(df_2.index))
        print(tau, p_value)
        res = stats.spearmanr(np.arange(df_1.shape[0]), np.array(df_2.index))
        print(res.statistic, res.pvalue)
    
    def compare(self):
        df = self.df_all
        df_rank = parse_random_rank_csv()
        bit_string_l = []
        for cf in df_rank['modelConfig']:
            conf = json.loads(cf)
            bit_string_l.append(conf['bit_string'])
        
        for out_channels, df_out in df.groupby(by=['out_channels']):
            df_1 = df_out.sort_values(by='bit_string',key=lambda x: x.map({k: i for i, k in enumerate(bit_string_l)})).reset_index(drop=True)
            df_2 = df_1.sort_values(by=[f'test_{self.args.monitor}'], ascending=False)
            print(df_2.index)
            tau, p_value = stats.kendalltau(np.arange(df_1.shape[0]), np.array(df_2.index))
            print(out_channels)
            print(tau, p_value)
            res = stats.spearmanr(np.arange(df_1.shape[0]), np.array(df_2.index))
            print(res.statistic, res.pvalue)
            
    def compare_2(self):
        df = self.df_all
        ranks = []
        for keys, d in df.groupby(['out_channels']):
            print(keys, d.shape[0])
            d_sort = d.sort_values(['bit_string']).reset_index(drop=True)
            # print(d_sort['bit_string'])
            d1 = d_sort.sort_values([f'test_{self.args.monitor}'], ascending=False)
            ranks.append(d1.index)
        
        for i in range(len(ranks)-1):
            tau, p_value = stats.kendalltau(ranks[i], ranks[i+1])
            print(tau, p_value)
            res = stats.spearmanr(ranks[i], ranks[i+1])
            print(res.statistic, res.pvalue)
        
    def compare_3(self):
        df = self.df_all
        ranks = []
        for keys, d in df.groupby(['out_channels']):
            print(keys, d.shape[0])
            d_sort = d.sort_values([f'val_{self.args.monitor}'], ascending=False).reset_index(drop=True)
            d1 = d_sort.sort_values([f'test_{self.args.monitor}'], ascending=False)
            ranks.append([d_sort.index, d1.index])
        
        for i in range(len(ranks)):
            tau, p_value = stats.kendalltau(ranks[i][0], ranks[i][1])
            print(tau, p_value)
            res = stats.spearmanr(ranks[i][0], ranks[i][1])
            print(res.statistic, res.pvalue)
        
    def plot(self):
        df = self.df_all.sort_values(['bit_string'])
        print(df)
        points = []
        bits_id = {}
        i = 0
        for df_bs_oc in df.groupby(['bit_string','batch_size']):
            keys, dd = df_bs_oc
            if keys[0] not in bits_id:
                bits_id[keys[0]] = i
                i += 1

            Y = dd[f'test_{self.args.monitor}'].iloc[0]
            X = dd['batch_size'].iloc[0]
            print(Y,X)
            points.append([X, Y, bits_id[keys[0]], keys[1],keys[0]])
        
        values = []
        for keys, dd in df.groupby(['out_channels']):
            
            df_sorted = dd.sort_values('bit_string').groupby(['bit_string'])[f'test_{self.args.monitor}'].mean().reset_index(drop=True)
            # print(df_sorted)
            ranks = df_sorted.sort_values().index
            # print(ranks)
            values.append(np.array(ranks))
        
        print(np.vstack(values).sum(axis=0))
            
        fig, ax = plt.subplots(1,1, figsize=(10,10))
        print(mcolors.TABLEAU_COLORS)
        color_keys = list(mcolors.TABLEAU_COLORS.keys())
        points = pd.DataFrame(points)
        # for p in points:
        #     ax.scatter(p[3],p[1], label=p[2], c=mcolors.TABLEAU_COLORS[color_keys[p[2]]])
        marker_style = ['.','o','v','s','1','8','p']
        for k, p in points.groupby(by=[2]):
            ax.plot(p.iloc[:,0], p.iloc[:,1], label=p.iloc[0,4], marker=marker_style[k[0]])
        # ax.set_xscale(value='log')
        ax.legend()
        
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        path = self.plot_dir.joinpath(f'{self.args.exp_names[0]}_curve.png')
        plt.savefig(path,dpi=800)
        print(f'save path {path}')
    
def parse_config():
    args = sp.parse(Config)
    return args

if __name__ == '__main__':
    args = parse_config()
    gp = TopRanking(args)
    gp.plot()