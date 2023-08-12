from __future__ import annotations
import sys

sys.path.append('.')
import json
import os
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import datetime

import pandas as pd
import simple_parsing as sp

from src.config_options import MyProgramArgs
from src.database.SQLiteMapper import Results
from src.summary.utils import read_all_results, get_pareto_front_df
from src.summary.confusion_matrix import plot_confusion_matrix_api
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science','ieee','no-latex'])

from pymoo.indicators.hv import Hypervolume


@dataclass
class Config:
    output: str
    exp_dir: str = None
    exp_names: List[str] = None
    task: str = ''
    x_axis: str = 'flops'
    y_axis: str = 'val_f1macro'
    format: str = 'png'
    mean_repeat: bool = False

Model_Name_Mapping = {
    'BitcnNILM': 'BitcnNILM',
    'CNN_LSTM': 'CNNLSTM',
    'LSTM_AE': 'LSTMAE',
    'TSNet': 'NILM-NAS',
    'KNC': 'MLkNN',
    'MLSVM': 'MLSVM',
}


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

def add_mid_point_pareto_front(pareto_front: list, maxX=True, maxY=True):
    new_pareto_front =[]
    for i in range(len(pareto_front)-1):
        a = pareto_front[i] 
        b = pareto_front[i+1]
        new_pareto_front.append(a)
        if not maxX and maxY:
            new_pareto_front.append([b[0],a[1]])
    new_pareto_front.append(pareto_front[-1])
    return new_pareto_front

def get_hypervolume_front(Xs, Ys, reference_point, maxX=True, maxY=True):
    pareto_front = get_pareto_front(Xs, Ys, maxX, maxY)
    pareto_front = add_mid_point_pareto_front(pareto_front, maxX, maxY)
    if not maxX and maxY:
        pareto_front = [[pareto_front[0][0], reference_point[1]]] + pareto_front
        pareto_front = pareto_front + [[reference_point[0], pareto_front[-1][1]]]
    return pareto_front

def plot_pareto_frontier(Xs, Ys, maxX=True, maxY=True, ax=None, color='r',label=None):
    pareto_front = get_pareto_front(Xs, Ys, maxX, maxY)
    pareto_front = add_mid_point_pareto_front(pareto_front, maxX, maxY)
    if ax == None:
        ax = plt.subplot()
    handle1 = ax.scatter(Xs,Ys, s=4, color=color, label=label)
    pf_X = [pair[0] for pair in pareto_front]
    pf_Y = [pair[1] for pair in pareto_front]
    handle2 = ax.plot(pf_X, pf_Y, color=color, label=label)
    # ax.margins(0.2,0.1)
    ax.grid(visible=True, which='major', axis='both')
    ax.set_xscale('log')
    return ax



class GenerationPlot():
    def __init__(self, args: Config) -> None:
        self.args: Config = args
        self.df_all = self._load_all_results()
        if args.exp_dir:
            self.plot_dir = Path('results').joinpath(args.output).joinpath(Path(args.exp_dir).name.replace('/','_'))
        elif args.exp_names:
            self.plot_dir = Path('results').joinpath(args.output)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        # self.plot_dir = Path('results/').joinpath(Path(args.exp_dir).as_posix().replace('/','_'))
    
    def _results_to_pd(self, results):
        dc: Dict[str, list] = defaultdict(list)
        for result in results:
            p: MyProgramArgs = result['params']
            r: Results = result['result']
            data_params = json.loads(r.data_params)
            if r.nas_params:
                nas_params = json.loads(r.nas_params)
            else:
                nas_params = {}
            dc['exp'].append(p.systemOption.exp_name)
            dc['val_acc'].append(r.val_acc)
            dc['val_f1macro'].append(r.val_f1macro)
            dc['val_loss'].append(r.__getattribute__('val_loss'))
            dc['val_confmx'].append(r.val_confmx)
            dc['test_acc'].append(r.test_acc)
            dc['test_f1macro'].append(r.test_f1macro)
            dc['test_confmx'].append(r.test_confmx)
            dc['flops'].append(r.flops)
            dc['dataset'].append(r.dataset)
            dc['model'].append(r.model)
            dc['modelConfig'].append(p.modelConfig.dumps_json())
            dc['house_no'].append(data_params['house_no'])
            dc['win_size'].append(data_params['win_size'])
            dc['stride'].append(data_params['stride'])
            dc['training_time'].append(datetime.datetime.strptime(r.training_time, '%Y-%m-%d %H:%M:%S.%f')-datetime.datetime(1970,1,1))
            dc['n_gen'].append(nas_params.get('n_gen'))
            
        df = pd.DataFrame.from_dict(dc).reset_index(drop=True)
        return df
    
    def _load_all_results(self):
        if self.args.exp_names:
            df_all = []
            for fn in self.args.exp_names:
                results = read_all_results('nas_results.db',fn)
                if results is None:
                    print('empty file', fn)
                    exit()
                print('read from', fn)
                df_all.append(self._results_to_pd(results))    
                # if fn == 'logging/REDD_424/TSNET_pareto_0706_1652':
                #     print(df_all[-1]['model'])
                #     exit()
            df_results = pd.concat(df_all).reset_index(drop=True)
        elif self.args.exp_dir and Path(self.args.exp_dir).joinpath('nas_results.db').is_file():
            results = read_all_results('nas_results.db', Path(self.args.exp_dir).as_posix())
            df_results = self._results_to_pd(results)
        else:
            df_all = []
            exp_dir = Path(self.args.exp_dir)
            for fn in os.listdir(exp_dir):
                results = read_all_results('nas_results.db',exp_dir.joinpath(fn).as_posix())
                df_all.append(self._results_to_pd(results))
                
            df_results = pd.concat(df_all).reset_index(drop=True)
        return df_results
        
    def plot(self):
        for keys, df in self.df_all.groupby(by=['house_no','win_size']):
            print('house_no', keys[0], 'win_size', keys[1])
            nrows = 6
            ncols = 6
            fig, axes = plt.subplots(nrows=nrows,ncols=ncols, figsize=(20,20), sharex=True, sharey=True)
            # for i, gen in enumerate(df['n_gen'].unique()):
            for i, gp in enumerate(df.groupby(by=['n_gen'])):
                gen_key, df_gen = gp
                # df = df[df['n_gen'] == gen]
                Xs = df_gen[self.args.x_axis].to_list()
                Ys = df_gen[self.args.y_axis].to_list()
                print(gen_key, len(Xs), len(Ys), df_gen.shape)
                ax = axes[i//ncols][i%ncols]
                ax = plot_pareto_frontier(Xs, Ys, maxX=False, maxY=True, ax=ax)
                ax.set_title(f'generation {i+1}')
            
            Xs = df[self.args.x_axis].to_list()
            Ys = df[self.args.y_axis].to_list()
            i += 1
            ax = axes[i//ncols][i%ncols]
            ax = plot_pareto_frontier(Xs, Ys, maxX=False, maxY=True, ax=ax)
            ax.set_title(f'all generation')
            
            fig.tight_layout()
            self.plot_dir.mkdir(parents=True, exist_ok=True)
            path = self.plot_dir.joinpath(f'pareto_frontier_hn={keys[0]}_ws={keys[1]}.png')
            fig.savefig(path)
            print(f'save fig {path}')
            plt.close()
        
    def hypervolume(self):
        for keys, df in self.df_all.groupby(by=['house_no','win_size']):
            df[self.args.x_axis] = (df[self.args.x_axis] - df[self.args.x_axis].min()) / (df[self.args.x_axis].max() - df[self.args.x_axis].min())
            df[self.args.y_axis] = 1- (df[self.args.y_axis] - df[self.args.y_axis].min()) / (df[self.args.y_axis].max() - df[self.args.y_axis].min())
            print(keys)
            ref_point = np.array([1, 1])
            ind = Hypervolume(ref_point=ref_point)
            hvs = []
            Xs_pre_gens = []
            Ys_pre_gens = []
            for _, gen in df.groupby(by=['n_gen']):
                Xs = gen[self.args.x_axis].to_list()
                Ys = gen[self.args.y_axis].to_list()
                Xs_pre_gens.extend(Xs)
                Ys_pre_gens.extend(Ys)
                pf = get_pareto_front(Xs_pre_gens, Ys_pre_gens, maxX=False, maxY=False)
                pf = np.array(pf)
                hvs.append(ind(pf))
            print(hvs)
            print((np.array(hvs) - np.min(hvs)) /(np.max(hvs) - np.min(hvs)))
            # print(hvs)
    
    def plot_hypervolume(self):
        for keys, df in self.df_all.groupby(by=['house_no','win_size']):
            df[self.args.x_axis] = (df[self.args.x_axis] - df[self.args.x_axis].min()) / (df[self.args.x_axis].max() - df[self.args.x_axis].min())
            df[self.args.y_axis] = 1- (df[self.args.y_axis] - df[self.args.y_axis].min()) / (df[self.args.y_axis].max() - df[self.args.y_axis].min())
            print(keys)
            ref_point = np.array([1, 1])
            ind = Hypervolume(ref_point=ref_point)
            hvs = []
            Xs_pre_gens = []
            Ys_pre_gens = []
            for _, gen in df.groupby(by=['n_gen']):
                Xs = gen[self.args.x_axis].to_list()
                Ys = gen[self.args.y_axis].to_list()
                Xs_pre_gens.extend(Xs)
                Ys_pre_gens.extend(Ys)
                pf = get_pareto_front(Xs_pre_gens, Ys_pre_gens, maxX=False, maxY=False)
                pf = np.array(pf)
                hvs.append(ind(pf))
            fig, ax = plt.subplots()
            hvs = (np.array(hvs) - np.min(hvs)) /(np.max(hvs) - np.min(hvs))
            ax.plot(np.arange(len(hvs))+1, hvs, marker='D', color='green', linewidth=2)
            ax.set_ylabel('Normalized Hypervolume')
            ax.set_xlabel('Generations')
            # major_ticsk = np.arange(0,1.2, 0.2)
            # minor_ticsk = np.arange(0,1.05, 0.05)
            # ax.set_yticks(major_ticsk)
            # ax.set_yticks(minor_ticsk, minor=True)
            ax.grid(visible=True,which='major',  linestyle='-')
            self.plot_dir.mkdir(parents=True, exist_ok=True)
            # fig.suptitle('')
            
            fig.savefig(self.plot_dir.joinpath(f'hypervolume_{self.args.y_axis}_{keys[0]}_{keys[1]}.{self.args.format}'))
            plt.close()
        
    def plot_pareto_front_all(self):
        for keys, df in self.df_all.groupby(by=['house_no','win_size']):
            fig, ax = plt.subplots()
            Xs = df[self.args.x_axis].to_list()
            Ys = df[self.args.y_axis].to_list()
            ax = plot_pareto_frontier(Xs, Ys, maxX=False, maxY=True, ax=ax)
            ax.set_ylabel('Validation F1-Macro Score')
            ax.set_xlabel('FLOPs')
            fig.tight_layout()
            self.plot_dir.mkdir(parents=True, exist_ok=True)
            path = self.plot_dir.joinpath(f'pareto_front_all_{keys[0]}_{keys[1]}.{self.args.format}')
            fig.savefig(path)
            plt.close()
            
    def plot_pareto_front_by_generation(self):
        for keys, df in self.df_all.groupby(by=['house_no','win_size']):
            colorlist = iter(plt.get_cmap('tab10').colors)
            marker_cycle = iter(['o','v','D','s','^','p'])
            linestyle_cycle = iter([':','--','-','-'])
            fig, ax = plt.subplots(figsize=(5,5))
            pre_gen = 0
            hds = []
            maxX = False
            maxY = True
            labels = ['Initial (n_gen<=1)','Middle (1<n_gen<=10)','End (10<n_gen<=20)']
            # nadir point
            reference_point = df[[self.args.x_axis,self.args.y_axis]].sort_values([self.args.x_axis,self.args.y_axis], ascending=[maxX,maxY]).iloc[0,:].tolist()
            # worst point
            reference_point = [df[self.args.x_axis].sort_values(ascending=maxX).iloc[0],df[self.args.y_axis].sort_values(ascending=maxY).iloc[0]]
            print(reference_point)
            for label, gen in zip(labels,[1,10,200]):
                d = df.query(f'n_gen <= {gen} and n_gen > {pre_gen}')
                # d = df.query(f'n_gen <= {gen}')
                if len(d) == 0:
                    continue
                pre_gen = gen
                
                Xs = d[self.args.x_axis].to_list()
                Ys = d[self.args.y_axis].to_list()
                
                
                color = next(colorlist)
                
                pareto_front = get_pareto_front(Xs, Ys, maxX, maxY)
                hv_front = get_hypervolume_front(Xs, Ys, reference_point, maxX, maxY)

                pf_X = [pair[0] for pair in pareto_front]
                pf_Y = [pair[1] for pair in pareto_front]
                
                hvf_X = [pair[0] for pair in hv_front]
                hvf_Y = [pair[1] for pair in hv_front]
                
                marker = next(marker_cycle)
                
                # plot all searched points
                hd_1 = ax.scatter(Xs,Ys, s=2, color=color, label=label, marker=marker)
                # plot pareto front line
                hd_2 = ax.plot(hvf_X, hvf_Y, color=color, label=label,
                               linewidth=1, linestyle=next(linestyle_cycle))
                # plot pareto front points
                hd_3 = ax.scatter(pf_X, pf_Y, color=color, label=label, s=20, marker=marker,zorder=3)
                hds.append((hd_1,hd_2[0],hd_3))

            Xs = df[self.args.x_axis].to_list()
            Ys = df[self.args.y_axis].to_list()
            pareto_front = get_pareto_front(Xs, Ys, maxX, maxY)
            pf_X = [pair[0] for pair in pareto_front]
            pf_Y = [pair[1] for pair in pareto_front]
            hd_3 = ax.plot(pf_X, pf_Y, color=next(colorlist), label=label,
                               linewidth=2, linestyle=next(linestyle_cycle))
            hds.append(hd_3[0])
            labels.append('Final Pareto Front')
            ax.margins(0.2,0.1)            
            ax.grid(visible=True, which='major', axis='both')
            ax.set_xscale('log')    
            ax.set_ylabel('Validation F1-Macro Score')
            ax.set_xlabel('FLOPs')
            
            ax.legend(hds, labels, loc='lower right', frameon=True)
            self.plot_dir.mkdir(parents=True, exist_ok=True)
            path = self.plot_dir.joinpath(f'pareto_frontier_{self.args.y_axis}_{keys[0]}_{keys[1]}.{self.args.format}')
            fig.tight_layout()
            print(f'save path {path}')
            fig.savefig(path)
            plt.close()
        
    def plot_comparsion_on_pareto(self):
        for keys, df in self.df_all.groupby(by=['house_no','win_size','dataset']):
            colorlist = iter(plt.get_cmap('tab10').colors)
            marker_cycle = iter(['o','s','D','v','^','p'])
            
            fig, ax = plt.subplots(figsize=(3,3))
            for k , df_m in df.groupby('model'):
                df_m = df_m.sort_values('val_f1macro', ascending=False)
                if k == 'TSNet':
                    pf = df_m[[self.args.x_axis,self.args.y_axis]]
                else:
                    pf = df_m[[self.args.x_axis,self.args.y_axis]].iloc[0:1]
                ax.scatter(pf[self.args.x_axis].to_list(),pf[self.args.y_axis].to_list(), color=next(colorlist),marker=next(marker_cycle), label=Model_Name_Mapping[k], zorder=2)
            
            ax.set_xscale('log')
            ax.set_xlabel('FLOPs')
            ax.set_ylabel('Test F1-Macro')
            ax.grid(visible=True, which='major')
            ax.legend(frameon=True, loc='upper right')
            
            path = self.plot_dir.joinpath(f'comparison_pareto_{self.args.y_axis}_{keys[0]}_{keys[1]}_{keys[2]}.{self.args.format}')
            print("save fn", path)
            fig.savefig(path)
    
    def plot_all_cf(self):
        redd_mapping = {0:'refrigerator', 1:'microwave', 2:'dishwasher', 3:'washer_dryer'}
        ukdale_mapping = {0:'kettle', 1:'refrigerator', 2:'microwave', 3:'dishwasher', 4:'washer_dryer'}
        
        appliance_mapping = {'kettle':'Kettle','microwave':'MW', 'dishwasher': 'DW','washer_dryer': 'WM','refrigerator':'Fridge'}
        
        for keys, df in self.df_all.groupby(by=['house_no','win_size','dataset']):
            dc_confmx = {}
            model_cnf = None
            for model, df_m in df.groupby('model'):
                cnfmx = df_m.sort_values('test_f1macro',ascending=False).iloc[0]['test_confmx']
                cnfmx = json.loads(cnfmx)
                # print(cnfmx)
                model_cnf = {}
                for i, cf in enumerate(cnfmx):
                    if len(cnfmx) == 4:
                        model_cnf[appliance_mapping[redd_mapping[i]]] = cf
                    else:
                        model_cnf[appliance_mapping[ukdale_mapping[i]]] = cf
                dc_confmx[Model_Name_Mapping[model]] = model_cnf
            path = self.plot_dir.joinpath(f'confusion_matrix_{keys[0]}_{keys[1]}_{keys[2]}.{self.args.format}')
            # print(dc_confmx, len(dc_confmx), len(model_cnf))
            model_list = ['MLSVM','MLkNN','LSTMAE','CNNLSTM','BitcnNILM','NILM-NAS']
            plot_confusion_matrix_api(dc_confmx,model_list, len(dc_confmx), len(model_cnf), path, figsize=(len(dc_confmx), len(model_cnf)), show_absolute=True, show_normed=True)
    
    def plot_gpu_wall_time(self):
        # fig, ax = plt.subplots()
        house_winsize = []
        gpuwalltime = []
        for keys , df in self.df_all.groupby(['house_no','win_size']):
            k = f'house_{keys[0]}_ws_{keys[1]}'
            house_winsize.append(k)
            s = pd.to_timedelta(df['training_time']).sum()
            gpuwalltime.append(s)
        print(house_winsize)
        print(gpuwalltime)
        # ax.bar(house_winsize, gpuwalltime, label=house_winsize)
        # path = self.plot_dir.joinpath(f'gpu_wall_time.{self.args.format}')
        # print('save path', path)
        # fig.savefig(path)
    
def parse_config():
    args = sp.parse(Config)
    return args

if __name__ == '__main__':
    args = parse_config()
    gp = GenerationPlot(args)
    
    if args.task == 'plot':
        gp.plot()
    else:
        if args.exp_dir and Path(args.exp_dir).joinpath('nas_results.db').is_file():
            if args.task == 'walltime':
                gp.plot_gpu_wall_time()
            else:
                gp.plot_hypervolume()
                gp.plot_pareto_front_by_generation()
        elif args.exp_dir:
            gp.plot_pareto_front_all()
        elif args.exp_names:
            if args.task == 'comp_pf':
                gp.plot_comparsion_on_pareto()
            elif args.task == 'all_cnf':
                gp.plot_all_cf()