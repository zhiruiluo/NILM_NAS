import sys
sys.path.append('.')
from src.database.Persistence import Persistence, PersistenceFactory
from src.database.SQLiteMapper import Results
from src.config_options import MyProgramArgs
from src.config_options.options import loads_json
from src.config_options.modelbase_configs import HyperParm
import itertools
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import patchworklib as pw
plt.style.use(["science","no-latex"])
from pathlib import Path
from typing import List, Dict
import json
import simple_parsing as sp
from loguru import logger
from typing import Literal
from collections import defaultdict
from dataclasses import dataclass, field
import pandas as pd

@dataclass
class PlotConfig():
    ...

@dataclass
class SearchScore(PlotConfig):
    score: str = 'val_acc'
    x_axis: str = 'flops'
    
@dataclass
class LrWd(PlotConfig):
    ...
    
@dataclass
class LrOrWd(PlotConfig):
    ...
    
class LrWdGaussian(PlotConfig):
    ...

@dataclass
class ParetoFrontier(PlotConfig):
    score: str = 'val_acc'
    x_axis: str = 'flops'


@dataclass
class ParetoCompare(PlotConfig):
    baselines: List[str]
    score: str = 'val_acc'
    x_axis: str = 'flops'


@dataclass
class ParetoDemo(PlotConfig):
    score: str = 'val_acc'
    x_axis: str = 'flops'



@dataclass
class Config():
    exp_name: List[str]
    # task: Literal['plot_lr_wd', 'plot_search_score'] = None
    task: PlotConfig = sp.subgroups({
            "search_score": SearchScore, 
            "lr_wd": LrWd,
            "lr_or_wd": LrOrWd,
            "lr_wd_gaussian": LrWdGaussian,
            "pareto_frontier":  ParetoFrontier,
            "pareto_compare": ParetoCompare,
            "pareto_demo": ParetoDemo,
        },
        default_factory=SearchScore
    )
    dir: str = 'logging'
    results_path: str = 'results'

def row2dict(row):
    d = {}
    for column in row.__table__.columns:
        d[column.name] = str(getattr(row, column.name))
    return d


def parse_argument() -> Config:
    args = sp.parse(Config)
    return args
    
def select_pareto_frontier(df: pd.DataFrame, maxX=True, maxY=True):
    df = df.reset_index(drop=True)
    Xs = df['flops'].to_list()
    Ys = df['val_acc'].to_list()
    tYs = df['test_acc'].to_list()
    mCs = df['modelConfig'].to_list()
    params = df['params'].to_list()
    sorted_list = sorted([[Xs[i], Ys[i], tYs[i], mCs[i], params[i]] for i in range(len(Xs))], reverse=maxX)
    pareto_front = [sorted_list[0]]
    for pair in sorted_list[1:]:
        if maxY:
            if pair[1] >= pareto_front[-1][1]:
                pareto_front.append(pair)
        else:
            if pair[1] <= pareto_front[-1][1]:
                pareto_front.append(pair)
    return pareto_front
    
class Summary:
    def __init__(self, args: Config):
        self.args = args
        if len(args.exp_name) == 0:
            print('exp_name is an empty list')
            return 
        
        self._exp_results = {}
        for n in args.exp_name:
            db_dir = Path(args.dir).joinpath(n)
            persistence = PersistenceFactory(
                db_name='nas_results.db', db_dir=str(db_dir),
                create_db=False).get_persistence()
            self._exp_results[n] = self.parse_params(persistence.get_all_results())
            print(f"!!! exp {n}")
            
        
        self._baseline_results = {}
            
        self.result_pth = Path(args.results_path)
        self.result_pth.mkdir(parents=True, exist_ok=True)
        if isinstance(args.task, LrWd):
            self.plot_lr_wd()
        elif isinstance(args.task, LrOrWd):
            self.plot_lr_or_wd()
        elif isinstance(args.task, LrWdGaussian):
            self.plot_gaussian_regression()
        elif isinstance(args.task, SearchScore):
            self.plot_search_score()
        elif isinstance(args.task, ParetoFrontier):
            self.plot_pareto_front()
        elif isinstance(args.task, ParetoCompare):
            self.plot_pareto_compare()
        elif isinstance(args.task, ParetoDemo):
            self.plot_pareto_front_demo()
    
    def get_results_df_from_exp(self):
        df_all = []
        for exp, results in self._exp_results.items():
            dc: Dict[str, list] = defaultdict(list)
            for result in results:
                p: MyProgramArgs = result['params']
                r: Results = result['result']
                # nas_params = json.loads(r.nas_params)
                data_params = json.loads(r.data_params)
                dc['exp'].append(exp)
                dc['modelConfig'].append(p.modelConfig.dumps_json())
                dc['model'].append(p.expOption.model)
                dc['val_acc'].append(r.val_acc)
                dc['val_f1macro'].append(r.val_f1macro)
                dc['question_no'].append(data_params['question_no'])
                # dc['val_acc_flops'].append(nas_params['val_acc_flops'])
                dc['test_acc'].append(r.test_acc)
                dc['test_f1macro'].append(r.test_f1macro)
                dc['flops'].append(r.flops)
                # dc['params'].append(nas_params['params'])
                dc['dataset'].append(r.dataset)
                # dc['alpha'].append(p.nasOption.alpha)
                dc['t'].append(r.start_time)
            
            df = pd.DataFrame.from_dict(dc)
            unique_columns = df.groupby(['dataset', 'question_no']).size().reset_index().drop(0, axis=1)
            df_keep = []
            for r in unique_columns.to_dict(orient='records'):
                df_ = df.query("dataset == '{}' and question_no == {}".format(r['dataset'], r['question_no']))
                # if len(df_) != 100:
                #     print(f'skip {r} {exp} {len(df_)}')
                #     continue
                df_keep.append(df_)
            df_all.extend(df_keep)
        
        df_all = pd.concat(df_all).reset_index(drop=True)
        return df_all
    
    def get_results_df_from_baseline(self, baseline_results):
        dc: Dict[str, list] = defaultdict(list)
        for bl, results in baseline_results.items():
            for result in results:
                p: MyProgramArgs = result['params']
                r: Results = result['result']
                data_params = json.loads(r.data_params)
                dc['exp'].append(bl)
                dc['model'].append(p.expOption.model)
                # dc['regions'].append(data_params['regions'])
                dc['question_no'].append(data_params['question_no'])
                dc['val_acc'].append(r.val_acc)
                dc['test_acc'].append(r.test_acc)
                dc['val_f1macro'].append(r.val_f1macro)
                dc['test_f1macro'].append(r.test_f1macro)
                dc['flops'].append(r.flops)
                dc['dataset'].append(r.dataset)
                dc['t'].append(r.start_time)
        df = pd.DataFrame.from_dict(dc)
        
        return df

    def plot_lr_wd(self):
        x, y, hue = [], [], []
        for epn, results in self._exp_results.items():
            for result in results:
                p: MyProgramArgs = result['params']
                baseConfig: HyperParm = p.modelBaseConfig
                x.append(baseConfig.lr)
                y.append(baseConfig.weight_decay)
                
                r: Results = result['result']
                hue.append(r.val_acc)
            plt.figure(figsize=(5,4))
            plt.scatter(x, y, c=hue, cmap='viridis')
            # plt.yscale('log', base=10)
            # plt.xscale('log', base=10)
            plt.xlabel('learning rate')
            plt.ylabel('weight_decay')
            plt.colorbar()
            fn = self.result_pth.joinpath(f'plot_lr_and_wd_{epn}.png')
            plt.savefig(fn)
    
    def plot_lr_or_wd(self):
        x, y, hue = [], [], []
        for epn, results in self._exp_results.items():
            for result in results:
                p: MyProgramArgs = result['params']
                baseConfig: HyperParm = p.modelBaseConfig
                x.append(baseConfig.lr)
                y.append(baseConfig.weight_decay)
                r: Results = result['result']
                hue.append(r.val_acc)
            for x_axis in ['lr', 'wd']:
                plt.figure(figsize=(4,4))
                if x_axis == 'lr':
                    plt.scatter(x, hue)
                else:
                    plt.scatter(y, hue)
                # plt.yscale('log', base=10)
                # plt.xscale('log', base=10)
                plt.xlabel(x_axis)
                plt.ylabel('val acc')
                # plt.colorbar()
                fn = self.result_pth.joinpath(f'plot_{x_axis}_{epn}.png')
                plt.savefig(fn)
                plt.close()
    
    def plot_gaussian_regression(self):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
        
        
        x_mesh = np.linspace(1e-4, 1e-1, num=1000)
        y_mesh = np.linspace(1e-4, 1e-1, num=1000)
        xv, yv = np.meshgrid(x_mesh,y_mesh)
        xy_mesh = np.stack((xv, yv), axis=2)
        xy_points = xy_mesh.reshape(-1, 2)

        x, y, hue = [], [], []
        for epn, results in self._exp_results.items():
            for result in results:
                p: MyProgramArgs = result['params']
                baseConfig: HyperParm = p.modelBaseConfig
                x.append(baseConfig.lr)
                y.append(baseConfig.weight_decay)
                
                r: Results = result['result']
                hue.append(r.val_acc)
            
            xy_ob = np.stack((x,y),axis=1)
            score_ob = np.asarray(hue)
            # kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
            gp = GaussianProcessRegressor()
            
            gp.fit(xy_ob, score_ob)
            
            xy_hue = gp.predict(xy_points)
            xy_mesh = xy_hue.reshape(1000, 1000)
            print(xv.max(), yv.min(), xy_mesh.shape, xy_mesh.max(), xy_mesh.min())
            plt.figure(figsize=(5,4))
            c = plt.pcolormesh(xv, yv, xy_mesh, cmap='RdBu')
            plt.colorbar(c)
            # plt.yscale('log', base=10)
            # plt.xscale('log', base=10)
            fn = self.result_pth.joinpath(f'plot_lrwd_gaussian_{epn}.png')
            plt.savefig(fn)
            
    
    def plot_search_score(self):
        df = self.get_results_df_from_exp()
        unique_columns = df.groupby(['dataset','question_no']).size().reset_index().drop(0, axis=1)
        for r in unique_columns.to_dict(orient='records'):
            my_df = df.query(f"dataset == '{r['dataset']}' and question_no == {r['question_no']}")
            print(my_df.iloc[my_df[self.args.task.score].argmax()])
            if len(my_df) == 0:
                continue
            my_df = my_df.sort_values(by=['t']).reset_index(drop=True)
            my_df['score_cummax'] = my_df[self.args.task.score].cummax()
            plt.figure(figsize=(4,4))
            plt.plot(my_df.index, my_df.score_cummax)
            plt.xlabel('Sampled models')
            plt.ylabel(f'{self.args.task.score}')
            plt.title(f'Cumulative {self.args.task.score} for {r["dataset"]} on Bayesian Optimization')
            # plt.xticks(list(range(0,100,20)))
            # plt.ylim(0.94,1)
            plt.grid(visible=True, which='major', axis='both')
            plt.tight_layout()
            fn = self.result_pth.joinpath(f'plot_search_score_{self.args.task.score}_{self.args.task.x_axis}_{r["dataset"]}_{r["question_no"]}.pdf')
            plt.savefig(fn)
            plt.close()
            print('save file',fn)

    def plot_pareto_front(self):
        df = self.get_results_df_from_exp()
        
        def plot_pareto_frontier(Xs, Ys, xlim, maxX=True, maxY=True):
            '''Pareto frontier selection process'''
            sorted_list = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxX)
            pareto_front = [sorted_list[0]]
            for pair in sorted_list[1:]:
                if maxY:
                    if pair[1] >= pareto_front[-1][1]:
                        pareto_front.append(pair)
                else:
                    if pair[1] <= pareto_front[-1][1]:
                        pareto_front.append(pair)
            
            '''Plotting process'''
            ax = pw.Brick(figsize=(2,2))
            ax.scatter(Xs,Ys, s=4)
            pf_X = [pair[0] for pair in pareto_front]
            pf_Y = [pair[1] for pair in pareto_front]
            ax.plot(pf_X, pf_Y, 'r')
            # ax.set_xlabel("Model flops in log scale")
            # ax.set_ylabel("Val Accuracy")
            # ax.set_ylim(0.7,1.01)
            # ax.set_xlim(*xlim)
            ax.margins(0.2,0.1)
            ax.grid(visible=True, which='major', axis='both')
            ax.set_xscale('log')
            
            return ax
        
        
        # flops_lim = (df['flops'].min(), df['flops'].max())
        # params_lim = (df['params'].min(), df['params'].max())
        flops_lim = (1e4, 5e8)
        params_lim = (1e2,1e5)
        
        unique_columns = df.groupby(['dataset','question_no']).size().reset_index().drop(0, axis=1)
        for r in unique_columns.to_dict(orient='records'):
            df_dr = df.query(f"dataset == '{r['dataset']}' and question_no == {r['question_no']}")
            count = 0
            row_ax = None
            print(len(df_dr))
            if len(df_dr) == 0:
                continue
            df_t = df_dr.sort_values(by=['t']).reset_index(drop=True)
            x = df_t[self.args.task.x_axis].to_list()
            y = df_t[self.args.task.score].to_list()
            if self.args.task.x_axis =='flops':
                xlim = flops_lim
            else: 
                xlim = params_lim
            ax = plot_pareto_frontier(x, y, xlim, maxX=False)
            # ax.set_title(f"$\\alpha$={alpha['alpha']}")
            ax.set_xlabel("$\it{num\_flops}$")
            ax.set_ylabel("Validation Accuracy")
            count += 1
            
            if row_ax is None:
                row_ax = ax
                
            if row_ax is not None:
                fn = self.result_pth.joinpath(f'plot_pareto_{self.args.task.score}_{self.args.task.x_axis}_{r["dataset"]}_{r["question_no"]}.pdf')
                row_ax.savefig(fn)
                print(f'save file {fn}')

    def plot_pareto_compare(self):
        df = self.get_results_df_from_exp()
        baseline_results = {}
        conf: ParetoCompare = self.args.task
        
        for n in conf.baselines:
            db_dir = Path(self.args.dir).joinpath(n)
            persistence = PersistenceFactory(
                db_name='nas_results.db', db_dir=str(db_dir),
                create_db=False).get_persistence()
            baseline_results[n] = self.parse_params(persistence.get_all_results())
            print(f"!!! baselines {n}")
        df_baselines = self.get_results_df_from_baseline(baseline_results)

        def plot_pareto_frontier(pareto_front, Lb, bXs, bYs, bLb, xlim, score):
            '''Plotting process'''
            ax = pw.Brick(figsize=(3,3))
            # ax.scatter(Xs,Ys, s=4)
            marker = itertools.cycle(('v','o','^','P','*')) 
            for bx, by, bl in zip(bXs, bYs, bLb):
                if bl == 'EEGNet':
                    ax.axvline(x=bx, linestyle='-.', color='grey', linewidth=1,zorder=1)
                    EEGNet_x = bx
                ax.scatter(bx, by, s=16, label=bl, marker=next(marker),zorder=2)
            pf_X = [pair[0] for pair in pareto_front]
            pf_Y = [pair[score] for pair in pareto_front]
            pf_MC = [pair[3] for pair in pareto_front]
            pf_test_acc = [pair[2] for pair in pareto_front]
            pf_params = [pair[4] for pair in pareto_front]
            small_ele = min(pf_X, key=lambda x:abs(x-EEGNet_x))
            small_idx = pf_X.index(small_ele)
            small_modelConfig = pf_MC[small_idx]
            small_test_acc = pf_test_acc[small_idx]
            small_params = pf_params[small_idx]
            BUDGET_S = (pf_X[small_idx], pf_Y[small_idx])
            print("BUDGET_S")
            print(BUDGET_S)
            print(small_modelConfig)
            print(small_test_acc)
            print(small_params)
            ax.plot(pf_X, pf_Y, 'black',zorder=1)
            ax.scatter(pf_X, pf_Y, s = 16, label=Lb, marker=next(marker),zorder=2)
            ax.scatter(pf_X[small_idx],pf_Y[small_idx], s = 16, label='FD-LiteNet', color='violet', marker="D",zorder=2)
            ax.set_xlabel("$\it{num\_flops}$")
            ax.set_ylabel("Validation Accuracy")
            ax.legend(prop={'size': 8})
            # ax.move_legend(new_loc='upper left', bbox_to_anchor=(1.05, 1.0))
            ax.set_ylim(0.6,1.01)
            ax.set_xlim(*xlim)
            ax.margins(0.2,0.1)
            ax.grid(visible=True, which='major', axis='both')
            ax.set_xscale('log')
            return ax

        flops_lim = (1e4, 1e7)
        params_lim = (1e2,1e5)
        unique_columns = df.groupby(['dataset','regions','alpha']).size().reset_index().drop(0, axis=1)
        for r in unique_columns.to_dict(orient='records'):
            my_df = df.query(f"dataset == '{r['dataset']}' and regions == '{r['regions']}' and alpha == {r['alpha']}")
            if len(my_df) == 0:
                continue
            for b in [100]:
                df_t = my_df.sort_values(by=['t']).reset_index(drop=True)
                df_ = df_t[:b]
                pareto_frontier = select_pareto_frontier(df_, maxX=False)
                # x = df_[self.args.task.x_axis].to_list()
                # y = df_[self.args.task.score].to_list()
                Lb = df_['model'].unique()[0]
                if Lb == 'BasicV2':
                    Lb = 'Pareto-optimal model'
                if self.args.task.x_axis =='flops':
                    xlim = flops_lim
                else: 
                    xlim = params_lim
                df_b = df_baselines.query("dataset == '{}' and regions == '{}'".format(r['dataset'], r['regions']))
                bXs = df_b[self.args.task.x_axis].to_list()
                bYs = df_b[self.args.task.score].to_list()
                bLb = df_b['model'].to_list()
                score = 1 if self.args.task.score == 'val_acc' else 2
                print(score)
                ax = plot_pareto_frontier(pareto_frontier, Lb, bXs, bYs, bLb, xlim, score)
                
            fn = self.result_pth.joinpath(f'plot_pareto_compare_{self.args.task.score}_{self.args.task.x_axis}_{r["dataset"]}_{r["regions"]}_{r["alpha"]}.pdf')
            ax.savefig(fn)
            print(f'save file {fn}')
            
            
    def parse_params(self, results: List[Results]) -> List:
        l_results = []
        for result in results:
            ret = {}
            ret['params'] = loads_json(result.params)
            # ret['params'] = MyProgramArgs.loads_json(result.params, drop_extra_fields=False)
            ret['result'] = result
            l_results.append(ret)
            
        return l_results
        
    def results_to_dict(self, obj):
        l = []
        for row in obj:
            dc = {}
            for k in row.keys():
                dc[k] = str(row[k])
            l.append(dc)
        
        return dc

    def to_json(self, fn: str):
        obj = self.results_to_dict(self._exp_results)
        with open(fn, mode='w') as fp:
            json.dump(obj, fp, indent=2)
        

    def plot_pareto_front_demo(self):
        df = self.get_results_df_from_exp()
        
        def plot_pareto_frontier(Xs, Ys, xlim, maxX=True, maxY=True):
            '''Pareto frontier selection process'''
            Xs.extend([1e5,0.8e5])
            Ys.extend([0.82,0.8])
            Ys = np.array(Ys) + np.log10(np.array(Xs)) * 0.04
            Ys = Ys.tolist()
            sorted_list = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxX)
            non_pareto_front = []
            pareto_front = [sorted_list[0]]
            for pair in sorted_list[1:]:
                if maxY:
                    if pair[1] >= pareto_front[-1][1]:
                        pareto_front.append(pair)
                    else:
                        non_pareto_front.append(pair)
                else:
                    if pair[1] <= pareto_front[-1][1]:
                        pareto_front.append(pair)
                    else:
                        non_pareto_front.append(pair)
            
            '''Plotting process'''
            ax = pw.Brick(figsize=(5,5))
            npf_X = [pair[0] for pair in non_pareto_front]
            npf_Y = [pair[1] for pair in non_pareto_front]
            ax.scatter(np.log10(npf_X),npf_Y, s=8,marker='o', label='Models')
            pf_X = [pair[0] for pair in pareto_front]
            pf_Y = [pair[1] for pair in pareto_front]
            ax.scatter(np.log10(pf_X), pf_Y, c='red', marker='*', label='Optimal Models', zorder=2)
            ax.plot(np.log10(pf_X), pf_Y, 'black', label='Pareto Frontier',zorder=1)
            # ax.set_xlabel("Model flops in log scale")
            # ax.set_ylabel("Val Accuracy")
            # ax.set_ylim(0.7,0.90)
            # ax.set_xlim(*xlim)
            ax.margins(0.2,0.2)
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.yaxis.set_tick_params(labelleft=False)
            ax.xaxis.set_tick_params(labelright=False)
            ax.yaxis.set_tick_params(labeltop=False)
            
            # ax.grid(visible=True, which='major', axis='both')
            # ax.set_xscale('log')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.legend()
            
            return ax
        
        
        # flops_lim = (df['flops'].min(), df['flops'].max())
        # params_lim = (df['params'].min(), df['params'].max())
        flops_lim = (1e4, 5e8)
        params_lim = (1e2,1e5)
        unique_columns = df.groupby(['dataset','regions']).size().reset_index().drop(0, axis=1)
        for r in unique_columns.to_dict(orient='records'):
            df_dr = df.query(f"dataset == '{r['dataset']}' and regions == '{r['regions']}'")
            alphas = df_dr.groupby(['alpha']).size().reset_index().drop(0, axis=1)
            alphas_ax = None
            row_ax = None
            count = 0
            for alpha in alphas.to_dict(orient='records'):
                print(alpha)
                my_df = df_dr.query("alpha == {}".format(alpha['alpha']))
                print(len(my_df))
                if len(my_df) == 0:
                    continue
                df_t = my_df.sort_values(by=['t']).reset_index(drop=True)
                x = df_t[self.args.task.x_axis].to_list()
                y = df_t[self.args.task.score].to_list()
                if self.args.task.x_axis =='flops':
                    xlim = flops_lim
                else: 
                    xlim = params_lim
                ax = plot_pareto_frontier(x, y, xlim, maxX=False)
                # ax.set_title(f"$\\alpha$={alpha['alpha']}")
                ax.set_xlabel("Computational Complexity")
                ax.set_ylabel("Performance")
                count += 1
                
                if row_ax is None:
                    row_ax = ax
                else:
                    row_ax = row_ax | ax
                if count % 2 == 0:
                    if alphas_ax is None:
                        alphas_ax = row_ax
                    else:
                        alphas_ax = alphas_ax / row_ax
                    row_ax = None
                if alphas_ax is None:
                    alphas_ax = row_ax
                
            if alphas_ax is not None:
                fn = self.result_pth.joinpath(f'demo_pareto_{self.args.task.score}_{self.args.task.x_axis}_{r["dataset"]}_{r["regions"]}.pdf')
                alphas_ax.savefig(fn)
                print(f'save file {fn}')


def main():
    args = parse_argument()
    s = Summary(args)
        

if __name__ == '__main__':
    main()
