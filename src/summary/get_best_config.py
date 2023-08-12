import sys
sys.path.append('.')
import dataclasses
from simple_parsing import parse
from pathlib import Path
from src.summary.utils import read_all_results, get_pareto_front_df, get_pareto_front
from typing import Dict
from collections import defaultdict
from src.config_options import MyProgramArgs
from src.database.SQLiteMapper import Results
import json
import pandas as pd

@dataclasses.dataclass
class Config:
    exp_names: list[str]
    use_winsize: bool = True
    pf: bool = False
    output: str = None
    
def parse_args() -> Config:
    args = parse(Config)
    return args

def get_df(args):
    df_all = []
    for exp_path in args.exp_names:
        db_dir = Path(exp_path)
        results = read_all_results(db_name='nas_results.db', db_dir=db_dir)
        dc: Dict[str, list] = defaultdict(list)
        for result in results:
            p: MyProgramArgs = result['params']
            r: Results = result['result']
            data_params = json.loads(r.data_params)
            nas_params = json.loads(r.nas_params)
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
            dc['out_channels'].append(json.dumps(p.modelConfig.out_channels))
            dc['bit_string'].append(p.modelConfig.bit_string)
            dc['house_no'].append(data_params['house_no'])
            dc['win_size'].append(data_params['win_size'])
            dc['stride'].append(data_params['stride'])
            dc['training_time'].append(r.training_time)
            dc['n_gen'].append(nas_params.get('n_gen'))
            dc['lr'].append(p.modelBaseConfig.lr)
            dc['batch_size'].append(p.modelBaseConfig.batch_size)
            dc['lstm_hidden_features'].append(p.modelConfig.__getattribute__('lstm_hidden_features'))
            dc['lstm_out_features'].append(p.modelConfig.__getattribute__('lstm_out_features'))
        df = pd.DataFrame.from_dict(dc)
        df_all.append(df)
    df_all = pd.concat(df_all)
    return df_all

def main():
    args = parse_args()
    df = get_df(args)
    all_models = []
    if args.use_winsize:
        for k, d in df.groupby(['house_no','win_size']):
            d = d.sort_values(['val_f1macro'], ascending=False)
            model_dc = {}
            if args.pf:
                pf = get_pareto_front_df(d, ['val_f1macro','flops'], ['max','min'])
                for i, iter_row  in enumerate(pf.iterrows()):
                    index, row = iter_row
                    model_dc = {}
                    model_dc['dataset'] = str(row['dataset'])
                    model_dc['house_no'] = int(row['house_no'])
                    model_dc['win_size'] = int(row['win_size'])
                    model_dc['bit_string'] = str(row['bit_string'])
                    model_dc['out_channels'] = json.loads(row['out_channels'])
                    model_dc['lstm_hidden_features'] = int(row['lstm_hidden_features'])
                    model_dc['lstm_out_features'] = int(row['lstm_out_features'])
                    all_models.append(model_dc)
            else:
                house = d.iloc[0,:]
                model_dc['dataset'] = str(house['dataset'])
                model_dc['house_no'] = str(house['house_no'])
                model_dc['win_size'] = int(house['win_size'])
                model_dc['bit_string'] = str(house['bit_string'])
                model_dc['out_channels'] = json.loads(house['out_channels'])
                model_dc['lstm_hidden_features'] = int(house['lstm_hidden_features'])
                model_dc['lstm_out_features'] = int(house['lstm_out_features'])
                all_models.append(model_dc)
    else:
        for k, d in df.groupby(['house_no']):
            d_avg = d.groupby(['bit_string','dataset','out_channels']).mean(numeric_only=True)
            d_avg = d_avg.sort_values(['val_f1macro'], ascending=False) 
            print(d_avg)
            model_dc = {}
            name = d_avg.iloc[0,:].name
            house = d_avg.iloc[0,:]
            model_dc['house_no'] = str(k[0])
            model_dc['bit_string'] = str(name[0])
            model_dc['dataset'] = str(name[1])
            model_dc['out_channels'] = json.loads(name[2])
            all_models.append(model_dc)
            
    if args.output is None:
        out_path = Path('./best_config.json')
    else:
        out_path = Path(args.output)
    
    out_path.parents[0].mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(all_models, f, indent=4)
        print(f'save file {out_path}')
        
        
main()