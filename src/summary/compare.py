import sys

sys.path.append('.')
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
from simple_parsing import parse

from src.config_options import MyProgramArgs
from src.database.SQLiteMapper import Results
from src.summary.utils import read_all_results


@dataclass
class Config:
    exp_dir: str
    
def parse_config():
    args = parse(Config)
    return args


class Compare():
    def __init__(self, args: Config) -> None:
        self.args = args
        self.df_all = self._load_all_results()
        
    def _load_all_results(self):
        df_all = []
        exp_dir = Path(self.args.exp_dir)
        for fn in os.listdir(self.args.exp_dir):
            results = read_all_results('nas_results.db', exp_dir.joinpath(fn).as_posix())
            if results is None:
                continue
            dc: Dict[str, list] = defaultdict(list)
            for result in results:
                p: MyProgramArgs = result['params']
                r: Results = result['result']
                data_params = json.loads(r.data_params)
                dc['exp'].append(fn)
                dc['val_acc'].append(r.val_acc)
                dc['val_f1macro'].append(r.val_f1macro)
                if hasattr(r,'val_loss'):
                    dc['val_loss'].append(r.val_loss)
                else:
                    dc['val_loss'].append(None)
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
                
            df = pd.DataFrame.from_dict(dc)
            df_all.append(df)
        
        df_all = pd.concat(df_all).reset_index(drop=True)

        return df_all
    
    def get_max_val_f1macro(self):
        df = self.df_all.groupby(by=['exp','house_no','win_size']).max()
        df = df.sort_values(by=['house_no','win_size','test_f1macro'], ascending=[True,True,False])
        path = Path('results').joinpath(f'{self.args.exp_dir.replace("/","_")}.csv')
        df.to_csv(path.as_posix())
        
        print(f'save csv {path}')
    
def main():
    args = parse_config()
    c = Compare(args)
    c.get_max_val_f1macro()
    
if __name__ == '__main__':
    main()