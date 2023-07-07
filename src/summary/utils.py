from src.database.Persistence import Persistence, PersistenceFactory
from src.config_options.options import loads_json
from typing import List, Dict, Optional
from pathlib import Path
import pandas as pd


def read_all_results(db_name: str, db_dir: str) -> Optional[List[Dict]]:
    if not Path(db_dir).joinpath(db_name).is_file():
        return None
    persistence = PersistenceFactory(db_name=db_name,
                       db_dir=db_dir,create_db=False).get_persistence()
    results = persistence.get_all_results()
    
    l_results = []
    for result in results:
        ret = {}
        ret['params'] = loads_json(result.params)
        ret['result'] = result
        l_results.append(ret)
    return l_results

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

def get_pareto_front_df(df: pd.DataFrame, objectives: list, modes: list):
    assert len(objectives) == 2
    ascendings = [modes[0] != 'max', modes[1] != 'max']
    df_sorted = df.sort_values(objectives, ascending=ascendings)
    pareto_front = []
    pareto_front.append(df_sorted.iloc[0])
    for index, row in df_sorted.iterrows():
        is_dominated = False
        for p in pareto_front:
            if all(((row[objective] >= p[objective]) ^ (not ascending)) for objective, ascending in zip(objectives,ascendings)):
                is_dominated = True
                break
        if not is_dominated:
            pareto_front.append(row)
            
    return pd.DataFrame(pareto_front[1:])

    """
    Calculate the Pareto front from a DataFrame with objective modes using pandas.
    """
    # Create a copy of the DataFrame
    ascendings = [objective_modes[0] != 'max', objective_modes[1] != 'max']
    pareto_df = df.copy()
    pareto_df = pareto_df.sort_values(objectives, ascending=ascendings)
    # Iterate over each row in the DataFrame
    for i, row in df.iterrows():
        # Check if the current row is dominated by any other row
        for _, other_row in df.iterrows():
            if i != _:
                is_dominated = True
                # Check each column based on the objective mode
                for j, col in enumerate(objectives):
                    if objective_modes[j] == 'max' and row[col] < other_row[col]:
                        is_dominated = False
                        break
                    elif objective_modes[j] == 'min' and row[col] > other_row[col]:
                        is_dominated = False
                        break
                # Remove the dominated row from the Pareto front DataFrame
                if is_dominated:
                    pareto_df.drop(i, inplace=True)
                    break

    return pareto_df