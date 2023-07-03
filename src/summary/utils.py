from src.database.Persistence import Persistence, PersistenceFactory
from src.config_options.options import loads_json
from typing import List, Dict, Optional
from pathlib import Path


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