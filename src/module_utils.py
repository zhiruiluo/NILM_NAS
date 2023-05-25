from __future__ import annotations

import copy
import importlib
import inspect
import logging
from typing import Callable
from typing import Dict

logger = logging.getLogger(__name__)


def import_class(name):
    components = name.split('.')
    mod = importlib.import_module('.'.join(components[:-1]))
    mod = getattr(mod, components[-1])
    return mod


def import_function_or_class(module_name, method_name):
    module = importlib.import_module(f'{module_name}')
    method = getattr(module, method_name, None)
    if not method or not isinstance(method, type):
        module = importlib.import_module(f'{module_name}.{method_name}')
        method = getattr(module, method_name, None)
        if not method or not isinstance(method, type):
            raise ValueError(
                f"module {module_name}.{method_name} has no attribute '{method_name}'",
            )
    return method


def filter_dict(func, kwarg_dict, args):
    sign = inspect.signature(func).parameters.values()
    sign = {val.name for val in sign}
    if 'args' not in kwarg_dict:
        kwarg_dict.update(args)
    else:
        raise ValueError('args exists in kwarg_dict')
    common_args = sign.intersection(kwarg_dict.keys())
    filtered_dict = {key: kwarg_dict[key] for key in common_args}
    return filtered_dict


def init_class_from_namespace(class_, namespace):
    common_kwargs = filter_dict(
        class_, copy.deepcopy(vars(namespace)), {'args': namespace},
    )
    return class_(**common_kwargs)


def init_module(class_, args):
    sign = inspect.signature(class_).parameters.values()
    sign = {val.name for val in sign}
    kwarg_dict = {'args': args}
    common_args = sign.intersection(kwarg_dict.keys())
    filtered_dict = {key: kwarg_dict[key] for key in common_args}
    return class_(**filtered_dict)


class ModuleScannerBase:
    def __init__(self, module_dict_init: Callable[[], dict[str, str]]) -> None:
        self.module_dict_init = module_dict_init

    @property
    def module_dict(self):
        if not hasattr(self, '_module_dict'):
            if self.module_dict_init is None:
                raise ValueError('')
            else:
                self._module_dict = self.module_dict_init()
        return self._module_dict

    def default(self) -> str:
        return self.choices()[0]

    def choices(self):
        return sorted(self.module_dict.keys())

    def getClass(self, name):
        if name not in self.choices():
            raise ValueError(f'No module named {name}!')
        cls = import_class(self.module_dict[name])
        return cls

    def getObj(self, name, args):
        cls = self.getClass(name)
        return init_module(cls, args)
