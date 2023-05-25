from __future__ import annotations
from simple_parsing.utils import (
    DataclassT,
    PossiblyNestedDict,
    V,
    contains_dataclass_type_arg,
    is_dataclass_instance,
    is_dataclass_type,
    is_optional,
    unflatten_split,
)
from simple_parsing.annotation_utils.get_field_annotations import (
    get_field_type_from_annotations,
)
from .option_def import MyProgramArgs, ModelbaseGroups, ModelGroups, DatasetGroups
import simple_parsing
from typing import Dict, Optional
import json
from pathlib import Path
import time
import os
import logging
import dataclasses
import copy

import sys

sys.path.append('.')


logger = logging.getLogger(__name__)


def select_field(_dataclass, field_name) -> dataclasses.Field:
    field_dict = {}
    for f in dataclasses.fields(_dataclass):
        field_dict[f.name] = f

    return field_dict[field_name]


def unflatten_selection_dict(
    flattened: dict, keyword: str = '__key__', sep: str = '.', recursive: bool = False,
) -> dict:
    """
    This function convert a flattened dict into a nested dict
    and it inserts the `keyword` as the selection into the nested dict.

    >>> unflatten_selection_dict({'ab_or_cd': 'cd', 'ab_or_cd.c_or_d': 'd'})
    {'ab_or_cd': {'__key__': 'cd', 'c_or_d': 'd'}}

    >>> unflatten_selection_dict({'lv1': 'a', 'lv1.lv2': 'b', 'lv1.lv2.lv3': 'c'})
    {'lv1': {'__key__': 'a', 'lv2': 'b', 'lv2.lv3': 'c'}}

    >>> unflatten_selection_dict({'lv1': 'a', 'lv1.lv2': 'b', 'lv1.lv2.lv3': 'c'}, recursive=True)
    {'lv1': {'__key__': 'a', 'lv2': {'__key__': 'b', 'lv3': 'c'}}}

    >>> unflatten_selection_dict({'ab_or_cd.c_or_d': 'd'})
    {'ab_or_cd': {'c_or_d': 'd'}}

    >>> unflatten_selection_dict({"a": 1, "b": 2})
    {'a': 1, 'b': 2}
    """
    dc = {}

    unflatten_those_top_level_keys = set()
    for k, v in flattened.items():
        splited_keys = k.split(sep)
        if len(splited_keys) >= 2:
            unflatten_those_top_level_keys.add(splited_keys[0])

    for k, v in flattened.items():
        keys = k.split(sep)
        top_level_key = keys[0]
        rest_keys = keys[1:]
        if top_level_key in unflatten_those_top_level_keys:
            sub_dc = dc.get(top_level_key, {})
            if len(rest_keys) == 0:
                sub_dc[keyword] = v
            else:
                sub_dc['.'.join(rest_keys)] = v
            dc[top_level_key] = sub_dc
        else:
            dc[k] = v

    if recursive:
        for k in unflatten_those_top_level_keys:
            v = dc.pop(k)
            unflatten_v = unflatten_selection_dict(v, recursive=recursive)
            dc[k] = unflatten_v
    return dc


def replace_subgroups(obj: DataclassT, selections: dict | None = None) -> DataclassT:
    """
    This function replaces the dataclass of subgroups, union, and optional union.
    The `selections` dict can be in flat format or in nested format.

    The values of selections can be `Key` of subgroups, dataclass type, and dataclass instance.
    """
    keyword = '__key__'

    if not selections:
        return obj
    selections = unflatten_selection_dict(selections, keyword)

    replace_kwargs = {}
    for field in dataclasses.fields(obj):
        if not field.init:
            raise ValueError(
                f'Cannot replace value of non-init field {field.name}.')

        if field.name not in selections:
            continue

        field_value = getattr(obj, field.name)
        field_annotation = get_field_type_from_annotations(
            obj.__class__, field.name)

        new_value = None
        # Replace subgroup is allowed when the type annotation contains dataclass
        if not contains_dataclass_type_arg(field_annotation):
            raise ValueError(
                f'The replaced subgroups contains no dataclass in its annotation {field_annotation}',
            )

        selection = selections.pop(field.name)
        if isinstance(selection, dict):
            value_of_selection = selection.pop(keyword, None)
            child_selections = selection
        else:
            value_of_selection = selection
            child_selections = None

        if is_dataclass_type(value_of_selection):
            field_value = value_of_selection()
        elif is_dataclass_instance(value_of_selection):
            field_value = copy.deepcopy(value_of_selection)
        elif field.metadata.get('subgroups', None):
            assert isinstance(value_of_selection, str)
            subgroup_selection = field.metadata['subgroups'][value_of_selection]
            if is_dataclass_instance(subgroup_selection):
                # when the subgroup selection is a frozen dataclass instance
                field_value = subgroup_selection
            else:
                # when the subgroup selection is a dataclass type
                field_value = field.metadata['subgroups'][value_of_selection]()
        elif is_optional(field_annotation) and value_of_selection is None:
            field_value = None
        elif (
            contains_dataclass_type_arg(
                field_annotation) and value_of_selection is None
        ):
            field_value = field.default_factory()
        else:
            raise ValueError(
                f"invalid selection key '{value_of_selection}' for field '{field.name}'",
            )

        if child_selections:
            new_value = replace_subgroups(field_value, child_selections)
        else:
            new_value = field_value

        replace_kwargs[field.name] = new_value
    return dataclasses.replace(obj, **replace_kwargs)


def replace(
    obj: DataclassT, changes_dict: dict | None = None, **changes,
) -> DataclassT:
    if changes_dict and changes:
        raise ValueError('Cannot pass both `changes_dict` and `changes`')
    changes = changes_dict or changes
    # changes can be given in a 'flat' format in `changes_dict`, e.g. {"a.b.c": 123}.
    # Unflatten them back to a nested dict (e.g. {"a": {"b": {"c": 123}}})
    changes = unflatten_split(changes)

    replace_kwargs = {}
    for field in dataclasses.fields(obj):
        if field.name not in changes:
            continue
        if not field.init:
            # if drop_non_init:
            #     print(f'Drop non init {field.name} {changes[field.name]}')
            #     continue
            raise ValueError(
                f'Cannot replace value of non-init field {field.name}.')

        field_value = getattr(obj, field.name)

        if is_dataclass_instance(field_value) and isinstance(changes[field.name], dict):
            field_changes = changes.pop(field.name)
            new_value = replace(field_value, **field_changes)
        else:
            new_value = changes.pop(field.name)
        replace_kwargs[field.name] = new_value

    # note: there may be some leftover values in `changes` that are not fields of this dataclass.
    # we still pass those.
    replace_kwargs.update(changes)

    return dataclasses.replace(obj, **replace_kwargs)


def replace_consistant(args: MyProgramArgs, config: dict) -> dict:
    key_dataset = config.pop('__key__@datasetConfig', None)
    if key_dataset:
        args = replace_subgroups(args, {'datasetConfig': key_dataset})
        args = replace(args, {'expOption.dataset': key_dataset})
    args = replace(args, config)
    config['args'] = args
    return config


def loads_json(s) -> MyProgramArgs:
    params = json.loads(s)
    modelConfig = params.pop('modelConfig', None)
    modelBaseConfig = params.pop('modelBaseConfig', None)
    datasetConfig = params.pop('datasetConfig', None)
    # datasetBaseConfig = params.pop('datasetBaseConfig', None)
    conf = MyProgramArgs.from_dict(params)
    conf.modelConfig = ModelGroups(
    ).groups[conf.expOption.model].from_dict(modelConfig)
    conf.modelBaseConfig = (
        ModelbaseGroups()
        .get_modelbase_by_model(conf.modelConfig.__class__)
        .from_dict(modelBaseConfig)
    )
    conf.datasetConfig = (
        DatasetGroups().groups[conf.expOption.dataset].from_dict(datasetConfig)
    )
    return conf


class OptionManager:
    def __init__(self) -> None:
        self.subgroups = {}
        self.args: MyProgramArgs = None
        self.default_sys_argv = sys.argv
        self._on_pre_build()

    def _update_subgroups(self, subgroups):
        self.subgroups['datasetConfig'] = subgroups['args.datasetConfig']
        self.subgroups['modelConfig'] = subgroups['args.modelConfig']
        self.subgroups['modelBaseConfig'] = subgroups['args.modelBaseConfig']

    def _replace_subgroups(self, changes: dict, ori_subgroup):
        subgroups = {}
        subgroups['datasetConfig'] = (
            changes.get('datasetConfig')
            if changes.get('datasetConfig') is not None
            else ori_subgroup['datasetConfig']
        )
        subgroups['modelConfig'] = (
            changes.get('modelConfig')
            if changes.get('modelConfig') is not None
            else ori_subgroup['modelConfig']
        )
        subgroups['modelBaseConfig'] = (
            changes.get('modelBaseConfig')
            if changes.get('modelBaseConfig') is not None
            else ori_subgroup['modelBaseConfig']
        )
        changes.pop('datasetConfig', None)
        changes.pop('modelConfig', None)
        changes.pop('modelBaseConfig', None)
        return subgroups

    def _parse_config(self, argv):
        parser = simple_parsing.ArgumentParser()
        parser.add_argument('--config_path', type=str, default='')
        args, opt_seq = parser.parse_known_args(argv)
        if args.config_path != '':
            args = MyProgramArgs.load_yaml(args.config_path)
            print(f'[config_path] loading from file\n{args}')
            return args, opt_seq
        print('[config_path] not load from file')
        return None, opt_seq

    def _parse_default(self, argv, default):
        return simple_parsing.parse_known_args(
            MyProgramArgs, args=argv, default=default,
        )

    def _on_pre_build(self):
        default_args, opt_seq = self._parse_config(self.default_sys_argv)
        if default_args is not None:
            args = default_args
            self._update_subgroups(
                {
                    'args.datasetConfig': args.expOption.dataset,
                    'args.modelConfig': args.expOption.model,
                    'args.modelBaseConfig': 'lighting',
                },
            )
        else:
            args, opt_seq = self._parse_default(
                self.default_sys_argv, default_args)
            self._update_subgroups(
                {
                    'args.datasetConfig': args.expOption.dataset,
                    'args.modelConfig': args.expOption.model,
                    'args.modelBaseConfig': 'lighting',
                },
            )

        self.args = args

        self._make_consist_for_subgroups(self.args, self.subgroups)

        jobid = os.environ.get('SLURM_JOB_ID')
        if jobid is not None:
            args.systemOption.job_name = (
                f'{jobid}_{time.strftime("%m%d_%H%M", time.localtime())}'
            )
        else:
            args.systemOption.job_name = 'time_{}'.format(
                time.strftime('%m%d_%H%M', time.localtime()),
            )

        args.systemOption.update_dir()
        self.opt_seq = opt_seq
        logger.info('[parser configs] %s', opt_seq)

    def replace_params(self, changes: dict) -> MyProgramArgs:
        new_args = copy.deepcopy(self.args)
        subgroups = self._replace_subgroups(changes, self.subgroups)
        self._make_consist_for_subgroups(new_args, subgroups)
        new_args = self._replace_by_dict_known(new_args, changes)
        new_args.systemOption.update_dir()
        return new_args

    def _make_consist_for_subgroups(self, args, subgroups):
        args.expOption.dataset = subgroups['datasetConfig']
        args.expOption.model = subgroups['modelConfig']

        dataset_config_cls = select_field(MyProgramArgs, 'datasetConfig').metadata[
            'subgroups'
        ][subgroups['datasetConfig']]
        if not isinstance(args.datasetConfig, dataset_config_cls):
            args.datasetConfig = dataset_config_cls()

        model_config_cls = select_field(MyProgramArgs, 'modelConfig').metadata[
            'subgroups'
        ][subgroups['modelConfig']]
        if not isinstance(args.modelConfig, model_config_cls):
            args.modelConfig = model_config_cls()

        model_base_config_cls = ModelbaseGroups().get_modelbase_by_model(
            args.modelConfig.__class__,
        )
        if not isinstance(args.modelBaseConfig, model_base_config_cls):
            args.modelBaseConfig = model_base_config_cls()

    def _replace_by_dict_known(
        self, src: MyProgramArgs, changes: dict,
    ) -> MyProgramArgs:
        d = copy.deepcopy(vars(src))
        for k, v in changes.items():
            key = k.split('.')
            if d.get(key[0], None) is None:
                continue
            if len(key) == 1:
                d[key[0]] = v
            elif len(key) == 2:
                d1 = dataclasses.replace(d[key[0]], **{f'{key[1]}': v})
                d[key[0]] = d1
        new_d = {}
        for f in dataclasses.fields(MyProgramArgs):
            new_d[f.name] = d.get(f.name)
        return MyProgramArgs(**new_d)

    def save_yaml(self):
        self.args.save_yaml(
            Path(self.args.systemOption.task_dir).joinpath('conf.yaml'))
