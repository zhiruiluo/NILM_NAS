from __future__ import annotations

import logging

import torch

from .utilts import get_num_gpus

try:
    from deepspeed.profiling.flops_profiler import get_model_profile
except:
    pass

try:
    from fvcore.nn import FlopCountAnalysis
except:
    pass

try:
    from ptflops.flops_counter import get_model_complexity_info
except:
    pass

from .config_options.option_def import MyProgramArgs

logger = logging.getLogger(__name__)


class FlopsProfiler:
    def __init__(self, args: MyProgramArgs) -> None:
        self.print_flag = False
        self.batch_size = 1

    def _deepspeed_(self, model, input_shape=None, args=[], kwargs={}):
        if get_num_gpus() >= 1:
            with torch.cuda.device(0):
                flops, macs, params = get_model_profile(
                    model=model,
                    input_shape=input_shape,
                    args=args,
                    kwargs=kwargs,
                    print_profile=self.print_flag,
                    detailed=True,
                    module_depth=-1,
                    top_modules=1,
                    warm_up=10,
                    as_string=False,
                    output_file=None,
                    ignore_modules=None,
                )
            return flops, macs, params

    def _ptflops_(self, model, shape):
        macs, params = get_model_complexity_info(
            model,
            shape,
            print_per_layer_stat=self.print_flag,
            as_strings=False,
        )
        return macs, params

    def _fvcore_(self, model, shape):
        flops_counter = FlopCountAnalysis(model, torch.rand(self.batch_size, *shape))
        flops = flops_counter.total()
        return flops

    def get_flops(self, model, input_shape=None, args=[], kwargs={}) -> int:
        if get_num_gpus() >= 1:
            flops, macs, params = self._deepspeed_(model, input_shape, args, kwargs)
            logger.info(f"flops: {flops} macs: {macs} params: {params}")
        else:
            raise ValueError("Error")
            flops = self._fvcore_(model, input_shape)
            macs, params = self._ptflops_(model, input_shape)
            logger.info(f"flops: {flops} macs: {macs} params: {params}")
        return flops
