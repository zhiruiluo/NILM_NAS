from __future__ import annotations
OPS_Encoding = {
    '000': 'skip_connect',
    # '': 'max_pool_1x1',
    '001': 'conv_1x1',
    '010': 'max_pool_3x3',
    '011': 'avg_pool_3x3',
    '100': 'sep_conv_3x3',
    '101': 'sep_conv_5x5',
    '110': 'dil_conv_3x3',
    '111': 'dil_conv_5x5',
}


TSNet = [
    [[1], [0, 0], [0, 1, 0], [0, 1, 1, 1], [1, 0, 0, 1, 1], [0]],
    [[0], [0, 0], [0, 1, 0], [0, 1, 0, 1], [1, 1, 1, 1, 1], [0]],
    [[0], [0, 1], [1, 0, 1], [1, 0, 1, 1], [1, 0, 0, 1, 1], [0]],
]
