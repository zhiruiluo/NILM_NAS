from __future__ import annotations

import numpy as np


def connection_decode(conn_bit_string):
    n = int(np.sqrt(2 * len(conn_bit_string) - 7 / 4) - 1 / 2)
    genome = []
    for i in range(n):
        operator = []
        for j in range(i + 1):
            operator.append(conn_bit_string[int(i * (i + 1) / 2 + j)])
        genome.append(operator)
    genome.append([conn_bit_string[-1]])

    return genome


def ops_decode(ops_bit_string, operator_bit_len=3):
    n = len(ops_bit_string) // operator_bit_len
    genome = []
    for i in range(n):
        idx = operator_bit_len * i
        operator = ops_bit_string[idx: operator_bit_len + idx]
        genome.append(operator)
    return genome


def phase_decode(phase_bit_string):
    x = len(phase_bit_string) - 1
    n = int((np.sqrt(25 + 8 * x) - 5) / 2)
    con_d = connection_decode(phase_bit_string[: (n - 1) * n // 2 + 1])
    ops_d = ops_decode(phase_bit_string[(n - 1) * n // 2 + 1:])
    return [con_d, ops_d]


def reduction_decode(bit_string):
    """
    ops: [avg_pool_s2]
    """


def decode_genome(genome: list):
    genotype = []
    for gene in genome:
        genotype.append(phase_decode(gene))
    return genotype


def convert(bit_string, n_phases=3) -> list:
    assert bit_string.shape[0] % n_phases == 0
    phase_length = bit_string.shape[0] // n_phases
    genome = []
    for i in range(0, bit_string.shape[0], phase_length):
        genome.append(bit_string[i: i + phase_length].tolist())

    return genome


def test_ops_decode():
    bit_string = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1])
    print(ops_decode(bit_string))


def test_connection_decode():
    bit_string = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1])
    decoded = connection_decode(bit_string)
    print(decoded)


def test_phase_decode():
    bit_string = np.array([0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0])
    decode = phase_decode(bit_string)
    print(decode)


def test_decode():
    """
    block, reduction, block, reduction, block, reduction, block, classification
    block: node=4, in_chan, out_chan, attention,
    reduction: in_chan, out_chan, reduction_ratio,
    """
    block_1 = np.array([0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0])
    block_2 = np.array([0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0])
    # bit_string = np.random.randint(0,2,size=21)
    genome = np.concatenate((block_1, block_2))
    print(genome)
    decoded = decode_genome(convert(genome, n_phases=2))
    print(decoded)
