from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from collections import OrderedDict
from copy import copy
from typing import Callable
from typing import List
from typing import Optional
from typing import TypeAlias
from typing import TypeVar

import torch
import torch.nn as nn

from .operations import Identity


class Decoder(ABC):
    """
    Abstract genome decoder class.
    """

    @abstractmethod
    def __init__(self, list_genome):
        """
        :param list_genome: genome represented as a list.
        """
        self._genome = list_genome

    @abstractmethod
    def get_model(self):
        raise NotImplementedError()


def phase_active(gene):
    """
    Determine if a phase is active.
    :param gene: list, gene describing a phase.
    :return: bool, true if active.
    """
    # The residual bit is not relevant in if a phase is active, so we ignore it, i.e. gene[:-1].
    return sum([sum(t) for t in gene[:-1]]) != 0


class ChannelBasedDecoder(Decoder):
    """
    Channel based decoder that deals with encapsulating constructor logic.
    """

    def __init__(
        self, list_genome: list, channels: list, repeats: list | None = None,
    ):
        """
        Constructor.
        :param list_genome: list, genome describing the connections in a network.
        :param channels: list, list of tuples describing the channel size changes.
        :param repeats: None | list, list of integers describing how many times to repeat each phase.
        """
        super().__init__(list_genome)

        self._model = None

        # First, we remove all inactive phases.
        self._genome = self.get_effective_genome(list_genome)
        self._channels = channels[: len(self._genome)]

        # Use the provided repeats list, or a list of all ones (only repeat each phase once).
        if repeats is not None:
            # First select only the repeats that are active in the list_genome.
            active_repeats = []
            for idx, gene in enumerate(list_genome):
                if phase_active(gene):
                    active_repeats.append(repeats[idx])

            self.adjust_for_repeats(active_repeats)
        else:
            # Each phase only repeated once.
            self._repeats = [1 for _ in self._genome]

        # If we had no active nodes, our model is just the identity, and we stop constructing.
        if not self._genome:
            self._model = Identity()

        # print(list_genome)

    def adjust_for_repeats(self, repeats):
        """
        Adjust for repetition of phases.
        :param repeats:
        """
        self._repeats = repeats

        # Adjust channels and genome to agree with repeats.
        repeated_genome = []
        repeated_channels = []
        for i, repeat in enumerate(self._repeats):
            for j in range(repeat):
                if j == 0:
                    # This is the first instance of this repeat, we need to use the (in, out) channel convention.
                    repeated_channels.append(
                        (self._channels[i][0], self._channels[i][1]),
                    )
                else:
                    # This is not the first instance, use the (out, out) convention.
                    repeated_channels.append(
                        (self._channels[i][1], self._channels[i][1]),
                    )

                repeated_genome.append(self._genome[i])

        self._genome = repeated_genome
        self._channels = repeated_channels

    def build_layers(self, phases):
        """
        Build up the layers with transitions.
        :param phases: list of phases
        :return: list of layers (the model).
        """
        layers = []
        last_phase = phases.pop()
        for phase, repeat in zip(phases, self._repeats):
            for _ in range(repeat):
                layers.append(phase)
            layers.append(
                nn.MaxPool2d(kernel_size=2, stride=2),
            )  # TODO: Generalize this, or consider a new genome.

        layers.append(last_phase)
        return layers

    @staticmethod
    def get_effective_genome(genome):
        """
        Get only the parts of the genome that are active.
        :param genome: list, represents the genome
        :return: list
        """
        return [
            [pos_gene, op_gene]
            for pos_gene, op_gene in genome
            if phase_active(pos_gene)
        ]

    @abstractmethod
    def get_model(self):
        raise NotImplementedError()


def get_node_constructor(genotype) -> Callable[[int, int, bool], nn.Module]:
    from .genotypes import OPS_Encoding
    from .operations import OPS

    op_lambda = OPS[OPS_Encoding[genotype]]
    return op_lambda


class ConnAndOpsPhase(nn.Module):
    """
    Residual Genome phase.
    """

    def __init__(self, gene, in_channels, out_channels, idx, preact=False):
        """
        Constructor.
        :param gene: list, element of genome describing connections in this phase.
        :param in_channels: int, number of input channels.
        :param out_channels: int, number of output channels.
        :param idx: int, index in the network.
        :param preact: should we use the preactivation scheme?
        """
        super().__init__()

        pos_gene, node_gene = gene
        self.channel_flag = (
            in_channels != out_channels
        )  # Flag to tell us if we need to increase channel size.
        self.first_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1 if idx != 0 else 3,
            stride=1,
            padding=0 if idx != 0 else 1,
            bias=False,
        )
        self.dependency_graph = ConnAndOpsPhase.build_dependency_graph(
            pos_gene)

        nodes: list[nn.Module] = []
        for i in range(len(pos_gene)):
            if len(self.dependency_graph[i + 1]) > 0:
                node_constructor = get_node_constructor(
                    ''.join(str(x) for x in node_gene[i]),
                )
                node_module = node_constructor(
                    out_channels, stride=1, affine=True)
                nodes.append(node_module)
            else:
                nodes.append(None)  # Module list will ignore NoneType.

        self.nodes = nn.ModuleList(nodes)

        #
        # At this point, we know which nodes will be receiving input from where.
        # So, we build the 1x1 convolutions that will deal with the depth-wise concatenations.
        #
        conv1x1s = [Identity()] + [
            Identity() for _ in range(max(self.dependency_graph.keys()))
        ]
        for node_idx, dependencies in self.dependency_graph.items():
            if len(dependencies) > 1:
                conv1x1s[node_idx] = nn.Conv2d(
                    len(dependencies) * out_channels,
                    out_channels,
                    kernel_size=1,
                    bias=False,
                )

        self.processors = nn.ModuleList(conv1x1s)
        self.out = nn.Sequential(nn.BatchNorm2d(
            out_channels), nn.ReLU(inplace=True))

    @staticmethod
    def build_dependency_graph(gene):
        """
        Build a graph describing the connections of a phase.
        "Repairs" made are as follows:
            - If a node has no input, but gives output, connect it to the input node (index 0 in outputs).
            - If a node has input, but no output, connect it to the output node (value returned from forward method).
        :param gene: gene describing the phase connections.
        :return: dict
        """
        graph = {}
        residual = gene[-1][0] == 1

        # First pass, build the graph without repairs.
        graph[1] = []
        for i in range(len(gene) - 1):
            graph[i + 2] = [j +
                            1 for j in range(len(gene[i])) if gene[i][j] == 1]

        graph[len(gene) + 1] = [0] if residual else []

        # Determine which nodes, if any, have no inputs and/or outputs.
        no_inputs = []
        no_outputs = []
        for i in range(1, len(gene) + 1):
            if len(graph[i]) == 0:
                no_inputs.append(i)

            has_output = False
            for j in range(i + 1, len(gene) + 2):
                if i in graph[j]:
                    has_output = True
                    break

            if not has_output:
                no_outputs.append(i)

        for node in no_outputs:
            if node not in no_inputs:
                # No outputs, but has inputs. Connect to output node.
                graph[len(gene) + 1].append(node)

        for node in no_inputs:
            if node not in no_outputs:
                # No inputs, but has outputs. Connect to input node.
                graph[node].append(0)

        return graph

    def forward(self, x):
        if self.channel_flag:
            x = self.first_conv(x)

        outputs = [x]

        for i in range(1, len(self.nodes) + 1):
            if not self.dependency_graph[i]:  # Empty list, no outputs to give.
                outputs.append(None)

            else:
                outputs.append(
                    self.nodes[i - 1](self.process_dependencies(i, outputs)))

        return self.out(self.process_dependencies(len(self.nodes) + 1, outputs))

    def process_dependencies(self, node_idx, outputs):
        """
        Process dependencies with a depth-wise concatenation and
        :param node_idx: int,
        :param outputs: list, current outputs
        :return: Variable
        """
        return self.processors[node_idx](
            torch.cat([outputs[i]
                      for i in self.dependency_graph[node_idx]], dim=1),
        )


class ConnAndOpsDecoder(ChannelBasedDecoder):
    def __init__(self, list_genome: list, channels: list, repeats=None):
        super().__init__(list_genome, channels, repeats)

        if self._model is not None:
            return

        phases = []
        for idx, (gene, (in_channels, out_channels)) in enumerate(
            zip(self._genome, self._channels),
        ):
            phases.append(ConnAndOpsPhase(
                gene, in_channels, out_channels, idx))

        self._model = nn.Sequential(*self.build_layers(phases))

    def get_model(self):
        return self._model


def test_decoder():
    genome = [
        [[[0], [1, 0], [1]], [[0, 0, 0], [1, 1, 1], [0, 1, 0]]],
        [[[0], [1, 0], [1]], [[0, 0, 0], [1, 1, 1], [0, 1, 0]]],
    ]
    model = ConnAndOpsDecoder(
        genome, [(3, 128), (128, 128), (128, 128)]).get_model()
    print(model)
    print(model(torch.randn(1, 3, 32, 32)).shape)
