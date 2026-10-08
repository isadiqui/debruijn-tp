#!/bin/env python3
# -*- coding: utf-8 -*-
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#    A copy of the GNU General Public License is available at
#    http://www.gnu.org/licenses/gpl-3.0.html

"""Perform assembly based on debruijn graph."""

import argparse
import os
import random
from random import randint
import statistics
import sys
import textwrap
from pathlib import Path
from collections import Counter
from typing import Iterator, Dict, List

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from networkx import (
    DiGraph,
    all_simple_paths,
    lowest_common_ancestor,
    has_path,
)

random.seed(9001)
matplotlib.use("Agg")

__author__ = "Imane Sadiqui"
__copyright__ = "Universite Paris Cite"
__credits__ = ["Imane Sadiqui"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Imane Sadiqui"
__email__ = "imane.sadiqui@etu.u-paris.fr"
__status__ = "Developpement"


def isfile(path: str) -> Path:
    """Check if path is an existing file.

    :param path: (str) Path to the file

    :raises ArgumentTypeError: If file does not exist

    :return: (Path) Path object of the input file
    """
    myfile = Path(path)
    if not myfile.is_file():
        if myfile.is_dir():
            msg = f"{myfile.name} is a directory."
        else:
            msg = f"{myfile.name} does not exist."
        raise argparse.ArgumentTypeError(msg)
    return myfile


def get_arguments():
    """Retrieves the arguments of the program.

    :return: An object that contains the arguments
    """
    parser = argparse.ArgumentParser(
        description=__doc__, usage=f"{sys.argv[0]} -h"
    )
    parser.add_argument(
        "-i", dest="fastq_file", type=isfile, required=True, help="Fastq file"
    )
    parser.add_argument(
        "-k", dest="kmer_size", type=int, default=22, help="k-mer size (default 22)"
    )
    parser.add_argument(
        "-o",
        dest="output_file",
        type=Path,
        default=Path(os.curdir + os.sep + "contigs.fasta"),
        help="Output contigs in fasta file (default contigs.fasta)",
    )
    parser.add_argument(
        "-f", dest="graphimg_file", type=Path, help="Save graph as an image (png)"
    )
    return parser.parse_args()


def read_fastq(fastq_file: Path) -> Iterator[str]:
    """Extract reads from fastq files.

    :param fastq_file: (Path) Path to the fastq file.
    :return: A generator object that iterate the read sequences.
    """
    with open(fastq_file, "r", encoding="utf-8") as handle:
        for _ in handle:
            yield next(handle).strip()
            next(handle)
            next(handle)


def cut_kmer(read: str, kmer_size: int) -> Iterator[str]:
    """Cut read into kmers of size kmer_size.

    :param read: (str) Sequence of a read.
    :param kmer_size: (int) Size of the k-mers.
    :return: A generator object that provides the kmers (str) of size kmer_size.
    """
    for i in range(len(read) - kmer_size + 1):
        yield read[i : i + kmer_size]


def build_kmer_dict(fastq_file: Path, kmer_size: int) -> Dict[str, int]:
    """Build a dictionnary object of all kmer occurrences in the fastq file.

    :param fastq_file: (Path) Path to the fastq file.
    :param kmer_size: (int) Size of the k-mers.
    :return: A dictionnary object that identify all kmer occurrences.
    """
    kmer_counts = Counter()
    for seq in read_fastq(fastq_file):
        kmer_counts.update(cut_kmer(seq, kmer_size))
    return dict(kmer_counts)


def build_graph(kmer_dict: Dict[str, int]) -> DiGraph:
    """Build the debruijn graph

    :param kmer_dict: A dictionnary object that identify all kmer occurrences.
    :return: A directed graph (nx) of all kmer substring and weight (occurrence).
    """
    graph = nx.DiGraph()
    for kmer, count in kmer_dict.items():
        graph.add_edge(kmer[:-1], kmer[1:], weight=count)
    return graph


def remove_paths(
    graph: DiGraph,
    path_list: List[List[str]],
    delete_entry_node: bool,
    delete_sink_node: bool,
) -> DiGraph:
    """Remove a list of path in a graph. A path is set of connected node in
    the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param path_list: (list) A list of path
    :param delete_entry_node: (boolean) True->We remove the first node of a path
    :param delete_sink_node: (boolean) True->We remove the last node of a path
    :return: (nx.DiGraph) A directed graph object
    """
    for path in path_list:
        nodes_to_remove = list(path)
        if not delete_sink_node:
            nodes_to_remove = nodes_to_remove[:-1]
        if not delete_entry_node:
            nodes_to_remove = nodes_to_remove[1:]
        graph.remove_nodes_from(nodes_to_remove)
    return graph


def select_best_path(
    graph: DiGraph,
    path_list: List[List[str]],
    path_length: List[int],
    weight_avg_list: List[float],
    delete_entry_node: bool = False,
    delete_sink_node: bool = False,
) -> DiGraph:
    """Select the best path between different paths

    :param graph: (nx.DiGraph) A directed graph object
    :param path_list: (list) A list of path
    :param path_length: (list) A list of length of each path
    :param weight_avg_list: (list) A list of average weight of each path
    :param delete_entry_node: (boolean) True->We remove the first node of a path
    :param delete_sink_node: (boolean) True->We remove the last node of a path
    :return: (nx.DiGraph) A directed graph object
    """
    if len(path_list) <= 1:
        return graph

    std_weight = statistics.stdev(weight_avg_list) if len(weight_avg_list) > 1 else 0
    if std_weight > 0:
        best_index = weight_avg_list.index(max(weight_avg_list))
    else:
        std_length = statistics.stdev(path_length) if len(path_length) > 1 else 0
        if std_length > 0:
            best_index = path_length.index(max(path_length))
        else:
            best_index = randint(0, len(path_list) - 1)

    bad_paths = [p for i, p in enumerate(path_list) if i != best_index]
    return remove_paths(graph, bad_paths, delete_entry_node, delete_sink_node)


def path_average_weight(graph: DiGraph, path: List[str]) -> float:
    """Compute the weight of a path

    :param graph: (nx.DiGraph) A directed graph object
    :param path: (list) A path consist of a list of nodes
    :return: (float) The average weight of a path
    """
    return statistics.mean(
        [d["weight"] for (u, v, d) in graph.subgraph(path).edges(data=True)]
    )


def solve_bubble(graph: DiGraph, ancestor_node: str, descendant_node: str) -> DiGraph:
    """Explore and solve bubble issue

    :param graph: (nx.DiGraph) A directed graph object
    :param ancestor_node: (str) An upstream node in the graph
    :param descendant_node: (str) A downstream node in the graph
    :return: (nx.DiGraph) A directed graph object
    """
    paths = list(all_simple_paths(graph, ancestor_node, descendant_node))
    if len(paths) <= 1:
        return graph

    lengths = [len(p) for p in paths]
    weights = [path_average_weight(graph, p) for p in paths]

    return select_best_path(
        graph,
        paths,
        lengths,
        weights,
        delete_entry_node=False,
        delete_sink_node=False,
    )


def simplify_bubbles(graph: DiGraph) -> DiGraph:
    """Detect and explode bubbles

    :param graph: (nx.DiGraph) A directed graph object
    :return: (nx.DiGraph) A directed graph object
    """
    for node in list(graph.nodes()):
        predecessors = list(graph.predecessors(node))
        if len(predecessors) > 1:
            for idx, pred_i in enumerate(predecessors):
                for pred_j in predecessors[idx + 1:]:
                    ancestor = lowest_common_ancestor(graph, pred_i, pred_j)
                    if ancestor is not None:
                        graph = solve_bubble(graph, ancestor, node)
                        return simplify_bubbles(graph)
    return graph


def solve_entry_tips(graph: DiGraph, starting_nodes: List[str]) -> DiGraph:
    """Remove entry tips

    :param graph: (nx.DiGraph) A directed graph object
    :param starting_nodes: (list) A list of starting nodes
    :return: (nx.DiGraph) A directed graph object
    """
    for node in list(graph.nodes()):
        starts_reaching_node = [
            start for start in starting_nodes
            if start != node and has_path(graph, start, node)
        ]
        if len(starts_reaching_node) > 1:
            incoming_paths = []
            for start in starts_reaching_node:
                for path in all_simple_paths(graph, start, node):
                    incoming_paths.append(path)

            if len(incoming_paths) > 1 and len({p[0] for p in incoming_paths}) > 1:
                lengths = [len(p) for p in incoming_paths]
                weights = [path_average_weight(graph, p) for p in incoming_paths]
                graph = select_best_path(
                    graph,
                    incoming_paths,
                    lengths,
                    weights,
                    delete_entry_node=True,
                    delete_sink_node=False,
                )
                return solve_entry_tips(graph, get_starting_nodes(graph))
    return graph


def solve_out_tips(graph: DiGraph, ending_nodes: List[str]) -> DiGraph:
    """Remove out tips

    :param graph: (nx.DiGraph) A directed graph object
    :param ending_nodes: (list) A list of ending nodes
    :return: (nx.DiGraph) A directed graph object
    """
    for node in list(graph.nodes()):
        if graph.out_degree(node) > 1:
            reachable_ends = [
                end for end in ending_nodes
                if end != node and has_path(graph, node, end)
            ]
            if len(reachable_ends) > 1:
                succs_to_ends = {}
                for succ in graph.successors(node):
                    for end in reachable_ends:
                        if succ == end or has_path(graph, succ, end):
                            succs_to_ends[succ] = end

                if len(set(succs_to_ends.values())) > 1:
                    outgoing_paths = []
                    for succ, end in succs_to_ends.items():
                        if succ == end:
                            outgoing_paths.append([node, end])
                        else:
                            for p in all_simple_paths(graph, succ, end):
                                outgoing_paths.append([node] + p)

                    if len(outgoing_paths) > 1:
                        lengths = [len(p) for p in outgoing_paths]
                        weights = [path_average_weight(graph, p) for p in outgoing_paths]
                        graph = select_best_path(
                            graph,
                            outgoing_paths,
                            lengths,
                            weights,
                            delete_entry_node=False,
                            delete_sink_node=True,
                        )
                        return solve_out_tips(graph, get_sink_nodes(graph))
    return graph


def get_starting_nodes(graph: DiGraph) -> List[str]:
    """Get nodes without predecessors

    :param graph: (nx.DiGraph) A directed graph object
    :return: (list) A list of all nodes without predecessors
    """
    return [node for node in graph.nodes() if graph.in_degree(node) == 0]


def get_sink_nodes(graph: DiGraph) -> List[str]:
    """Get nodes without successors

    :param graph: (nx.DiGraph) A directed graph object
    :return: (list) A list of all nodes without successors
    """
    return [node for node in graph.nodes() if graph.out_degree(node) == 0]


def get_contigs(
    graph: DiGraph, starting_nodes: List[str], ending_nodes: List[str]
) -> List:
    """Extract the contigs from the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param starting_nodes: (list) A list of nodes without predecessors
    :param ending_nodes: (list) A list of nodes without successors
    :return: (list) List of [contiguous sequence and their length]
    """
    contigs = []
    for start in starting_nodes:
        for end in ending_nodes:
            if has_path(graph, start, end):
                for path in all_simple_paths(graph, start, end):
                    contig = path[0] + "".join(node[-1] for node in path[1:])
                    contigs.append((contig, len(contig)))
    return contigs


def save_contigs(contigs_list: List[tuple], output_file: Path) -> None:
    """Write all contigs in fasta format

    :param contig_list: (list) List of [contiguous sequence and their length]
    :param output_file: (Path) Path to the output file
    """
    with open(output_file, "w", encoding="utf-8") as out:
        for idx, (contig, length) in enumerate(contigs_list):
            out.write(f">contig_{idx} len={length}\n")
            out.write(textwrap.fill(contig, width=80) + "\n")


def draw_graph(graph: DiGraph, graphimg_file: Path) -> None:
    """Draw the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param graphimg_file: (Path) Path to the output file
    """
    plt.subplots()
    elarge = [(u, v) for (u, v, d) in graph.edges(data=True) if d["weight"] > 3]
    esmall = [(u, v) for (u, v, d) in graph.edges(data=True) if d["weight"] <= 3]
    pos = nx.random_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_size=6)
    nx.draw_networkx_edges(graph, pos, edgelist=elarge, width=6)
    nx.draw_networkx_edges(
        graph, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed"
    )
    plt.savefig(graphimg_file.resolve())


def main() -> None:
    """
    Main program function
    """
    args = get_arguments()

    kmer_dict = build_kmer_dict(args.fastq_file, args.kmer_size)
    graph = build_graph(kmer_dict)

    graph = simplify_bubbles(graph)

    starting_nodes = get_starting_nodes(graph)
    graph = solve_entry_tips(graph, starting_nodes)

    sink_nodes = get_sink_nodes(graph)
    graph = solve_out_tips(graph, sink_nodes)

    starting_nodes = get_starting_nodes(graph)
    sink_nodes = get_sink_nodes(graph)
    contigs = get_contigs(graph, starting_nodes, sink_nodes)
    save_contigs(contigs, args.output_file)

    if args.graphimg_file:
        draw_graph(graph, args.graphimg_file)


if __name__ == "__main__":
    main()
    