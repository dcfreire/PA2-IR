# Natural Language Toolkit: Utility functions
#
# Copyright (C) 2001-2022 NLTK Project
# Author: Steven Bird <stevenbird1@gmail.com>
#         Eric Kafe <kafe.eric@gmail.com> (acyclic closures)
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT
# Modified by Daniel Carneiro Freire (daniel.carneiro.freire@gmail.com)

def invert_graph(graph):
    """
    Inverts a directed graph.

    :param graph: the graph, represented as a dictionary of sets
    :type graph: dict(set)
    :return: the inverted graph
    :rtype: dict(set)
    """
    inverted = {}
    for key in graph:
        for value in graph[key]:
            inverted.setdefault(value, set()).add(key)
    return inverted

def transitive_closure(graph, reflexive=False):
    """
    Calculate the transitive closure of a directed graph,
    optionally the reflexive transitive closure.

    The algorithm is a slight modification of the "Marking Algorithm" of
    Ioannidis & Ramakrishnan (1998) "Efficient Transitive Closure Algorithms".

    :param graph: the initial graph, represented as a dictionary of sets
    :type graph: dict(set)
    :param reflexive: if set, also make the closure reflexive
    :type reflexive: bool
    :rtype: dict(set)
    """
    if reflexive:
        base_set = lambda k: {k}
    else:
        base_set = lambda k: set()
    # The graph U_i in the article:
    agenda_graph = {k: graph[k].copy() for k in graph}
    # The graph M_i in the article:
    closure_graph = {k: base_set(k) for k in graph}
    for i in graph:
        agenda = agenda_graph[i]
        closure = closure_graph[i]
        while agenda:
            j = agenda.pop()
            closure.add(j)
            closure |= closure_graph.setdefault(j, base_set(j))
            agenda |= agenda_graph.get(j, base_set(j))
            agenda -= closure
    return closure_graph