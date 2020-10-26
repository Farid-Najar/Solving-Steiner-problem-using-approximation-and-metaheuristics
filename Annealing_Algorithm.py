import itertools as it
import networkx as nx
from steinlib.parser import SteinlibParser
from threading import Thread
import Approximation
import random as rd
import copy as cp

def recuit(graph, terms : list) :
    """
    This function finds a solution for the steiner problem
        using annealing algorithm
    :param graph: the graph for each we search a solution
    :param terms: the list of terminal nodes
    :return: the solution found which is a set of edges
    """
    return 0

def init(graph):
    """
    This gives the first proposition of solution for the algorithm.
    :param graph: the graph for each we search a solution
    :param terms: the list of terminal nodes
    :return:
    """
    #sol = TP1.approx_steiner(graph, terms)
    return [round(rd.random()) == 1 for _ in range(len(graph.edges))]


def eval_annealing(sol, graph, terms : list, malus = 500):
    """
    This evaluates the solution of the algorithm.
    :param sol: the solution which is list of booleans
    :param graph: the graph for each we search a solution
    :param terms: the list of terminal nodes
    :param malus: the coefficient that we use to penalize bad solutions
    :return: the evaluation of the solution which is an integer
    """
    graph_sol = nx.Graph()
    nb_absent_terms = len(terms)
    edges = [e for e in graph.edges]
    assert len(edges) == len(sol)
    seen = set()
    for i in range(len(sol)):
        if sol[i] :
            (k, j) = edges[i]
            if k in terms and k not in seen:
                seen.add(k)
                nb_absent_terms -= 1
            if j in terms and j not in seen:
                seen.add(j)
                nb_absent_terms -= 1
            graph_sol.add_edge(k,j,weight=graph[k][j]['weight'])
    weights = graph_sol.size(weight='weight')
    assert nb_absent_terms >= 0
    #print(f'absent elements = {nb_absent_terms}')
    #print(f'connexe compo = {nx.number_connected_components(graph_sol)-1}')
    return weights + 2*malus*nb_absent_terms + malus*(nx.number_connected_components(graph_sol)-1)

