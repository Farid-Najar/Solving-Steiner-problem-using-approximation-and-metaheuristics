import sys
import itertools as it
import matplotlib.pyplot as plt
import networkx as nx
from steinlib.instance import SteinlibInstance
from steinlib.parser import SteinlibParser
import scipy
from threading import Thread
import matplotlib.pyplot as plt
import TP1
import random as rd


stein_file = "data/B/b1.stp"

def recuit(graph, terms, nb_iter=20):
    """
    This is the main.
    :param graph:
    :param terms:
    :param nb_iter:
    :return:
    """
    best = init(graph, terms)
    solutions = [best]
    eval_best = eval_genetic(best, graph, terms, 500)
    graph_edges = [e for e in graph.edges]
    coeff = 3
    for i in range(nb_iter):
        print(len(solutions))
        generation(solutions)
        for sol in solutions :
            if sol != best :
                eval_sol =  eval_genetic(sol, graph, terms, eval_best)
                if eval_sol > coeff*eval_best :
                    solutions.remove(sol)
                elif eval_sol < eval_best:
                    best = sol
        eval_best = eval_genetic(best, graph, terms, eval_best)
    solution = set()
    assert len(graph.edges) == len(best)
    for i in range(len(best)):
        if best[i] :
            solution.add(graph_edges[i])
    return solution


def init(graph, terms):
    """
    This gives the first proposition of solution for the algorithm.
    :param graph:
    :param terms:
    :return:
    """
    sol = TP1.approx_steiner(graph, terms)
    print(f'evaluation of the initial solution = {TP1.eval_sol(graph,terms,sol)}')
    return edges_to_bool(sol, [e for e in graph.edges])


def generation(solutions : list, proba = .1) :
    """
    Generate new generation of solutions.
    :param sol:
    :param graph:
    :param terms:
    :return:
    """
    #print(solutions)
    rd.seed(42)
    new_generation = []
    nb_changes = round(rd.random()*round(len(solutions)/2)) + 1
    for i in range(nb_changes) :
        s1 = rd.choice(solutions)
        s2 = rd.choice(solutions)
        if s1 != s2 :
            new_generation.append([s1[i] if i < len(s1)//2 else s2[i] for i in range(len(s1))])

    for i in range(len(new_generation)) :
        for j in range(len(new_generation[i])):
            if rd.random() < proba :
                new_generation[i][j] = not new_generation[i][j]
        if new_generation[i] not in solutions :
            solutions.append(new_generation[i])


def eval_genetic(sol, graph, terms, malus):
    """
    This evaluates the algorithm.
    :param sol:
    :param graph:
    :param terms:
    :param malus:
    :return:
    """
    graph_sol = nx.Graph()
    nb_absent_terms = len(terms)
    edges = [e for e in graph.edges]
    assert len(edges) == len(sol)
    for i in range(len(sol)):
        if sol[i] :
            (k, j) = edges[i]
            if k in terms :
                nb_absent_terms -= 1
            if j in terms :
                nb_absent_terms -= 1
            graph_sol.add_edge(k,j,weight=graph[k][j]['weight'])
    weights = graph_sol.size(weight='weight')
    print(f'absent elements = {nb_absent_terms}')
    print(f'connexe compo = {nx.number_connected_components(graph_sol)-1}')
    return weights + malus*nb_absent_terms + malus*(nx.number_connected_components(graph_sol)-1)

def edges_to_bool(edges : set, graph_edges):
    """
    It converts a set of edges to a list of booleans
        k-th element of the list tells if graph_edges[k] is in the set or not
    :param edges: set of edges
    :return: list of booleans
    """
    print(edges)
    print(edges.intersection(graph_edges))
    res = [False for _ in range(len(graph_edges))]
    for i in range(len(graph_edges)):
        res[i] = (graph_edges[i] in edges)
    return res


if __name__ == "__main__" :
    my_class = TP1.MySteinlibInstance()
    with open(stein_file) as my_file:
        my_parser = SteinlibParser(my_file, my_class)
        my_parser.parse()
        terms = my_class.terms
        graph = my_class.my_graph
        #TP1.print_graph(graph,terms)
        sol=recuit(graph,terms)
        TP1.print_graph(graph,terms,sol)
        print(TP1.eval_sol(graph,terms,sol))
