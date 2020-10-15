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

def recuit(graph, terms, nb_iter=100):
    """
    This is the main.
    :param graph:
    :param terms:
    :param nb_iter:
    :return:
    """
    best = init(graph, terms)
    solutions = []
    solutions.append(best)
    eval_best = eval_genetic(best, graph, terms, 500)
    coeff = 2
    for i in range(nb_iter):
        generation(solutions)
        for sol in solutions :
            eval_sol =  eval_genetic(sol, graph, terms, eval_best)
            if eval_sol > coeff*eval_best :
                solutions.remove(sol)
            elif eval_sol < eval_best:
                best = sol
        eval_best = eval_genetic(best, graph, terms, eval_best)
    solution = set()
    for i in range(len(best)):
        if best[i] :
            solution.add(graph.edges[i])
    return solution


def init(graph, terms):
    """
    This gives the first proposition of solution for the algorithm.
    :param graph:
    :param terms:
    :return:
    """
    sol = TP1.approx_steiner(graph, terms)
    edges = [e for e in graph.edges]
    return [(edges[i] in sol) for i in range(len(edges))]


def generation(solutions : list, proba = .01) :
    """
    Generate new generation of solutions.
    :param sol:
    :param graph:
    :param terms:
    :return:
    """
    print(solutions)
    rd.seed(42)
    new_generation = []
    nb_changes = round(rd.random()*round(len(solutions)/2))
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
    :return:
    """
    graph_sol = nx.Graph()
    nb_absent_terms = 0
    edges = [e for e in graph.edges]
    for i in range(len(sol)):
        if sol[i] :
            (k, j) = edges[i]
            graph_sol.add_edge(k,j,weight=graph[k][j]['weight'])

    for t in terms:
        if t not in graph_sol.nodes:
            nb_absent_terms +=1

    weights = graph_sol.size(weight='weight')
    return weights + malus*nb_absent_terms + malus*(nx.number_connected_components(graph_sol)-1)

if __name__ == "__main__" :
    my_class = TP1.MySteinlibInstance()
    with open(stein_file) as my_file:
        my_parser = SteinlibParser(my_file, my_class)
        my_parser.parse()
        terms = my_class.terms
        graph = my_class.my_graph
        TP1.print_graph(graph,terms)
        sol=recuit(graph,terms)
        TP1.print_graph(graph,terms,sol)
        print(TP1.eval_sol(graph,terms,sol))
