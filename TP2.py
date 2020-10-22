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
import copy as cp


stein_file = "data/B/b1.stp"
#stein_file = "data/test.std"

def genetic(graph, terms, nb_iter=200, taille_max_population = 5):
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
    i = 0
    while(i < nb_iter ):#or TP1.eval_sol(graph, terms, bool_to_edges(best, graph_edges)) == -1):
        #print(len(solutions))
        generation(solutions)
        #print(f'len(solutions) = {len(solutions)}')
        for sol in solutions :
            if sol != best :
                eval_sol =  eval_genetic(sol, graph, terms, 500)
                #print(f'{i} = {eval_sol}')
                if eval_sol < eval_best:
                    best = cp.copy(sol)
                    eval_best = eval_genetic(best, graph, terms, 500)
        selection(graph, terms, solutions, taille_max_population)
        i+=1
    return bool_to_edges(best, graph_edges)

def selection(graph, terms, solutions, taille_max_population) :
    sorted_solutions = sorted(solutions, key=lambda sol : eval_genetic(sol, graph, terms, 500))
    return [sorted_solutions[i] for i in range(min(len(sorted_solutions),taille_max_population))]

def init(graph, terms):
    """
    This gives the first proposition of solution for the algorithm.
    :param graph:
    :param terms:
    :return:
    """
    #sol = TP1.approx_steiner(graph, terms)
    return [round(rd.random()) == 1 for _ in range(len(graph.edges))]#edges_to_bool(sol, [e for e in graph.edges])


def generation(solutions : list, proba = .1, nb_changes = 5) :
    """
    Generates new generation of solutions.
    :param solutions:
    :param proba:
    :return:
    """
    #print(solutions)
    new_generation = []
    for i in range(nb_changes) :
        s1 = rd.choice(solutions)
        s2 = rd.choice(solutions)
        new_generation.append([s1[i] if i < len(s1)//2 else s2[i] for i in range(len(s1))])
    #print(f'len(new_generation) = {len(new_generation)}')
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

def edges_to_bool(sol : set, graph_edges):
    """
    It converts a set of edges to a list of booleans
        k-th element of the list tells if graph_edges[k] is in the set or not
    :param edges: set of edges
    :return: list of booleans
    """
    print(sol)
    for s in sol :
        if s not in graph_edges :
            print(f'probleme pour {s}')
    print(sol.intersection(graph_edges))
    res = [False for _ in range(len(graph_edges))]
    for i in range(len(graph_edges)):
        res[i] = (graph_edges[i] in sol)
    return res

def bool_to_edges(sol, graph_edges):
    solution = set()
    assert len(graph.edges) == len(sol)
    for i in range(len(sol)):
        if sol[i] :
            solution.add(graph_edges[i])
    return solution

if __name__ == "__main__" :
    my_class = TP1.MySteinlibInstance()
    with open(stein_file) as my_file:
        my_parser = SteinlibParser(my_file, my_class)
        my_parser.parse()
        terms = my_class.terms
        graph = my_class.my_graph
        #TP1.print_graph(graph,terms)
        sol=genetic(graph,terms)
        TP1.print_graph(graph,terms,sol)
        print(TP1.eval_sol(graph,terms,sol))
