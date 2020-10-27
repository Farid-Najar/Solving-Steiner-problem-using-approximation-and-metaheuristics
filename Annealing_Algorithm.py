#import itertools as it
import networkx as nx
from threading import Thread
import Approximation
import random as rd
import copy as cp
from numpy import exp
import Genetic_Algorithm

def recuit(graph : nx.Graph, terms : list, T_init, T_limit, lamb = .99) -> set :
    """
    This function finds a solution for the steiner problem
        using annealing algorithm
    :param graph: the graph for each we search a solution
    :param terms: the list of terminal nodes
    :param T_init: the initial temperature
    :param T_limit: the lowest temperature allowed
    :return: the solution found which is a set of edges
    """
    best = init(graph)
    T = T_init
    eval_best = eval_annealing(best, graph, terms)
    m = 0
    while(T>T_limit):
        sol = rand_neighbor(best)#, nb_changes=int(T)%len(best))
        eval_sol = eval_annealing(sol, graph, terms)
        if eval_sol <= eval_best :
            prob = 1
        else :
            prob = exp((eval_best - eval_sol)/T)
            print(f'prob = {prob}')
        if rd.random() <= prob :
            best = sol
        print(f'eval_best = {eval_best}')
        T *= lamb
        m += 1
    print(f'm = {m}')
    return Genetic_Algorithm.bool_to_edges(best, [e for e in graph.edges])

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

def rand_neighbor(solution : list, nb_changes = 1) :
    """
    Generates new random solution.
    :param solution: the solution for which we search a neighbor
    :param nb_changes: maximum number of the changes alowed
    :return: returns a random neighbor for the solution
    """
    new_solution = cp.deepcopy(solution)

    for _ in range(nb_changes):
        i = rd.choice(range(len(new_solution)))
        new_solution[i] = not new_solution[i]
    return new_solution

if __name__ == '__main__' :
    stein_file = 'data/test.std'
    #stein_file = 'data/B/b1.stp'
    my_class = Approximation.MySteinlibInstance()
    with open(stein_file) as my_file:
        my_parser = Approximation.SteinlibParser(my_file, my_class)
        my_parser.parse()
        terms = my_class.terms
        graph = my_class.my_graph
        #Approximation.print_graph(graph,terms)
        sol=recuit(graph,terms, 500, 1)
        Approximation.print_graph(graph,terms,sol)
        print(Approximation.eval_sol(graph,terms,sol))
