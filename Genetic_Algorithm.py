import itertools as it
import networkx as nx
from threading import Thread
import Approximation
import random as rd
import copy as cp


stein_file = "data/B/b1.stp"
#stein_file = "data/test.std"

def genetic(graph, terms, nb_iter=500, taille_max_population = 10):
    """
    This is the main.
    :param graph: the graph for each we search a solution
    :param terms: the list of terminal nodes
    :param nb_iter: maximum number of iterations
    :return:
    """
    graph_edges = [e for e in graph.edges]
    best = init(graph)
    best_list = []
    #print(bool_to_edges(best, graph_edges))
    #print(terms)
    eval_best = eval_genetic(best, graph, terms)
    best_list.append(eval_best)
    solutions = {eval_best : best}
    i = 0
    while(i < nb_iter ):#or TP1.eval_sol(graph, terms, bool_to_edges(best, graph_edges)) == -1):
        #print(len(solutions))
        #print(f'{i} = {eval_best}')
        generation(graph, terms, solutions)
        #print(f'keys = {solutions.keys()}')
        #print(f'len(solutions) = {len(solutions)}')
        for eval_sol, sol in solutions.items() :
            if sol != best :                #print(f'{i} = {eval_sol}')
                if eval_sol < eval_best:
                    best = cp.copy(sol)
                    eval_best = eval_sol
                    best_list.append(eval_best)
        solutions = selection(solutions, taille_max_population)
        i+=1
    return bool_to_edges(best, graph_edges), best_list

def selection(solutions : dict, taille_max_population) :
    solutions = {key : val for key, val in sorted(solutions.items(), key= lambda sol: sol[0])}
    return dict(it.islice(solutions.items(), taille_max_population))

def init(graph):
    """
    This gives the first proposition of solution for the algorithm.
    :param graph: the graph for each we search a solution
    :return: returns a random boolean list
    """
    #sol = TP1.approx_steiner(graph, terms)
    return [round(rd.random()) == 1 for _ in range(len(graph.edges))]#edges_to_bool(sol, [e for e in graph.edges])


def generation(graph, terms, solutions : dict, nb_changes = 2) :
    """
    Generates new generation of solutions.
    :param graph: the graph for each we search a solution
    :param terms: the list of terminal nodes
    :param solutions: it is a dict containing the current population
    :param nb_changes: the maximum number of crossing alloxed
    :return: it doesn't return anything but updates solutions with new generation
    """
    #print(solutions)
    #new_generation = {}
    new_generation = cp.deepcopy(list(solutions.values()))

    for i in range(nb_changes) :
        s1 = rd.choice(list(solutions.values()))
        s2 = rd.choice(list(solutions.values()))
        new_generation.append([s1[i] if i < len(s1)//2 else s2[i] for i in range(len(s1))])

    for i in range(len(new_generation)) :
        j = rd.choice(range(len(new_generation[i])))
        new_generation[i][j] = not new_generation[i][j]
        solutions.update({eval_genetic(new_generation[i], graph, terms) : new_generation[i]})

def eval_genetic(sol : list, graph, terms : list, malus = 500):
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

def edges_to_bool(sol : set, graph_edges):
    """
    It converts a set of edges to a list of booleans
        k-th element of the list tells if graph_edges[k] is in the set or not
    :param edges: set of edges
    :return: list of booleans
    """
    #print(sol)
    for s in sol :
        if s not in graph_edges :
            print(f'probleme pour {s}')
    #print(sol.intersection(graph_edges))
    res = [False for _ in range(len(graph_edges))]
    for i in range(len(graph_edges)):
        res[i] = (graph_edges[i] in sol)
    return res

def bool_to_edges(sol : list, graph_edges):
    """
    It converts a boolean table into edges
    :param sol: a table of boolean
    :param graph_edges: the list of the graph's edges
    :return: a set of edges
    """
    solution = set()
    assert len(graph_edges) == len(sol)
    for i in range(len(sol)):
        if sol[i] :
            solution.add(graph_edges[i])
    return solution


def eval_file(number_file : int, path : str, res : list, i : int):
    """
    Given the file's number, this function finds a solution for the associated graph and store it
        in the list passed as argument.
    :param number_file: the file's number
    :param path: the path to the file
    :param res: the list in which the result will be stored in
    :param i: the index of the free place in the list
    :return: the total weight of the solution
    """
    print(f"Processing file number {number_file} begins for genetic algorithm.\n")
    my_class = Approximation.MySteinlibInstance()
    with open(path+f'{number_file+1}.stp') as file :
        my_parser = Approximation.SteinlibParser(file, my_class)
        my_parser.parse()
        terms = my_class.terms
        graph = my_class.my_graph
        #print_graph(graph,terms)
        sol, best_list = genetic(graph,terms)
        #print_graph(graph,terms,sol)
        result = Approximation.eval_sol(graph,terms,sol)
    print(f'Processing file number {number_file} ended.\n')
    res[i] = (result, best_list)


def simulation(data_size : int, path : str):
    """
    This function does the simulation with our data using parallel operations
    :param data_size: The number of files
    :param path: The path to those files
    :return: results of the simulation
    """

    res = [0 for _ in range(data_size)]
    threads = []
    for i in range(data_size) :
        threads.append(Thread(target=eval_file, args=(i, path, res, i)))
        threads[i].start()

    for i in range(data_size):
        threads[i].join()
    print(res)
    return res