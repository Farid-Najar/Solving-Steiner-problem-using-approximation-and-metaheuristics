import matplotlib.pyplot as plt
import networkx as nx
from steinlib.instance import SteinlibInstance
from steinlib.parser import SteinlibParser
from threading import Thread

# draw a graph in a window
def print_graph(graph,terms=None,sol=None):

    pos=nx.kamada_kawai_layout(graph)

    nx.draw(graph,pos)
    if (not (terms is None)):
        nx.draw_networkx_nodes(graph,pos, nodelist=terms, node_color='r')
    if (not (sol is None)):
        nx.draw_networkx_edges(graph,pos, edgelist=sol, edge_color='r')
    plt.show()
    return


# verify if a solution is correct and evaluate it
def eval_sol(graph,terms,sol):

    if (len(sol)==0):
        print("Error: empty solution")
        return -1

    graph_sol = nx.Graph()
    for (i,j) in sol:
        if (i, j) not in graph.edges :
            print('Error : invalide edge')
            return -1
        graph_sol.add_edge(i,j,weight=graph[i][j]['weight'])

    # is sol a tree
    if (not (nx.is_tree(graph_sol))):
        print ("Error: the proposed solution is not a tree")
        return -1

    # are the terminals covered
    for i in terms:
        if not i in graph_sol:
            print ("Error: a terminal is missing from the solution")
            return -1

    # cost of solution
    cost = graph_sol.size(weight='weight')

    return cost



# compute an approximate solution to the steiner problem
def approx_steiner(graph,terms):
    """
    :param graph: a graph of type nx.Graph
    :param terms: the list of terminals in the given graph
    :return: list of edges which makes a tree
    """
    #print(terms)
    res = set()
    G = nx.complete_graph(terms)
    paths = {}
    counter = len(terms) #it allows us to stop when we found paths for terms
    seen = set(set())
    #In this loop, we compute the shortest path and update the weights
    #  in the graph 'G'. Then we memorize the optimal paths found in
    #  'paths'. Note that we try to avoid repetitions by looking 'seen'
    #  in every iteration.
    #Once we found optimal paths for all terminals, we get out of the loop.
    for node, (w, path) in nx.all_pairs_dijkstra(graph):
        if node in terms :
            for terminal in terms :
                if (node, terminal) not in seen and node != terminal:
                    G.edges[node, terminal]['weight'] = w[terminal]
                    paths.update({(node, terminal): (w[terminal], path[terminal])})
                    seen.add((node, terminal))
            counter -= 1
        if counter == 0 :
            break

    #This is the second part of the algorithm when we search the spanning tree
    min_tree = nx.minimum_spanning_tree(G)
    #print(min_tree.edges)
    #print(paths)

    #In this loop we add the edges of the paths.
    #Note that we use a set to have the unicity.
    for edge in min_tree.edges :
        _, path = paths[edge]
        for i in range(len(path)-1):
            res.add((path[i], path[i+1]))
    #print(res)
    return res



# class used to read a steinlib instance
class MySteinlibInstance(SteinlibInstance):

    def __init__(self):
        self.my_graph = nx.Graph()
        self.terms = []

    def terminals__t(self, line, converted_token):
        self.terms.append(converted_token[0])

    def graph__e(self, line, converted_token):
        e_start = converted_token[0]
        e_end = converted_token[1]
        weight = converted_token[2]
        self.my_graph.add_edge(e_start,e_end,weight=weight)


def eval_file(number_file, path, res, i):
    """
    Given the file's number, this function finds a solution for the associated graph and store it
        in the list passed as argument.
    :param number_file: the file's number
    :param path: the path to the file
    :param res: the list in which the result will be stored in
    :param i: the index of the free place in the list
    :return: the total weight of the solution
    """
    print(f"Processing file number {number_file+1} begins.\n")
    my_class = MySteinlibInstance()
    with open(path+f'{number_file+1}.stp') as file :
        my_parser = SteinlibParser(file, my_class)
        my_parser.parse()
        terms = my_class.terms
        graph = my_class.my_graph
        #print_graph(graph,terms)
        sol = approx_steiner(graph,terms)
        #print_graph(graph,terms,sol)
        result = eval_sol(graph,terms,sol)
    print(f'Processing file number {number_file+1} ended.\n')
    res[i] = result

def simulation(data_size, path):
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

