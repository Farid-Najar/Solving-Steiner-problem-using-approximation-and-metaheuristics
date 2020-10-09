import sys
import itertools as it
import matplotlib.pyplot as plt
import networkx as nx
from steinlib.instance import SteinlibInstance
from steinlib.parser import SteinlibParser
import scipy
import concurrent.futures as cf
import matplotlib.pyplot as plt

stein_file = "data/B/b1.stp"

#optimal solutions for files B
B_opts = [82,83,138,59,61,122,111,104,220,86,88,174,165,235,318,127,131,218]
#optimal solutions for files C
C_opts = [85,144,754,1079,1579,55,102,509,707,1093,32,46,258,323,556,11,18,113,146,267]

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


def eval_file(number_file, path):
    """
    Given the file's number, this function find a solution for the associated graph
    :param number_file: the file's number
    :return: the total weight of the solution
    """
    my_class = MySteinlibInstance()
    with open(path+f'{number_file+1}.stp') as file :
        my_parser = SteinlibParser(file, my_class)
        my_parser.parse()
        terms = my_class.terms
        graph = my_class.my_graph
        #print_graph(graph,terms)
        sol = approx_steiner(graph,terms)
        #print_graph(graph,terms,sol)
        res = eval_sol(graph,terms,sol)
    return res

def simulation(data_size, path):
    """
    This function does the simulation with our data using parallel operations
    :param data_size: The number of files
    :param path: The path to those files
    :return: results of the simulation
    """
    with cf.ThreadPoolExecutor() as executor :
        res = [executor.submit(eval_file, i, path).result() for i in range(data_size)]
    print(res)
    return res


if __name__ == "__main__":
    my_class = MySteinlibInstance()
    #Results of B
    results_B = simulation(len(B_opts), 'data/B/b')
    approximation_rate_B = sum([(results_B[i] - B_opts[i])/B_opts[i] for i in range(len(results_B))])/len(results_B)
    plt.scatter(range(1,len(B_opts)+1), results_B, label='Approximation')
    plt.scatter(range(1, len(B_opts)+1), B_opts, label='Optimal')
    plt.title('Simulation on B files')
    plt.text(1,260, f"Approximation rate = {round(approximation_rate_B, 2)}")
    plt.xlabel('Number of file')
    plt.xticks(range(1, len(B_opts)+1))
    plt.ylabel('Weight')
    plt.legend()
    plt.savefig('plot_B.png')
    plt.show()

    #Results of C
    results_C = simulation(len(C_opts), 'data/C/c')
    approximation_rate_C = sum([(results_C[i] - C_opts[i])/C_opts[i] for i in range(len(results_C))])/len(results_C)
    plt.scatter(range(1, len(C_opts)+1), results_C, label='Approximation')
    plt.scatter(range(1, len(C_opts)+1), C_opts, label='Optimal')
    plt.title('Simulation on C files')
    plt.text(13,1300, f"Approximation rate = {round(approximation_rate_C, 2)}")
    plt.xlabel('Number of file')
    plt.xticks(range(1, len(C_opts)+1))
    plt.ylabel('Weight')
    plt.legend()
    plt.savefig('plot_C.png')
    plt.show()

    with open(stein_file) as my_file:
        my_parser = SteinlibParser(my_file, my_class)
        my_parser.parse()
        terms = my_class.terms
        graph = my_class.my_graph
        print_graph(graph,terms)
        sol=approx_steiner(graph,terms)
        print_graph(graph,terms,sol)
        print(eval_sol(graph,terms,sol))