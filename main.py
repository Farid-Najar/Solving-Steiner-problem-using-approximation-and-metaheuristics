import Approximation
import Genetic_Algorithm as GA
import matplotlib.pyplot as plt

path_B = 'data/B/'
path_C = 'data/C/'
stein_file = 'data/test.std'

#optimal solutions for files B
B_opts = [82,83,138,59,61,122,111,104,220,86,88,174,165,235,318,127,131,218]
#optimal solutions for files C
C_opts = [85,144,754,1079,1579,55,102,509,707,1093,32,46,258,323,556,11,18,113,146,267]


def simulation_Approximation() :
    #my_class = Approximation.MySteinlibInstance()
    #Results of B
    results_B = Approximation.simulation(len(B_opts), 'data/B/b')
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
    results_C = Approximation.simulation(len(C_opts), 'data/C/c')
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

    """
    with open(stein_file) as my_file:
        my_parser = Approximation.SteinlibParser(my_file, my_class)
        my_parser.parse()
        terms = my_class.terms
        graph = my_class.my_graph
        Approximation.print_graph(graph,terms)
        sol=Approximation.approx_steiner(graph,terms)
        Approximation.print_graph(graph,terms,sol)
        print(Approximation.eval_sol(graph,terms,sol))
    """

def simulation_Genetic():
    results = [(None, None) for _ in range (25)]
    res = [None for _ in range(25)]
    #simulation on b1.stp
    for i in range(len(res)):
        GA.eval_file(1, path_B+'b', results, i)
        res[i] = results[i][1]
        plt.plot(range(len(results[i][1])),res[i], color = '#7DD6CA')

    plt.axhline(B_opts[0], color='red')
    plt.clabel()
    plt.show()

if __name__ == '__main__' :
    print('Simulation begins')
    print('------------------------------------------------------------------------------------')
    print('processing simulation for Approximation.py')
    #simulation_Approximation()
    print('simulation for Approximation.py done')
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    print('------------------------------------------------------------------------------------')
    print('processing simulation for Genetic_Algorithm.py')
    simulation_Genetic()
    print('simulation for Genetic_Algorithm.py done')
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
