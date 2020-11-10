import Approximation
import Genetic_Algorithm as GA
import Annealing_Algorithm as AA
import matplotlib.pyplot as plt
import numpy as np

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

def simulation_GA():
    number_of_simulation = 100
    nfile = 1
    tfile = 'b'
    path = path_B
    if tfile == 'c':
        path = path_C
    opt = 0
    if tfile == 'b':
        opt = B_opts[nfile-1]
    else:
        opt = C_opts[nfile-1]
    results = GA.simulation(number_of_simulation,nfile,path+tfile)
    res = [np.array(p[1]) for p in results]
    
    average_values = np.zeros(len(res[0]))
    for r in res:
        average_values =average_values + r
    average_values = np.array(average_values) / number_of_simulation
    
    error_values = [0 for i in range(len(average_values))]
    for j in range(len(error_values)):
        if j%10==0 :
            for i in range(len(res)):
                error_values[j] += (res[i][j] - average_values[j])**2
            error_values[j] = np.sqrt(error_values[j]/number_of_simulation)

    plt.errorbar(range(len(average_values)),average_values,yerr = error_values, ecolor = "black", linewidth = 1, elinewidth = 1)


    plt.axhline(opt, color='red', label = "Optimal solution")
    plt.title(f'The evolution of the best evaluation (in average) \nfor graph {tfile}{nfile}.stp for {number_of_simulation} simulation ')
    plt.xlabel("steps")
    plt.ylabel("evaluation")
    plt.legend()
    plt.ylim((opt-5,max(opt*2,average_values[-1]+10)))
    plt.savefig(f'best_{tfile}{nfile}_evaluation_genetic_ones_init.png')
    plt.show()




def simulation_recuit():
    number_of_simulation = 10
    nfile = 1
    tfile = 'b'
    path = path_B
    if tfile == 'c':
        path = path_C
    opt = 0
    if tfile == 'b':
        opt = B_opts[nfile-1]
    else:
        opt = C_opts[nfile-1]
    results = AA.simulation(number_of_simulation,nfile,path+tfile)
    res = [np.array(p[1]) for p in results]
    
    average_values = np.zeros(len(res[0]))
    for r in res:
        average_values =average_values + r
    average_values = np.array(average_values) / number_of_simulation
    
    error_values = [0 for i in range(len(average_values))]
    for j in range(len(error_values)):
        if j%10==0 :
            for i in range(len(res)):
                error_values[j] += (res[i][j] - average_values[j])**2
            error_values[j] = np.sqrt(error_values[j]/number_of_simulation)

    plt.errorbar(range(len(average_values)),average_values,yerr = error_values, ecolor = "black", linewidth = 1, elinewidth = 1)


    plt.axhline(opt, color='red', label = "Optimal solution")
    plt.title(f'The evolution of the best evaluation (in average) \nfor graph {tfile}{nfile}.stp for {number_of_simulation} simulation ')
    plt.xlabel("steps")
    plt.ylabel("evaluation")
    plt.legend()
    plt.ylim((opt-5,max(opt*2,average_values[-1]+10)))
    plt.savefig(f'best_{tfile}{nfile}_evaluation_genetic_ones_init.png')
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
    print(f'best for c1 = {C_opts[0]}')
    #res = [(None, None), None]
    #GA.eval_file(1, path_C+'c', res, 0)
    #print(f'genetic = {res[0][0]}')
    #Approximation.eval_file(0, path_C+'c', res, 1)
    #print(f'Approximation = {res[1]}')
    #plt.plot(range(len(res[0][1])), res[0][1])
    #plt.show()
    simulation_recuit()
    print('simulation for Genetic_Algorithm.py done')
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
