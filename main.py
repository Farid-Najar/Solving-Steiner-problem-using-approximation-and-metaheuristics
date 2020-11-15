import Approximation
import Genetic_Algorithm as GA
import Annealing_Algorithm as AA
import matplotlib.pyplot as plt
import numpy as np
from threading import Thread
import time

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


def simulation(data_size : int,nbr_file : int, path : str, target):
    """
    This function does the simulation with our data using parallel operations
    :param data_size : The number of simulations
    :param nbr_file : wich file we are evaluating
    :param path : The path to the file
    :param target : The target evaluation function
    :return : results of the simulation
    """
    res = [0 for _ in range(data_size)]
    threads = []
    for i in range(data_size):
        threads.append(Thread(target = target, args = (nbr_file,path,res,i)))
        threads[i].start()

    for i in range(data_size):
        threads[i].join()

    return res



def plotEvaluation(results,nbr_file : int,path : str, labels = [""],target_name = ""):
    """
    This function plot the results of the evaluation 
    :param results : the tab of the results of the evaluations
    :param nbr_file : wich file we are evaluating
    :param path : the path to the file
    :params labels : labels of the plots
    :param target_name : name of the target evaluation function (used for the name of the saved png)
    :return : none
    """
    fig,ax = plt.subplots(1,1)
    ax.set_yscale("log")
    for res in range(len(results)):
        data = [np.array(p[1]) for p in results[res]]
        number_of_simulation = len(data)

        average_values = np.zeros(len(data[0]))
        for d in data:
            average_values =average_values + d
        average_values = np.array(average_values) / number_of_simulation
    
        error_values = [0 for i in range(len(average_values))]
        for j in range(len(error_values)):
            if j%int(len(error_values)/50)==0 :
                for i in range(len(data)):
                    error_values[j] += (data[i][j] - average_values[j])**2
                error_values[j] = np.sqrt(error_values[j]/number_of_simulation)
        opt = 0
        tfile = ''
        if path == 'data/B/b':
            opt = B_opts[nbr_file-1]
            tfile = 'b'
        else:
            opt = C_opts[nbr_file-1]
            tfile = 'c'
    
        
        ax.errorbar(range(len(average_values)),average_values,yerr = error_values, ecolor = "black", linewidth = 1, elinewidth = 1, label = labels[res])
            
    
        #ax.ylim((opt-5,max(opt*2,average_values[-1]+10)))
    plt.title(f'{target_name} : The evolution of the best evaluation (in average) \nfor graph {tfile}{nbr_file}.stp for {number_of_simulation} simulations')
    plt.xlabel("steps")
    plt.ylabel("evaluation")
    ax.legend()
    ax.axhline(opt, color='red', label = "Optimal solution")
    plt.savefig(f'best_{tfile}{nbr_file}_evaluation_{target_name}.png')
    plt.show()

def simulation_genetic(nbr_file : int, path : str, number_of_simulation = 100, target_name = "genetic", label = ""):
    """
    this function does the simulation for the genetic algorithm
    :param nbr_file: wich file we are evaluating
    :param path: the path to the file
    :param number_of_simulation: how many simulations we do
    :param target_name : the name of the target function
    :param label : label of the curve
    :return: none
    """
    plotEvaluation([simulation(number_of_simulation,nbr_file,path,GA.eval_file)]
                   ,nbr_file
                   ,path
                   ,labels = [label]
                   ,target_name = target_name)

def simulation_recuit(nbr_file : int, path : str,number_of_simulation = 100, target_name = "recuit", label = ""):
    """
    this function does the simulation for the annealing algorithm
    :param nbr_file: wich file we are evaluating
    :param path: the path to the file
    :param number_of_simulation: how many simulations we do
    :param target_name : the name of the target function
    :param : label of the curve
    :return: none
    """
    plotEvaluation([simulation(number_of_simulation,nbr_file,path,AA.eval_file)]
                   ,nbr_file
                   ,path
                   ,labels = [label]
                   ,target_name = target_name)



    
if __name__ == '__main__' :
    print('Simulation begins')
    print('------------------------------------------------------------------------------------')

    nfile = 2
    path = "data/B/b"

    #simulation_genetic(nfile,path,target_name = "genetic max_pop 30", label = "genetic")
    #simulation_recuit(nfile,path,target_name = "recuit 20000 0.999", label = "recuit")

    
    #diff 
    plotEvaluation([simulation(100,nfile,path,GA.eval_file)
                    ,simulation(100,nfile,path,AA.eval_file_m)]
                   ,nfile
                   ,path
                   ,target_name = "diff genetic multiple"
                   ,labels = ["genetic","recuit multiple"])


    
