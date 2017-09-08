#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 11:18:02 2017

@author: andrew
"""
import h5py 
import matplotlib.pyplot as plt
import os
import numpy as np 
from scipy.stats.kde import gaussian_kde
def main():
    fignum = [1]
    for filename in os.listdir('/home/andrew/REU/overcoming-catastrophic-master/09_08_17'):
        if filename.endswith('.h5'):
            try:
                f = h5py.File(filename, 'r')
                runs = int(f['params'][len(f['params']) - 1])
                d = {}
                run_dict = {}
                task_counter_dict = {}
                for runNum in range(runs):
                    run_dict['{}'.format(runNum + 1)] = 0
                    task_counter_dict['{}'.format(runNum + 1)] = 0
                
                for key in f.keys():
                    if key[0:5] == 'count':
                        if run_dict[str(int(key[-2:]))] == 0:
                            run_dict[str(int(key[-2:]))] += 1
                        else:
                            run_dict[str(int(key[-2:]))] += .5
                for dict_key in run_dict.keys():
                    while run_dict[dict_key] != 0:
                        task_counter_dict[dict_key] += 1
                        run_dict[dict_key] -= task_counter_dict[dict_key]
                
                complete_cycle = True
                for i in range(1, runs + 1):
                    if task_counter_dict[str(i)] == 0:
                        print(filename + " terminated early")
                        complete_cycle = False
                
                if complete_cycle == True:
                    """
                    change to range(runs) for complete looping
                    """
                    for run in range(runs):
                        EWC = 'run {} lambda {}'.format(str(run + 1), str(float(f['params'][2])))
                        SGD = 'run {} lambda {}'.format(str(run + 1), str(0))
                        
                        d[EWC] = np.zeros((task_counter_dict[str(run + 1)], task_counter_dict[str(run + 1)]))
                        d[SGD] = np.zeros((task_counter_dict[str(run + 1)], task_counter_dict[str(run + 1)]))
                        
                        for count in range(task_counter_dict[str(run + 1)]):
                            acc_EWC = []
                            acc_SGD = []
                            for task in range(count + 1):
                                if count == 0:
                                    acc_EWC = np.append(acc_EWC, f['count {} task {} lambda {} run {}'.format(str(count + 1), str(task + 1), str(0), str(run + 1))][len(f['count {} task {} lambda {} run {}'.format(str(count + 1), str(task + 1), str(0), str(run + 1))]) - 1])
                                    acc_SGD = np.append(acc_SGD, f['count {} task {} lambda {} run {}'.format(str(count + 1), str(task + 1), str(0), str(run + 1))][len(f['count {} task {} lambda {} run {}'.format(str(count + 1), str(task + 1), str(0), str(run + 1))]) - 1])
                                else:
                                    acc_EWC = np.append(acc_EWC, f['count {} task {} lambda {} run {}'.format(str(count + 1), str(task + 1), str(float(f['params'][2])), str(run + 1))][len(f['count {} task {} lambda {} run {}'.format(str(count + 1), str(task + 1), str(float(f['params'][2])), str(run + 1))]) - 1])
                                    acc_SGD = np.append(acc_SGD, f['count {} task {} lambda {} run {}'.format(str(count + 1), str(task + 1), str(0), str(run + 1))][len(f['count {} task {} lambda {} run {}'.format(str(count + 1), str(task + 1), str(0), str(run + 1))]) - 1])
                            while len(acc_EWC) < task_counter_dict[str(run + 1)]:
                                acc_EWC = np.append(acc_EWC, 0)
                            while len(acc_SGD) < task_counter_dict[str(run + 1)]:
                                acc_SGD = np.append(acc_SGD, 0)
                            d[EWC][count] = acc_EWC
                            d[SGD][count] = acc_SGD
                        
                        
                        
                        old_EWC = [0, d[EWC][0,0]]
                        latest_EWC = [0]
                        old_SGD = [0, d[SGD][0,0]]
                        latest_SGD = [0]
                        for row in range(1, len(d[EWC])):
                            old_acc_EWC_sum = 0
                            old_acc_SGD_sum = 0
                            
                            for col in range(row):
                                old_acc_EWC_sum += d[EWC][row, col]
                                old_acc_SGD_sum += d[SGD][row, col]
                                
                            old_EWC.append(old_acc_EWC_sum / float(row))
                            old_SGD.append(old_acc_SGD_sum / float(row))
                        
                        for row in range(0, len(d[EWC])):
                            latest_EWC.append(d[EWC][row, row])
                            latest_SGD.append(d[SGD][row, row])
                         
                          
                        plt.figure(num=fignum[0], figsize=(20,10))
                        plot1 = plt.subplot(111)
                        plot1.set_title(filename + " run: " + str(run + 1))
                        plt.plot(old_EWC, label="EWC old", marker = '^')
                        plt.plot(latest_EWC, label="EWC latest", marker = 'o')
                        """
                        plt.plot(old_SGD, label="SGD old", marker = 'D')
                        plt.plot(latest_SGD, label="SGD latest", marker = 's')
                        """
                        plt.xlabel('tasks')
                        plt.ylabel('accuracy')
                        plt.axis([1, 25, 0.5, 0.95])
                        plt.legend(loc=3)
                        plt.savefig(filename + "_run_ " + str(run + 1) + ".png")
                        plt.close()
                        plt.figure(num=fignum[0] + 1, figsize=(20, 10))
                        plot2 = plt.subplot(111)
                        
                        plot2.set_title(filename + "Error Sum Terms run {}".format(str(run + 1)))
                        plt.plot(f['Error Sum Terms run {}'.format(str(run + 1))], marker = 'o')
                        plt.xlabel('tasks')
                        plt.ylabel('Summed Term in EWC Error')
                        plt.axis([1, 25, 0.0, 1])
                        plt.savefig(filename + "_error_sums_run_" + str(run + 1) + ".png")
                        fignum[0] += 2
                        plt.close()
                        """
                        plot2.set_title(filename + "sum Fisher run {}".format(str(run + 1)))
                        plt.plot(f['fisher sum run {}'.format(str(run + 1))], marker = 'o')
                        plt.xlabel('tasks')
                        plt.ylabel('sum of average Fisher diagonal means')
                        plt.axis([1, 25, 0.0, 0.11])
                        plt.savefig(filename + "_fisher_sum_run_ " + str(run + 1) + ".png")
                        fignum[0] += 2
                        plt.close()
                    """
                    """
                    pdf_failure = []
                        
                    for r in range(runs):
                        pdf_failure.append(task_counter_dict['{}'.format(str(r + 1))])
                    
                    print(pdf_failure)
                    
                    pdf_fisher = []
                    
                    for r2 in range(runs):
                        pdf_fisher.append(f['fisher sum run {}'.format(str(r2 + 1))][len(f['fisher sum run {}'.format(str(r2 + 1))]) - 1])
                    
                    print(pdf_fisher)
                    
                    for index_fisher in range(len(pdf_fisher)):
                        if index_fisher == 'nan':
                            del pdf_fisher[index_fisher]
                            del pdf_failure[index_fisher]
                    
                    fail_gauss = gaussian_kde(pdf_failure)
                    fisher_gauss = gaussian_kde(pdf_fisher)
                    
                    dist_space_fail = np.linspace(min(pdf_failure), max(pdf_failure), 100)
                    dist_space_fisher = np.linspace(min(pdf_fisher), max(pdf_fisher), 100)
                    
                    
                    plt.figure(num=fignum[0], figsize=(20, 10))
                    fail = plt.subplot(111)
                    fail.set_title("PDF of Network Failure Task Count")
                    plt.plot(dist_space_fail, fail_gauss(dist_space_fail))
                    plt.xlabel("task count at failure")
                    plt.ylabel("probability density")
                    plt.savefig(filename + "failure_prob_dens.png")
                    plt.close()
                    plt.figure(num=fignum[0] + 1, figsize=(20, 10))
                    fisher = plt.subplot(111)
                    fisher.set_title("PDF of Fisher Sum Just Prior to Failure")
                    plt.plot(dist_space_fisher, fisher_gauss(dist_space_fisher))
                    plt.xlabel("fisher sum")
                    plt.ylabel("probability density")
                    plt.savefig(filename + "fisher_prob_dens.png")
                    plt.close()
                    fignum[0] += 2
                    """
	    except IOError:
                print(filename + " unable to be opened")
            
           
if __name__ == "__main__":
    main()
