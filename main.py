import copy
from globalParams import options
import torch
from kernelSearch import CKS, AKS
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.kernels import RBFKernel, PeriodicKernel, LinearKernel, ScaleKernel
from helpFunctions import get_string_representation_of_kernel as gsr, print_formatted_hyperparameters as pfh
from gpytorch.constraints import GreaterThan
from GaussianProcess import ExactStreamGPModel
from Datasets import *
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

torch.set_default_dtype(torch.float64)
options["plotting"]["legend"] = False

# potentially replace with process time
def time_func(f):
    def wrapper(*args, **kwargs):
        #t = time.time()
        t = time.process_time()
        res = f(*args, **kwargs)
        #print(f"Timing results: {time.time()-t} sec")
        time_diff = time.process_time()-t
        print(f"Timing results: {time_diff} sec")
        return time_diff
    return wrapper

# parameters
def main(data_names, directory = "figures/KI2022", cpd_threshold = 5.0, cpd_tolerance=0.5, save=True):
    if not save:
        overall_time = 0.0
        overall_performance = 0.0
    for data_name in data_names:
        data = globals()["dataset_" + data_name]
        data.normalize_z()

        lengthscale_minimum = 0

        base_kernels = [RBFKernel(lengthscale_constraint=GreaterThan(lengthscale_minimum)),
                        PeriodicKernel(lengthscale_constraint=GreaterThan(lengthscale_minimum)),
                        LinearKernel(lengthscale_constraint=GreaterThan(lengthscale_minimum))]
        training_points = int(len(data)/5+1)
        refitting_cycle = int(len(data)/5+1)
        AKS_iterations = 1
        AKS_max_complexity = 3

        # initial model search
        k_0, l_0 = CKS(data[:training_points],
                       GaussianLikelihood(),
                       base_kernels,
                       AKS_max_complexity)

        if save:
            # Test 1: AKS periodic
            print("Test 1: AKS periodic")
            m0 = ExactStreamGPModel(data, copy.deepcopy(l_0), copy.deepcopy(k_0), training_points)
            t0 = time_func(m0.process_data)(1, refitting_cycle, "AKS", base_kernels, AKS_iterations, AKS_max_complexity)
            likelihoods_0 = m0.likelihood_log
            p0 = sum(likelihoods_0)/len(likelihoods_0)
            p0 = sum([abs(m0.predictions[i] - data.Y[training_points+i+1]).item() for i in range(len(m0.predictions)-1)]) / (len(m0.predictions)-1)
            print(f"average likelihood: {p0}\n")

        # Test 2: AKS CPD
        print("Test 2: AKS CPD")
        m1 = ExactStreamGPModel(data, copy.deepcopy(l_0), copy.deepcopy(k_0), training_points)
        t1 = time_func(m1.process_data_cpd)("AKS", base_kernels, AKS_iterations, AKS_max_complexity, cpd_threshold, cpd_tolerance=cpd_tolerance)
        likelihoods_1 = m1.likelihood_log
        p1 = sum(likelihoods_1)/len(likelihoods_1)
        p1 = sum([abs(m1.predictions[i] - data.Y[training_points+i+1]).item() for i in range(len(m1.predictions)-1)]) / (len(m1.predictions)-1)
        print(f"average likelihood: {p1}\n")
        #print(f"Kernels: {[gsr(k) for k in m1.kernels]}\n")

        if save:
            # Test 3: HPO CPD
            print("Test 3: HPO CPD")
            m2 = ExactStreamGPModel(data, copy.deepcopy(l_0), copy.deepcopy(k_0), training_points)
            t2 = time_func(m2.process_data_cpd)("HPO", cpd_threshold=cpd_threshold, cpd_tolerance = cpd_tolerance)
            likelihoods_2 = m2.likelihood_log
            p2 = sum(likelihoods_2)/len(likelihoods_2)
            p2 = sum([abs(m2.predictions[i] - data.Y[training_points+i+1]).item() for i in range(len(m2.predictions)-1)]) / (len(m2.predictions)-1)
            print(f"average likelihood: {p2}\n")


        # Additional Evaluation:
        """
        print("Additional Sttatistics")
        print("KERNELS: PER")
        for k in m0.kernels:
            print(gsr(k))
            pfh(k)
            print()
        print("")
        print("KERNELS: CPD")
        for k in m1.kernels:
            print(gsr(k))
            pfh(k)
            print()
        print("")
        """

        if save:
            # Plotting:
            f, ax = m0.plot_model(return_figure=True)
            #ax.plot(data.X[training_points+1:], m0.predictions[:-1], "r.")
            #plt.show()
            f.savefig(f"{directory}/{data_name}_PER_AKS.png")
            f, ax = m1.plot_model(return_figure=True)
            #ax.plot(data.X[training_points+1:], m1.predictions[:-1], "r.")
            #plt.show()
            f.savefig(f"{directory}/{data_name}_CPD_AKS.png")
            f, ax = m2.plot_model(return_figure=True)
            #ax.plot(data.X[training_points+1:], m2.predictions[:-1], "r.")
            #plt.show()
            f.savefig(f"{directory}/{data_name}_CPD_HPO.png")

            # Save Evaluation

            runtime_df = pd.read_csv(f"{directory}/runtime.csv")
            performance_df = pd.read_csv(f"{directory}/performance.csv")

            diff_runtime_PER = (t1/t0 - 1) * 100
            diff_runtime_HPO = (t1/t2 - 1) * 100
            diff_performance_PER = (p1/p0 - 1) * 100
            diff_performance_HPO = (p1/p2 - 1) * 100

            d1 = {"dataset": data_name, "PER AKS": t0, "CPD AKS": t1, "CPD HPO": t2,
                  "diff PER": diff_runtime_PER, "diff HPO": diff_runtime_HPO}
            d2 = {"dataset": data_name, "PER AKS": p0, "CPD AKS": p1, "CPD HPO": p2,
                  "diff PER": diff_performance_PER, "diff HPO": diff_performance_HPO}

            runtime_df = runtime_df.append(d1, ignore_index=True)
            performance_df = performance_df.append(d2, ignore_index=True)

            runtime_df.to_csv(f"{directory}/runtime.csv")
            performance_df.to_csv(f"{directory}/performance.csv")
        else:
            overall_time += t1
            overall_performance += p1
    if not save:
        return overall_time, overall_performance

if __name__=="__main__":
    main(["d6_airline"], "Results/KI2022/test", save=True)