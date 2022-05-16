from main import main
import pandas as pd
import os
import torch
import random
import numpy as np

def set_all_seeds(seed: int = 42):
    """
    Set all seeds of libs with a specific function for reproducibility of results
    :param seed: seed to use
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def clear(directory):
    df = pd.DataFrame(columns=["dataset", "PER AKS", "CPD AKS", "CPD HPO", "diff PER", "diff HPO"])
    df.to_csv(f"{directory}/performance.csv")
    df.to_csv(f"{directory}/runtime.csv")

def evaluate_directory(directory):
    df_runtime = pd.read_csv(f"{directory}/runtime.csv")
    df_performance = pd.read_csv(f"{directory}/performance.csv")
    print(df_runtime[df_runtime.columns[-5:]])
    print(df_performance[df_performance.columns[-6:]])
    print(f"RUNTIME: \ncomparison PER: {sum(df_runtime['diff PER'])/len(df_runtime)} \ncomparison HPO: {sum(df_runtime['diff HPO'])/len(df_runtime)}")
    print(f"stds: \nPER {df_runtime['diff PER'].std()}\nHPO: {df_runtime['diff HPO'].std()}\n\n")
    print(f"PERFORMANCE: \ncomparison PER: {sum(df_performance['diff PER'])/len(df_runtime)} \ncomparison HPO: {sum(df_performance['diff HPO'])/len(df_runtime)}")
    print(f"stds: \nPER {df_performance['diff PER'].std()}\nHPO: {df_performance['diff HPO'].std()}")

datasets_simul = ["simul_noisy_1cp_1",
            "simul_noisy_1cp_2",
            "simul_noisy_1cp_3",
            "simul_noisy_1cp_4",
            "simul_noisy_2cp_1",
            "simul_noisy_2cp_2",
            "simul_noisy_2cp_3",
            "simul_noisy_2cp_4",
            "simul_varnoise_1cp_1",
            "simul_varnoise_1cp_2",
            "simul_varnoise_1cp_3",
            "simul_varnoise_1cp_4",
            "simul_varnoise_2cp_1",
            "simul_varnoise_2cp_2",
            "simul_varnoise_2cp_3",
            "simul_varnoise_2cp_4",
            "simul_noiseless_1cp_1",
            "simul_noiseless_1cp_2",
            "simul_noiseless_1cp_3",
            "simul_noiseless_1cp_4",
            "simul_noiseless_2cp_1",
            "simul_noiseless_2cp_2",
            "simul_noiseless_2cp_3",
            "simul_noiseless_2cp_4"]


datasets_real = ["d1_solar",
            "d2_mauna",
            "d6_airline",
            "d7_wheat",
            "d8_temperature",
            "d9_internet",
            "d10_call_centre",
            "d11_radio",
            "d12_gas_production",
            "d13_sulphuric",
            "d14_unemployment",
            "d15_births",
            "d16_wages",
            "d17_airquality"
            ]


directory = "Results"

set_all_seeds()
clear(directory)
main(datasets_simul, directory, cpd_threshold=5.0, cpd_tolerance=0.5)
evaluate_directory(directory)