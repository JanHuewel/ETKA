"""
This script provides all custom default options used in different steps of GP operations,
as well as operations to fill an arbitrary dictionary of options with the missing default values.
"""

default_options = {
    "training": {"print_training_output" : False,
                 "print_optimizing_output": False,
                 "max_iter": 10,
                 "learning_rate": 0.1,
                 "restarts": 10,
                 "optimization method": "botorch"},
    "kernel search": {"print": False,
                      "probability graph": False,
                      "multithreading": True},
    "plotting": {"border_ratio": 0.0,
                 "sample_points": 1000,
                 "legend": True},
    "data generation": {"const": {},
                        "lin": {"a": 1},
                        "per": {"p": 1, "a": 1, "p_offset": 0}}}


hyperparameter_limits = {"RBFKernel": {"lengthscale": [0,1]},
                         "LinearKernel": {"variance": [0,1]},
                         "PeriodicKernel": {"lengthscale": [0,10],
                                            "period_length": [0,10]},
                         "ScaleKernel": {"outputscale": [0,100]},
                         "Noise": [1e-4,2e-2],
                         "Mean": [0.0,1.0]}

def fill_default_options(options : dict):
    fill_tree(options, default_options)


def fill_tree(tree: dict, default_tree: dict):
    for key in default_tree:
        if key not in tree:
            tree[key] = default_tree[key]
        elif isinstance(tree[key], dict):
            fill_tree(tree[key], default_tree[key])
