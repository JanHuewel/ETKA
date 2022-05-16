from defaultOptions import default_options, hyperparameter_limits as hpl

global options
options = default_options
global hyperparameter_limits
hyperparameter_limits = hpl

def init():
    """
    Initialize global parameters for options (dict containing information for multiple parts of the pipeline, see defaultOptions)
    and hyperparameter_limits (dict containing ranges to search in for hyperparameter optimization, see defaultOptions)

    This needs to be called before any training!
    """
    global options, hyperparameter_limits
    options = default_options
    hyperparameter_limits = hpl