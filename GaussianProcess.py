import gpytorch as gpt
import torch
import matplotlib.pyplot as plt
import copy
from globalParams import options, hyperparameter_limits
from botorch.fit import fit_gpytorch_model
from botorch.optim.fit import fit_gpytorch_torch
from helpFunctions import get_kernels_in_kernel_expression
from Data import Data

class ExactGPModel(gpt.models.ExactGP):
    """
    A Gaussian Process class.
    This class saves input and target data, the likelihood function and the kernel.
    It can be used to train the hyperparameters or plot the mean function and confidence band.
    """
    def __init__(self, data, likelihood, kernel):
        super(ExactGPModel, self).__init__(data.X, data.Y, likelihood)
        self.mean_module = gpt.means.ZeroMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpt.distributions.MultivariateNormal(mean_x, covar_x)

    def predict(self, x):
        self.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpt.settings.fast_pred_var():
            observed_pred = self.likelihood(self(x))
        return observed_pred.mean

    def train_model(self):
        self.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam([{"params": self.parameters()}], lr=options["training"]["learning_rate"])
        mll = gpt.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        for i in range(options["training"]["max_iter"]):
            optimizer.zero_grad()
            output = self.__call__(self.train_inputs[0])
            loss = -mll(output, self.train_targets)
            loss.backward()
            if options["training"]["print_training_output"]:
                parameter_string = ""
                for param_name, param in self.covar_module.named_parameters():
                    parameter_string += f"{param_name:20}: raw: {param.item():10}, transformed: {self.covar_module.constraint_for_parameter_name(param_name).transform(param).item():10}\n"
                parameter_string += f"{'noise':20}: raw: {self.likelihood.raw_noise.item():10}, transformed: {self.likelihood.noise.item():10}"
                print(
                f"HYPERPARAMETER TRAINING: Iteration {i} - Loss: {loss.item()}  \n{parameter_string}")
            optimizer.step()

    def get_current_loss(self):
        self.train()
        self.likelihood.train()
        mll = gpt.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        output = self.__call__(self.train_inputs[0])
        loss = -mll(output, self.train_targets)
        return loss.item()

    def get_ll(self, data: Data = None, X = None, Y = None):
        self.eval()
        self.likelihood.eval()
        mll = gpt.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        if data:
            output = self.__call__(data.X)
            return torch.exp(mll(output, data.Y)).item()
        elif not (X is None or Y is None):
            output = self.__call__(X)
            return torch.exp(mll(output, Y)).item()

    def optimize_hyperparameters(self):
        """
        find optimal hyperparameters either by BO or by starting from random initial values multiple times, using an optimizer every time
        and then returning the best result
        """
        if options["training"]["optimization method"] == "botorch":
            self.train()
            self.likelihood.train()
            fit_gpytorch_torch(gpt.mlls.ExactMarginalLogLikelihood(self.likelihood, self), options={"disp": False})
            return None
        # setup
        best_loss = 1e400
        optimal_parameters = dict()
        limits = hyperparameter_limits
        # start runs
        for iteration in range(options["training"]["restarts"]+1):
            # optimize and determine loss
            self.train_model()
            current_loss = self.get_current_loss()
            # check if the current run is better than previous runs
            if current_loss < best_loss:
                # if it is the best, save all used parameters
                best_loss = current_loss
                for param_name, param in self.named_parameters():
                    optimal_parameters[param_name] = copy.deepcopy(param)

            # set new random inital values
            self.likelihood.noise_covar.noise = torch.rand(1) * (limits["Noise"][1] - limits["Noise"][0]) + limits["Noise"][0]
            #self.mean_module.constant = torch.rand(1) * (limits["Mean"][1] - limits["Mean"][0]) + limits["Mean"][0]
            for kernel in get_kernels_in_kernel_expression(self.covar_module):
                hypers = limits[kernel._get_name()]
                for hyperparameter in hypers:
                    setattr(kernel, hyperparameter, torch.rand(1) * (hypers[hyperparameter][1] - hypers[hyperparameter][0]) + hypers[hyperparameter][0])

            # print output if enabled
            if options["training"]["print_optimizing_output"]:
                print(f"HYPERPARAMETER OPTIMIZATION: Random Restart {iteration}: loss: {current_loss}, optimal loss: {best_loss}")

        # finally, set the hyperparameters those in the optimal run
        self.initialize(**optimal_parameters)

    def eval_model(self):
        pass

    def plot_model(self, return_figure = False, figure = None, ax = None):
        self.eval()
        self.likelihood.eval()

        interval_length = torch.max(self.train_inputs[0]) - torch.min(self.train_inputs[0])
        shift = interval_length * options["plotting"]["border_ratio"]
        test_x = torch.linspace(torch.min(self.train_inputs[0]) - shift, torch.max(self.train_inputs[0]) + shift, options["plotting"]["sample_points"])

        with torch.no_grad(), gpt.settings.fast_pred_var():
            observed_pred = self.likelihood(self(test_x))

        with torch.no_grad():
            if not (figure and ax):
                figure, ax = plt.subplots(1, 1, figsize=(8, 6))


            lower, upper = observed_pred.confidence_region()
            ax.plot(self.train_inputs[0].numpy(), self.train_targets.numpy(), 'k.', zorder=2)
            ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b', zorder=3)
            amount_of_gradient_steps = 30
            alpha_min=0.05
            alpha_max=0.8
            alpha=(alpha_max-alpha_min)/amount_of_gradient_steps
            c = ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=alpha+alpha_min, zorder=1).get_facecolor()
            for i in range(1,amount_of_gradient_steps):
                ax.fill_between(test_x.numpy(), (lower+(i/amount_of_gradient_steps)*(upper-lower)).numpy(), (upper-(i/amount_of_gradient_steps)*(upper-lower)).numpy(), alpha=alpha, color=c, zorder=1)
            if options["plotting"]["legend"]:
                ax.plot([], [], 'k.', label="Data")
                ax.plot([], [], 'b', label="Mean")
                ax.plot([], [], color=c, label="Confidence")
                ax.legend()
        if not return_figure:
            plt.show()
        else:
            return figure, ax

    def plot_prior(self):
        self.train()
        self.likelihood.train()

        interval_length = torch.max(self.train_inputs[0]) - torch.min(self.train_inputs[0])
        shift = interval_length * options["plotting"]["border_ratio"]

        with torch.no_grad(), gpt.settings.fast_pred_var():
            observed_pred = self.likelihood(self(self.train_inputs[0]))

        with torch.no_grad():
            f, ax = plt.subplots(1, 1, figsize=(10, 8))

            lower, upper = observed_pred.confidence_region()
            ax.plot(self.train_inputs[0].numpy(), self.train_targets.numpy(), 'k.', label="Data")
            ax.plot(self.train_inputs[0].numpy(), observed_pred.mean.numpy(), 'b', label="Prediction")
            ax.fill_between(self.train_inputs[0].numpy().reshape((len(self.train_inputs[0]))), lower.numpy(), upper.numpy(), alpha=0.4, label="Confidence")
            if options["plotting"]["legend"]:
                ax.legend()
        plt.show()


class ExactStreamGPModel(ExactGPModel):
    """
    A subclass of ExactGPModel that tracks stream data in a window which can be easily adjusted.
    """

    def __init__(self, data, likelihood, kernel: gpt.kernels.Kernel, window_size):
        self.data = data
        self.window_size = window_size
        self.window = data[0:window_size]
        self.window_start_index = 0
        super(ExactStreamGPModel, self).__init__(self.window, likelihood, kernel)
        self.changepoint_indices = [] # save changepoints for plotting
        self.kernels = [] # save kernels for plotting
        self.likelihoods = [] # save likelihood objects for plotting
        self.likelihood_log = [] # save likelihood evaluations for model evaluation

    def step_window(self, stepsize = 1):
        assert stepsize < self.window_size, "stepsize has to be smaller than window size"
        self.window = self.window[stepsize:] + self.data[self.window_start_index + self.window_size:self.window_start_index + self.window_size + stepsize]
        self.window_start_index += stepsize
        self.set_train_data(self.window.X, self.window.Y)

    def plot_model(self, return_figure = False):
        figure, ax = plt.subplots(1, 1, figsize=(8,6))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        indices = [0] + self.changepoint_indices + [-1]
        for i in range(len(self.changepoint_indices)+1):
            segment_data = self.data[indices[i]:indices[i + 1]]
            self.set_train_data(segment_data.X, segment_data.Y, strict=False)
            self.covar_module = self.kernels[i]
            self.likelihood = self.likelihoods[i]
            figure, ax = super().plot_model(True, figure, ax)

        y_bot, y_top = plt.ylim()
        for cp in self.changepoint_indices:
            ax.vlines(self.data[cp].X, y_bot, y_top, colors="r", linestyles='dashed')

        ax.plot([], [], 'k.', label="Data")
        ax.plot([], [], 'b', label="Mean")
        ax.legend()

        self.set_train_data(self.window.X, self.window.Y, strict=False)
        if return_figure:
            return figure, ax
        else:
            plt.show()

    def process_data(self, stepsize : int = 1, steps_per_kernel : int = 100, update_kernel : str = "AKS", kernels : list = [], iterations = 3, max_complexity=6):
        """
        slide window over the entire data and save the likelihoods
        """
        self.predictions = []
        from kernelSearch import AKS, CKS
        # save current likelihood
        steps = 0
        self.kernels.append(copy.deepcopy(self.covar_module))
        self.likelihoods.append(copy.deepcopy(self.likelihood))
        while self.window_start_index + self.window_size < len(self.data):
            next_point = self.data[self.window_start_index + self.window_size]
            next_prediction = self.predict(next_point.X)
            self.predictions.append(next_prediction)

            new_loss = self.get_ll(self.window)
            self.likelihood_log.append(new_loss)
            # if loss in current window is significantly different from loss before, a changepoint is detected
            if steps >= steps_per_kernel:
                steps = 0
                # update model
                if update_kernel:
                    if update_kernel == "AKS":
                        self.covar_module, self.likelihood = AKS(self.covar_module, self.window, self.likelihood, kernels, iterations, max_complexity=max_complexity)
                    elif update_kernel == "CKS":
                        self.covar_module, self.likelihood = CKS(self.window, self.likelihood, kernels, iterations)
                else:
                    self.optimize_hyperparameters()
                self.changepoint_indices.append(int(self.window_start_index + self.window_size))
                self.kernels.append(copy.deepcopy(self.covar_module))
                self.likelihoods.append(copy.deepcopy(self.likelihood))
            self.step_window(stepsize=stepsize)
            steps += 1

    def process_data_cpd(self, update : str = "AKS", base_kernels : list = [], iterations : int = 3, max_complexity : int = 5, cpd_threshold : float = 5.0, cpd_tolerance = 0.5):
        from kernelSearch import AKS, CKS

        cpd_cusum_score = 0
        change_point_detected = False

        self.kernels.append(copy.deepcopy(self.covar_module))
        self.likelihoods.append(copy.deepcopy(self.likelihood))
        self.predictions = []

        # detect the next changepoint, save likelihoods in the meantime
        while self.window_start_index + self.window_size < len(self.data):
            current_loss = self.get_ll(self.window)
            self.likelihood_log.append(current_loss)

            next_point = self.data[self.window_start_index + self.window_size]
            next_prediction = self.predict(next_point.X)
            self.predictions.append(next_prediction)

            # CPD
            # CUSUM
            cpd_cusum_score = max(0, cpd_cusum_score + abs(next_prediction-next_point.Y) - cpd_tolerance*2*torch.sqrt(self.likelihood.noise).item())
            if cpd_cusum_score > cpd_threshold:
                change_point_detected = True

            # potentially update model
            if change_point_detected:
                if update == "AKS":
                    self.covar_module, self.likelihood = AKS(self.covar_module, self.window, self.likelihood, base_kernels, iterations, max_complexity)
                elif update == "CKS":
                    self.covar_module, self.likelihood = CKS(self.window, self.likelihood, base_kernels, iterations)
                elif update == "HPO":
                    self.optimize_hyperparameters()
                else:
                    raise ValueError("Unknown update method")
                change_point_detected = False
                cpd_cusum_score = 0

                self.changepoint_indices.append(int(self.window_start_index + self.window_size))
                self.kernels.append(copy.deepcopy(self.covar_module))
                self.likelihoods.append(copy.deepcopy(self.likelihood))
            # step
            self.step_window()
