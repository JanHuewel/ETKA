import pandas as pd
import torch
import numpy as np
from scipy import io

class Data:
    """
    A class to represent data for Gaussian Processes.
    This data consists of input ("X"), targets ("Y") and noise ("sigma")
    """
    def __init__(self, X=None, Y=None, file=None, frame=None, X_key=None, Y_key=None, *args, **kwargs):
        """
        Contructor of the Data class. Takes an input and saves its data in the Data object
        The data can be provided in multiple forms, including a pandas DataFrame and separate keyword arguments.

        Possibility 1: Via X, Y, (Sigma)
        Possibility 2: Via a Dataframe and column names für X, Y, (Sigma)
        """
        # possibility 1: give all data via X and Y
        if (X is not None) and (Y is not None):
            if isinstance(X, torch.Tensor):
                self.X = X.clone().detach()
            else:
                self.X = torch.tensor(X, dtype=torch.float64)
            if isinstance(Y, torch.Tensor):
                self.Y = Y.clone().detach()
            else:
                self.Y = torch.tensor(Y, dtype=torch.float64)
        if "Sigma" in kwargs:
            self.sigma = kwargs["Sigma"]
        elif "sigma" in kwargs:
            self.sigma = kwargs["sigma"]

        # possibility 2: give a DataFrame containing all data
        # if a DataFrame is given, check the labels given in X_key and Y_key
        if frame is not None:
            if isinstance(X_key, list):
                pass  # multi input is not defined yet
            else:
                self.X = torch.tensor(frame[X_key])
            if isinstance(Y_key, list):
                pass # multi output is not defined yet
            else:
                self.Y = torch.tensor(args[0][Y_key])

        # possibility 3: give file and keys
        # multi input might need correction
        if file is not None:
            if isinstance(file, str):
                if file[-4:] == ".csv":
                    f = pd.read_csv(file).dropna().reset_index(drop=True)
                    if X_key is not None:
                        if X_key == "__index__" and "__index__" not in f.keys():
                            self.X=torch.tensor(f.index, dtype=torch.float64)
                        else:
                            self.X = torch.tensor(f[X_key], dtype=torch.float64)
                    if Y_key is not None:
                        self.Y = torch.tensor(f[Y_key], dtype=torch.float64)
                elif file[-4:] == ".mat":
                    f = io.loadmat(file)
                    if X_key is not None:
                        self.X = torch.tensor(f[X_key].reshape((len(f[X_key]),)), dtype=torch.float64)
                    if Y_key is not None:
                        self.Y = torch.tensor(f[Y_key].reshape((len(f[Y_key]),)), dtype=torch.float64)
                else:
                    raise ValueError("Unknown file extension")
            else:
                raise ValueError("File has to be provided as a string of the file name")

        if not hasattr(self, 'sigma'):
            self.sigma = None

    def time_to_X(self, file=None, frame=None, label_year=None, label_month=None, label_day=None, label_hour=None, label_minute=None, label_second=None):
        if frame == None and isinstance(file, str):
            if file[-4:] == ".csv":
                self.time_to_X(None, pd.read_csv(file), label_year, label_month, label_day, label_hour, label_minute, label_second)
            elif file[-4:] == ".mat":
                raise ValueError("Time conversion from mat files not yet implemented")
            else:
                raise ValueError("Unkown file extension")
        elif isinstance(frame, pd.DataFrame):
            #raise NotImplementedError("time_to_X not fully implemented!")
            time_ticks = pd.to_datetime(frame[label_year, label_month, label_day, label_hour, label_minute, label_second]).apply(lambda x: x.timestamp())
            self.X = torch.tensor(time_ticks, dtype=torch.float64)

    def normalize_z(self):
        self.X -= self.X.mean()
        self.X /= self.X.std(0)
        self.Y -= self.Y.mean()
        self.Y /= self.Y.std(0)

    def normalize_0_1(self):
        self.X -= self.X.min()
        self.X /= self.X.max()
        self.Y -= self.Y.min()
        self.Y /= self.Y.max()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        if isinstance(item, int):
            if item != -1:
                return Data(X=self.X[item:item+1], Y=self.Y[item:item+1], sigma=self.sigma)
            else:
                return Data(X=self.X[item:], Y=self.Y[item:], sigma=self.sigma)
        return Data(X=self.X.__getitem__(item), Y=self.Y.__getitem__(item), sigma=self.sigma)

    def __add__(self, other):
        if isinstance(other, Data):
            x = torch.cat((self.X, other.X), 0)
            y = torch.cat((self.Y, other.Y), 0)
            return Data(X=x, Y=y, sigma=self.sigma)

    def __iter__(self):
        return iter([Data(X=self[i].X, Y=self[i].Y) for i in range(len(self))])