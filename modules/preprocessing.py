import numpy as np
import pandas as pd


class MinmaxScaler:
    def __init__(self):
        self._max = None
        self._min = None
        self._columns = None

    def fit(self, data):
        self._max = np.max(data, axis=0)
        self._min = np.min(data, axis=0)

        if isinstance(data, pd.core.frame.DataFrame):
            self._columns = {name: i for i, name in enumerate(data.columns)}
        else:
            self._columns = f'This data is not a dataframe. {data.shape}'

        scale = (data - self._min) / (self._max - self._min)

        return scale

    def transform(self, data):
        scale = (data - self._min) / (self._max - self._min)

        return scale

    def inverse(self, data, categories=None):
        if categories is not None:
            inv = data * (self._max[categories] - self._min[categories]) + self._min[categories]
        else:
            if not isinstance(data, pd.core.frame.DataFrame):
                inv = data * (self._max.values - self._min.values) + self._min.values
            else:
                inv = data * (self._max - self._min) + self._min

        return inv

    def get_columns(self):
        return self._columns


class StandardScaler:
    def __init__(self):
        self._std = None
        self._mean = None
        self._columns = None

    def fit(self, data):
        self._mean = np.mean(data, axis=0)
        self._std = np.std(data, axis=0)

        if isinstance(data, pd.core.frame.DataFrame):
            self._columns = {name: i for i, name in enumerate(data.columns)}
        else:
            self._columns = f'This data is not a dataframe. {data.shape}'

        scale = (data - self._mean) / self._std

        return scale

    def transform(self, data):
        scale = (data - self._mean) / self._std

        return scale

    def inverse(self, data, categories=None):
        if categories is not None:
            inv = data * self._std[categories] + self._mean[categories]
        else:
            if not isinstance(data, pd.core.frame.DataFrame):
                inv = data * self._std.values + self._mean.values
            else:
                inv = data * self._std + self._mean

        return inv

    def get_columns(self):
        return self._columns
