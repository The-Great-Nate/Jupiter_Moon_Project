from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


class DataAnalysis(ABC):

    @abstractmethod
    def get_df(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def set_df(self) -> None:
        pass

    @abstractmethod
    def display_data_info(self) -> None:
        pass