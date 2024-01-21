import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML


class Moons:
    def __init__(self, database_path:str, table:str):
        database_service = "sqlite"
        connectable = f"{database_service}:///{database_path}"
        query = f"SELECT * FROM {table}"
        self.database = pd.read_sql(query, connectable)

    def display_df(self):
        display(self.database)

    def display_data_info(self):
        print(f"Number of fields: {len(self.database.columns)}")
        print(f"Number of records: {len(self.database)}")
        print(f"Column Names: {list(self.database.columns)}")

    def check_missing_values(self):
        return self.database.isnull().sum()

    def summary(self):
        return self.database.describe()

    def corr(self):
        return self.database.corr(numeric_only = True)

    def heatmap(self, square = True, vmin = -1, vmax = 1, cmap = "RdBu"):
        sns.heatmap(data = self.corr(), square = square, vmin = vmin, vmax = vmax, cmap = cmap)
        plt.title("Correlation Heatmap")
        plt.show()

    def return_moon_data(self, name):
        if name not in list(self.database["moon"]):
            print("This is not a moon in the database")
            return
        else:
            return self.database[self.database["moon"] == name]

    def return_corr(self, col_1, col_2):
        if col_1 not in list(self.database.columns) or col_2 not in list(self.database.columns):
            print("Enter a valid column name in the database")
            return
        else:
            return self.database[col_1].corr(self.database[col_2])

    def plot(self, col_1, col_2):
        if pd.api.types.infer_dtype(self.database[col_1]) == "string" or pd.api.types.infer_dtype(
                self.database[col_2]) == "string":
            print("Only integer/float based columns can be plotted")
        else:
            self.database.plot(col_1, col_2, xlabel=col_1, ylabel=col_2, title=f"{col_2}, against {col_1}")

    def scatter(self, col_1, col_2):
        if pd.api.types.infer_dtype(self.database[col_1]) == "string" or pd.api.types.infer_dtype(self.database[col_2]) == "string":
            print("Only integer/float based columns can be plotted")
        else:
            self.database.plot.scatter(col_1, col_2, xlabel = col_1, ylabel = col_2, title = f"{col_2}, against {col_1}")


