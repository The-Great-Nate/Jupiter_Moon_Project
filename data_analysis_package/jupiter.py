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

    def plot_relationship(self, col_1, col_2):
        if col_1 not in list(self.database.columns) or col_2 not in list(self.database.columns):
            print("Enter a valid column name in the database")
            return
        if pd.to_numeric(self.database[col_1], errors = "coerce").notna().all() == False and pd.to_numeric(self.database[col_2], errors = "coerce").notna().all() == False:
            print("Comparing 2 categorical variables")
            sns.countplot(data=self.database, x=col_1, hue=col_2)
        elif pd.to_numeric(self.database[col_1], errors = "coerce").notna().all() == False or pd.to_numeric(self.database[col_2], errors = "coerce").notna().all() == False:
            sns.catplot(data=self.database, x=col_1, y=col_2, kind="box", aspect=1.5)
        else:
            plot = sns.relplot(data=self.database, x=col_1, y=col_2)
            plot.fig.suptitle(f"{col_1} x {col_2}", fontsize=16)
            plot.fig.subplots_adjust(top=0.9)
