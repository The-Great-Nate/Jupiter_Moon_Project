import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, HTML

class Moons:
    def __init__(self, database_path:str, table:str):
        database_service = "sqlite"
        connectable = f"{database_service}:///{database_path}"
        query = f"SELECT * FROM {table}"
        self.database = pd.read_sql(query, connectable)

    def display_df(self):
        display(self.database)

    def describe(self):
        return self.database.describe()

