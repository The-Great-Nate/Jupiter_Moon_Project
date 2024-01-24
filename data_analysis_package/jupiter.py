import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


class Moons:
    def __init__(self, database_path:str, table:str):
        database_service = "sqlite"
        connectable = f"{database_service}:///{database_path}"
        query = f"SELECT * FROM {table}"
        self.__database = pd.read_sql(query, connectable)

    def get_df(self):
        return self.__database

    def set_df(self, df):
        self.__database = df

    def display_data_info(self):
        print(f"Number of fields: {len(self.__database.columns)}")
        print(f"Number of records: {len(self.__database)}")
        print(f"Column Names: {list(self.__database.columns)}")

    def return_col_names(self):
        return list(self.__database.columns)

    def check_missing_values(self):
        return self.__database.isnull().sum()

    def return_complete_records(self):
        return self.__database.dropna()

    def summary(self, df=None):
        if df is None:
            df = self.__database
        else:
            pass
        return df.describe()

    def corr(self):
        return self.__database.corr(numeric_only = True)

    def heatmap(self, square = True, vmin = -1, vmax = 1, cmap = "RdBu"):
        sns.heatmap(data = self.corr(), square = square, vmin = vmin, vmax = vmax, cmap = cmap)
        plt.title("Correlation Heatmap")
        plt.show()

    def return_moon_data(self, name):
        if name not in list(self.__database["moon"]):
            print("This is not a moon in the database")
            return
        else:
            return self.__database[self.__database["moon"] == name]

    def return_corr(self, col_1, col_2):
        if col_1 not in list(self.__database.columns) or col_2 not in list(self.__database.columns):
            print("Enter a valid column name in the database")
            return
        else:
            return self.__database[col_1].corr(self.__database[col_2])

    def plot_relationship(self, col_1, col_2, hue = None):
        if col_1 not in list(self.__database.columns) or col_2 not in list(self.__database.columns):
            print("Enter a valid column name in the database")
            return

        filter_nan = self.__database.dropna(subset=[col_1, col_2])
        if pd.to_numeric(filter_nan[col_1], errors = "coerce").notna().all() == False and pd.to_numeric(filter_nan[col_2], errors = "coerce").notna().all() == False:
            print("Comparing 2 categorical variables")
            sns.countplot(data=self.__database, x=col_1, hue=col_2)
        elif pd.to_numeric(filter_nan[col_1], errors = "coerce").notna().all() == False or pd.to_numeric(filter_nan[col_2], errors = "coerce").notna().all() == False:
            sns.catplot(data=self.__database, x=col_1, y=col_2, kind="box", aspect=1.5, hue = hue)
        else:
            plot = sns.relplot(data=self.__database, x=col_1, y=col_2, hue = hue)
            plot.fig.suptitle(f"{col_1} x {col_2}", fontsize=16)
            plot.fig.subplots_adjust(top=0.9)

    def pairwise_plots(self, hue = None):
        sns.pairplot(self.__database, hue = hue)

    def filter_for_characteristic(self, column, characteristic):
        if column not in list(self.__database.columns):
            print("This is not a column in the database")
            return
        else:
            return self.__database[self.__database[column] == characteristic]

    def return_unique_values(self, col):
        return self.__database[col].unique()

    @staticmethod
    def return_central_mass(gradient):
        return (4*np.pi**2)/(6.67E-11 * gradient)

    def get_model(self, col_1, col_2):
        model = linear_model.LinearRegression(fit_intercept = False)
        x = self.__database[[col_1]]
        y = self.__database[col_2]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
        model.fit(x_train, y_train)
        return model, x_test, y_test, col_1, col_2

    @staticmethod
    def get_model_parameters(model):
        if not isinstance(model[0], linear_model.LinearRegression):
            return "This is not a LinearRegression object"
        model_obj = model[0]
        return model_obj.coef_[0], model_obj.intercept_

    @staticmethod
    def predict(model:tuple):
        if not isinstance(model[0], linear_model.LinearRegression):
            return "This is not a LinearRegression object"
        pred = model[0].predict(model[1])
        return pred

    @staticmethod
    def plot_prediction(model:tuple, pred):
        if not isinstance(model[0], linear_model.LinearRegression):
            return "This is not a LinearRegression object"
        reg_model = model[0]
        data_for_predict = model[1]
        y_col = model[2]
        if not isinstance(reg_model, linear_model.LinearRegression):
            return "This is not a LinearRegression object"
        else:
            plt.scatter(data_for_predict, y_col, label = f"{model[3]} test data", marker = "x")
            plt.plot(data_for_predict, pred, label = f"Prediction", color = "orange")
            plt.title(f"{model[3]} & prediction x {model[3]}")
            plt.xlabel(model[3])
            plt.ylabel(model[4])
            plt.legend()

    @staticmethod
    def plot_residuals(model:tuple, pred):
        plt.plot(model[1], model[2] - pred, '.')
        # Add a horizontal line at zero to guide the eye
        plt.axhline(0, color='k', linestyle='dashed')
        # Add axis labels
        plt.xlabel(model[3])
        plt.ylabel("Residuals")

    @staticmethod
    def output_model_worth(model:tuple, pred):
        r2 = r2_score(model[2], pred)
        rmse = mean_squared_error(model[2], pred, squared = False)
        print(f"r2_score: {r2}")
        print(f"root mean squared error: {rmse}")
        return r2, rmse
