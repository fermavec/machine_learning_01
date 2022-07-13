#Libraries
import pandas as pd
#Models
from sklearn.linear_model import RANSACRegressor, HuberRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


#Emtry Point
if __name__ == "__main__":
    dataset = pd.read_csv("./Data/felicidad_corrupt.csv")

    print(dataset.info())