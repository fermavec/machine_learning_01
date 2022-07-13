#Libraries 
import pandas as pd
#Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


#Entry Point
if __name__ == "__main__":
    dataset = pd.read_csv('./Data/heart.csv')
    print(dataset.info())