#Libraries
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

#Modules
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.linear_model import LogisticRegression #Classifier
from sklearn.preprocessing import StandardScaler #Normalizer
from sklearn.model_selection import train_test_split #Data Splitter


#Entry Point
if __name__ == "__main__":
    data_heart = "./Data/heart.csv"
    df_heart = pd.read_csv(data_heart)

    print(df_heart.head(5))