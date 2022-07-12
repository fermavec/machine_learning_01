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

    
    #Splitting Dataset
    df_features = df_heart.drop(['target'], axis=1)
    df_target = df_heart['target']

    #Normalizing Features
    df_features = StandardScaler().fit_transform(df_features)

    #Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(df_features, df_target, test_size=0.3, random_state=42)