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
    #print(df_heart.head(5))
    
    #Splitting Dataset
    df_features = df_heart.drop(['target'], axis=1)
    df_target = df_heart['target']

    #Normalizing Features
    df_features = StandardScaler().fit_transform(df_features)

    #Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(df_features, df_target, test_size=0.3, random_state=42)
    #print(X_train.shape)
    #print(y_train.shape)

    #Configuring PCA Algorythm
    #Default n_components = min(number samples / number of features)
    pca = PCA(n_components=3)
    pca.fit(X_train)

    #Configuring IPCA for Low Resources Computing
    #Batch size = number of blocks to compute
    ipca = IncrementalPCA(n_components=3, batch_size=10)
    ipca.fit(X_train)

    #Comparision "r" Variance
    #plt.plot(range(len(pca.explained_variance_)), pca.explained_variance_ratio_)
    #plt.show()

    #Generating Logistic Regresion
    logistic = LogisticRegression(solver="lbfgs")

    #Applying PCA
    dt_train = pca.transform(X_train)
    dt_test = pca.transform(X_test)
    logistic.fit(dt_train, y_train)

    #Accuracy metric
    print("SCORE PCA: ", logistic.score(dt_test, y_test))

    #Applying IPCA
    dt_train = ipca.transform(X_train)
    dt_test = ipca.transform(X_test)
    logistic.fit(dt_train, y_train)

    #Accuracy metric
    print("SCORE IPCA: ", logistic.score(dt_test, y_test))