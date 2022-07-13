#Libraries
import matplotlib.pyplot as plt
import pandas as pd
#Models
from sklearn.linear_model import RANSACRegressor, HuberRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


#Emtry Point
if __name__ == "__main__":
    dataset = pd.read_csv("./Data/felicidad_corrupt.csv")

    #print(dataset.info())

    #Features & Target
    X = dataset.drop(['country', 'score'], axis=1)
    y = dataset['score']
    #print(y.head(5))

    #Data partition
    X_train, X_test,  y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    #Estimators
    estimators = {
        'SVR': SVR(gamma='auto', C=1.0, epsilon=0.1),
        'RANSAC': RANSACRegressor(), #It is a metaestimator. Linear Regresion by default
        'HUBER': HuberRegressor(epsilon=1.35) #1.35 by default >+atypicaldata <-atypicaldata
    }

    #Working with the estimators
    for name, estimator in estimators.items():
        estimator.fit(X_train, y_train)
        prediction = estimator.predict(X_test)

        print('='*32)
        print(name)
        print('MSE', mean_squared_error(y_test, prediction))

        plt.ylabel('Predicted Score')
        plt.xlabel('Real Score')
        plt.title(name)
        plt.scatter(y_test, prediction)
        plt.plot(prediction, prediction,'r--')
        plt.show()