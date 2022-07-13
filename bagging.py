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
    #print(dataset.info())

    #Features and target
    X = dataset.drop(['target'], axis=1)
    y = dataset['target']
    #print(y.head(5))

    #Train and Test division
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=42)

    #Implementation
    kn_class = KNeighborsClassifier().fit(X_train, y_train)
    kn_predict = kn_class.predict(X_test)
    print("="*32)
    print(accuracy_score(kn_predict, y_test))

    bag_class = BaggingClassifier(base_estimator=KNeighborsClassifier(), n_estimators=50).fit(X_train, y_train)
    bag_predict = bag_class.predict(X_test)
    print("="*32)
    print(accuracy_score(bag_predict, y_test))

    #Challenge: Try another classifier (not KN)