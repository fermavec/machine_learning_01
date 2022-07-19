from matplotlib.pyplot import axis
import pandas as pd

class Utils():
    def load_csv(self, path):
        return pd.read_csv(path)


    def load_mysql(self):
        pass


    def features_target(self, data, drop_columns, target):
        X = data.drop(drop_columns, axis=1)
        y = data[target]

        return X, y

    
    def model_export(self, clf, score):
        pass