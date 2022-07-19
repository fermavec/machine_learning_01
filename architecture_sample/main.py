from ctypes import util
from utils import Utils


#Entry point
if __name__ == "__main__":
    utils = Utils()

    df = utils.load_csv('./in/felicidad_corrupt.csv')
    print(df.head(5))