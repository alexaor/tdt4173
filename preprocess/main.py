import numpy as np
import pandas as pd
from tools import impute_data
from tools import create_reduced_set

""" CHECKLIST:
- reduced set: check


"""


def create_subset():
    rows = np.r_[:10]
    columns = np.r_[:20, -4:0]
    create_reduced_set(rows, columns, "reduced.csv")


def main():
    create_subset()
    dataset = pd.read_csv('reduced.csv')
    X = dataset.iloc[:, :].values
    imputed = impute_data(X)

    df = pd.DataFrame(imputed)
    df.to_csv('imputed.csv')
    

if __name__ == "__main__":
    main()

