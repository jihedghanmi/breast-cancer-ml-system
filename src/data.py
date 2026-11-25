import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

def load_data():
    data = load_breast_cancer(as_frame=True)
    X = data.data.copy()
    y = (data.target == 0).astype(int) #1 = Malignant, 0 = Benign
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(X_train.shape)
    print(X_test.shape)
    print(y_test.mean())
    print(y_train.mean())

    