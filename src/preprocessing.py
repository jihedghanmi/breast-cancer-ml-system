from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def build_preprocessor(feature_names):
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_pipeline, feature_names)]
    )

    return preprocessor

if __name__ == "__main__":
    from src.data import load_data, split_data
        # 1. Load and split the dataset
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    preprocessor = build_preprocessor(X_train.columns.tolist())

    X_train_scaled = preprocessor.fit_transform(X_train)

    X_test_scaled = preprocessor.transform(X_test)

    print(f"X_train_scaled shape: {X_train_scaled.shape}")
    print(f"X_test_scaled shape:  {X_test_scaled.shape}")
    print(f"X_train_scaled mean:  {X_train_scaled.mean():.6f}")
    print(f"X_test_scaled mean:   {X_test_scaled.mean():.6f}") 
