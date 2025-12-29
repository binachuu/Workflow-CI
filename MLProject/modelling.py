import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

EXPERIMENT_NAME = "training-model"
DATA_PATH = "heart_disease_clean.csv"
TARGET_COL = "num"

def main():
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.autolog()

    df = pd.read_csv(DATA_PATH)

    df[TARGET_COL] = (df[TARGET_COL] > 0).astype(int)

    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    acc_lr = accuracy_score(y_test, lr.predict(X_test))
    print("Accuracy LR:", acc_lr)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    acc_rf = accuracy_score(y_test, rf.predict(X_test))
    print("Accuracy RF:", acc_rf)

if __name__ == "__main__":
    main()