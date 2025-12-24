import mlflow
import mlflow.sklearn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ====== SET TRACKING ======
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("training-model")

# ====== LOAD DATA ======
df = pd.read_csv("heart_disease_clean.csv") 

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ====== TRAIN ======
with mlflow.start_run():
    mlflow.sklearn.autolog()

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy", acc)

    print("Accuracy:", acc)