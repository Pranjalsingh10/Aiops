from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import joblib
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
import numpy as np
import os

# CONFIGURE MLFLOW TRACKING
mlflow.set_tracking_uri("http://127.0.0.1:6001")
mlflow.set_experiment("Iris_Classification_Experiment")


iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

n_estimators = 10
random_state = 42

model_parameters = {
    "model_type": "RandomForestClassifier",
    "n_estimators": n_estimators,
    "random_state": random_state,
    "test_size": 0.2
}


# START MLFLOW RUN
with mlflow.start_run() as run:

    # CAPTURE CURRENT RUN ID
    current_run_id = run.info.run_id

    # LOG PARAMETERS
    mlflow.log_params(model_parameters)

    # BUILD AND TRAIN CLASSIFIER
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state
    )

    clf.fit(X_train, y_train)

    # PREDICTIONS
    y_pred = clf.predict(X_test)

    # EVALUATION
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, average="macro", zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, average="macro", zero_division=0
    )

    print(f"Model accuracy: {accuracy}")
    print(f"Model precision: {precision}")
    print(f"Model recall: {recall}")

    # LOG METRICS
    mlflow.log_metrics({
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall
    })

    # MODEL QUALITY GATE
    MIN_TEST_ACCURACY = 0.9

    if accuracy < MIN_TEST_ACCURACY:
        raise ValueError(
            f"Model rejected because accuracy {accuracy} "
            f"is below {MIN_TEST_ACCURACY}."
        )

    print("Model passed the quality gate.")

    # SAVE MODEL LOCALLY
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "iris_model.pkl"
    joblib.dump(clf, model_path)

    print(f"Model saved to {model_path}")

    # REGISTER MODEL WITH MLFLOW
    mlflow.sklearn.log_model(
        sk_model=clf,
        artifact_path="iris_model",
        registered_model_name="Iris_Classifier_Model",
        skops_trusted_types=[
        "sklearn.tree._tree.Tree"
    ]
    )

    # MODEL REGISTRY
    client = MlflowClient()

    registered_versions = client.search_model_versions(
        "name='Iris_Classifier_Model'"
    )

    # FIND VERSION CREATED BY CURRENT RUN
    current_model_version = next(
        (
            version
            for version in registered_versions
            if version.run_id == current_run_id
        ),
        None
    )

    if current_model_version is None:
        raise ValueError(
            "No registered model version found for the current run."
        )

    # ASSIGN CHAMPION ALIAS
    client.set_registered_model_alias(
        name="Iris_Classifier_Model",
        alias="champion",
        version=current_model_version.version
    )

    print(
        f"Registered Model: Iris_Classifier_Model, "
        f"Version: {current_model_version.version}"
    )

    print("Champion alias assigned successfully.")