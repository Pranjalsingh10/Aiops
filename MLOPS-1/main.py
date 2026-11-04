import os
from pathlib import Path
from fastapi import FastAPI
import joblib
from pydantic import BaseModel, Field
import mlflow
import mlflow.sklearn
from contextlib import asynccontextmanager
from typing import Callable
app = FastAPI(title="Iris Model API")
#ML Flow Configuration
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", "http://127.0.0.1:6001"

)
MODEL_URI="models:/iris-classifier@champion"
class IrisInput(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)

def load_mlflow_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model=mlflow.sklearn.load_model(MODEL_URI)
    return model

def create_app(model_loader: Callable=load_mlflow_model,
    version_loader: Callable=None):
    if(version_loader is None):
        def version_loader():
            client=mlflow.MLflowClient()
            champion_version=client.get_model_version_by_alias(
                    name="iris-classifier",
                    alias="champion"
            )
            return champion_version.version
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            app.state.model=model_loader()
        except Exception as ex:
            raise RuntimeError(f"Failed to load model from MLflow: {ex}")
        app.state.model_version=version_loader()
        yield
    app=FastAPI(
        title="Iris Classifier API",
        description="API for predicting iris species using a trained Random Forest model.",
        version="3.0.0",
        lifespan=lifespan
    )

    @app.get("/health")
    def health_check():
        return {
            "status": "healthy",
            "model_version": app.state.model_version
            }
    @app.post("/predict")
    def predict_species(data: IrisInput):
        model=app.state.model
        features=[[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
        prediction=int(model.predict(features)[0])
        return {"prediction": prediction,
                "class_name": ["setosa", "versicolor", "virginica"][prediction]}

    return app

app=create_app()


# MODEL_PATH = Path("models/iris_model.joblib")
# # Load the trained model
# model = joblib.load(MODEL_PATH)
#try:
    
#except Exception as ex:
#    raise RuntimeError(f"Failed to load model from MLflow: {ex}")

#app = FastAPI(title="Iris Classifier API", description="API for predicting iris species using a trained Random Forest model.")