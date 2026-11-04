from pathlib import Path
from fastapi import FastAPI
import joblib
from pydantic import BaseModel, Field

MODEL_PATH = Path("models/iris_model.pkl")
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Iris Classifier API", description="API for predicting iris species using a trained Random Forest model.")

class IrisInput(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)

@app.get("/health")
def health_check():
    return {"status": "healthy"}
@app.post("/predict")
def predict_species(data: IrisInput):
    features=[[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    prediction=int(model.predict(features)[0])
    return {"prediction": prediction,
            "class_name": ["setosa", "versicolor", "virginica"][prediction]}