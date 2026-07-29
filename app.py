import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn import set_config
from sklearn.pipeline import Pipeline

from scripts.data_clean_utils import perform_data_cleaning


set_config(transform_output="pandas")


PREPROCESSOR_PATH = Path(
    os.getenv("PREPROCESSOR_PATH", "models/preprocessor.joblib")
)

PORT = int(os.getenv("PORT", "8000"))

model_pipeline: Pipeline | None = None


class DeliveryData(BaseModel):
    ID: str
    Delivery_person_ID: str
    Delivery_person_Age: str
    Delivery_person_Ratings: str
    Restaurant_latitude: float
    Restaurant_longitude: float
    Delivery_location_latitude: float
    Delivery_location_longitude: float
    Order_Date: str
    Time_Orderd: str
    Time_Order_picked: str
    Weatherconditions: str
    Road_traffic_density: str
    Vehicle_condition: int = Field(ge=0)
    Type_of_order: str
    Type_of_vehicle: str
    multiple_deliveries: str
    Festival: str
    City: str


class PredictionResponse(BaseModel):
    predicted_delivery_time_minutes: float


def initialize_model_pipeline() -> Pipeline:
    """Load the registered MLflow model and local preprocessing object."""

    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessor not found at: {PREPROCESSOR_PATH.resolve()}"
        )

    dagshub.init(
        repo_owner=DAGSHUB_REPO_OWNER,
        repo_name=DAGSHUB_REPO_NAME,
        mlflow=True,
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
    regression_model = mlflow.sklearn.load_model(model_uri)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("regressor", regression_model),
        ]
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_pipeline

    try:
        model_pipeline = initialize_model_pipeline()
        app.state.model_ready = True
        app.state.startup_error = None
    except Exception as exc:
        model_pipeline = None
        app.state.model_ready = False
        app.state.startup_error = str(exc)

    yield

    model_pipeline = None


app = FastAPI(
    title="Food Delivery Time Prediction API",
    description=(
        "Predicts food delivery time using rider, order, traffic, weather, "
        "location, and delivery-context information."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def home() -> dict[str, str]:
    return {
        "message": "Food Delivery Time Prediction API",
        "documentation": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    ready = bool(getattr(app.state, "model_ready", False))

    response: dict[str, Any] = {
        "status": "healthy" if ready else "unhealthy",
        "model_ready": ready,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
    }

    startup_error = getattr(app.state, "startup_error", None)
    if startup_error:
        response["error"] = startup_error

    return response


@app.post("/predict", response_model=PredictionResponse)
def predict_delivery_time(data: DeliveryData) -> PredictionResponse:
    if model_pipeline is None:
        startup_error = getattr(
            app.state,
            "startup_error",
            "The model pipeline is not available.",
        )
        raise HTTPException(
            status_code=503,
            detail=f"Prediction service is not ready: {startup_error}",
        )

    try:
        raw_data = pd.DataFrame([data.model_dump()])
        cleaned_data = perform_data_cleaning(raw_data)
        prediction = float(model_pipeline.predict(cleaned_data)[0])

        return PredictionResponse(
            predicted_delivery_time_minutes=round(prediction, 2)
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=PORT,
        reload=False,
    )
