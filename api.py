from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from src.predict import KidneyPredictor
from src.train import train_and_export

predictor: KidneyPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    artifact_dir = Path("artifacts")
    if not (artifact_dir / "kidney_net_checkpoint.pt").exists():
        train_and_export(csv_path="data/kidney_dataset.csv", artifact_dir=str(artifact_dir))
    predictor = KidneyPredictor(artifact_dir=artifact_dir)
    yield


app = FastAPI(
    title="Kidney Disease Risk Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)


class PatientRecord(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Creatinine": 2.9,
                "BUN": 42.0,
                "Urine_Output": 850.0,
                "Diabetes": 1,
                "Hypertension": 1,
                "Age": 62.0,
                "Protein_in_Urine": 1.8,
                "Water_Intake": 1.5,
                "Medication": "None",
            }
        }
    )

    Creatinine: float = Field(...)
    BUN: float = Field(...)
    Urine_Output: float = Field(...)
    Diabetes: int = Field(..., ge=0, le=1)
    Hypertension: int = Field(..., ge=0, le=1)
    Age: float = Field(..., ge=0, le=120)
    Protein_in_Urine: float = Field(...)
    Water_Intake: float = Field(...)
    Medication: str = Field(...)


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": predictor is not None}


@app.post("/predict")
def predict_risk(patient: PatientRecord):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    return predictor.predict_one(patient.model_dump())
