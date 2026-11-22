@asynccontextmanager
async def lifespan(app: FastAPI):
  global predictor
  artifact_dir = Path("artifacts")
  if not (artifact_dir / "kidney_net_checkpoint.pt").exists():
    csv_path = Path("data/kidney_dataset.csv")
    if not csv_path.exists():
      # Generate synthetic fallback data in CI if raw CSV is not checked into Git
      import numpy as np
      import pandas as pd

      csv_path.parent.mkdir(parents=True, exist_ok=True)
      rng = np.random.default_rng(42)
      pd.DataFrame({
          "Creatinine": rng.uniform(0.6, 4.5, size=80),
          "BUN": rng.uniform(8, 60, size=80),
          "Urine_Output": rng.uniform(400, 2000, size=80),
          "Diabetes": rng.choice(list((0, 1)), size=80),
          "Hypertension": rng.choice(list((0, 1)), size=80),
          "Age": rng.integers(25, 80, size=80),
          "Protein_in_Urine": rng.uniform(0.0, 3.5, size=80),
          "Water_Intake": rng.uniform(0.5, 3.5, size=80),
          "Medication": rng.choice(
              list(("None", "ACE Inhibitor", "ARB")), size=80
          ),
          "CKD_Status": rng.choice(list((0, 1)), size=80),
      }).to_csv(csv_path, index=False)
    train_and_export(csv_path=str(csv_path), artifact_dir=str(artifact_dir))
  predictor = KidneyPredictor(artifact_dir=artifact_dir)
  yield
