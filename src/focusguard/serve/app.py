from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from focusguard.serve.model_loader import load_model, predict_one


class PredictRequest(BaseModel):
    # Derived-only features
    face_present: int = Field(ge=0, le=1)
    nose_offset_abs: float
    nose_offset_signed: float


class PredictResponse(BaseModel):
    prediction: str
    proba: Optional[Dict[str, float]] = None


class ConfigResponse(BaseModel):
    model_path: str
    feature_cols: list[str]


def create_app(model_path: str = "models/focusguard_logreg.joblib") -> FastAPI:
    app = FastAPI(title="FocusGuard", version="0.1.0")

    loaded = load_model(model_path)

    @app.get("/")
    def root() -> Dict[str, Any]:
        return {
            "name": "FocusGuard",
            "status": "ok",
            "endpoints": ["/health", "/config", "/predict"],
        }


    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {"status": "ok"}

    @app.get("/config", response_model=ConfigResponse)
    def config() -> ConfigResponse:
        return ConfigResponse(model_path=str(Path(model_path)), feature_cols=loaded.feature_cols)

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        out = predict_one(
            loaded,
            {
                "face_present": req.face_present,
                "nose_offset_abs": req.nose_offset_abs,
                "nose_offset_signed": req.nose_offset_signed,
            },
        )
        return PredictResponse(prediction=out["prediction"], proba=out.get("proba"))

    return app




app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("focusguard.serve.app:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
