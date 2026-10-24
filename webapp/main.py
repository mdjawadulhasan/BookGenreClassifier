"""FastAPI backend and the small, same-origin frontend."""

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from scraper import BookError, scrape_book
from text_features import prepare_text

APP_FOLDER = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app):
    artifact_folder = APP_FOLDER / "artifacts"
    metadata_file = artifact_folder / "metadata.json"
    if not metadata_file.exists():
        raise RuntimeError("Models are missing. Run: python train_models.py")
    app.state.metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    app.state.models = {}
    for model_info in app.state.metadata["models"]:
        feature = model_info["feature"]
        # Only load artifacts generated locally by train_models.py.
        app.state.models[feature] = joblib.load(artifact_folder / f"{feature}.joblib")
    app.state.train_urls = set(app.state.metadata["train_urls"])
    app.state.test_urls = set(app.state.metadata["test_urls"])
    yield


app = FastAPI(title="Book Category Explorer", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=APP_FOLDER / "static"), name="static")


class PredictionRequest(BaseModel):
    book_url: str = Field(min_length=1, max_length=2048)
    feature: Literal["title", "description", "title_description"]


@app.get("/")
def home():
    return FileResponse(APP_FOLDER / "static/index.html")


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": len(app.state.models)}


@app.get("/api/models")
def list_models():
    metadata = app.state.metadata
    return {
        "models": metadata["models"],
        "training_rows": metadata["training_rows"],
        "test_rows": metadata["test_rows"],
        "selection_metric": metadata["selection_metric"],
    }


@app.post("/api/predict")
def predict(request: PredictionRequest):
    try:
        book = scrape_book(request.book_url)
    except BookError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error

    if request.feature != "title" and not book["description"]:
        raise HTTPException(status_code=422, detail="This book has no description. Try Title only.")

    text = prepare_text(book["title"], book["description"], request.feature)
    bundle = app.state.models[request.feature]
    features = bundle["vectorizer"].transform([text])
    prediction = str(bundle["model"].predict(features)[0])
    model_info = next(
        info for info in app.state.metadata["models"] if info["feature"] == request.feature
    )

    warnings = []
    if book["book_url"] in app.state.train_urls:
        dataset_split = "training"
        warnings.append(
            "This book was used to train the model. Its prediction is not an independent test."
        )
    elif book["book_url"] in app.state.test_urls:
        dataset_split = "test"
        warnings.append("This book belongs to the original test set used in the reported results.")
    else:
        dataset_split = "outside_dataset"
        warnings.append(
            "This book was not in the cleaned project dataset. The reported scores do not measure performance on it."
        )
    if book["site_category"] not in {"Fiction", "Nonfiction"}:
        warnings.append(
            "The model was trained only on the website's Fiction and Nonfiction sections. This book is outside those sections."
        )
    if features.nnz == 0:
        warnings.append(
            "None of this text's words matched the trained vocabulary. Treat this prediction cautiously."
        )

    return {
        "book": book,
        "prediction": prediction,
        "feature": request.feature,
        "model": model_info["model"],
        "test_accuracy": model_info["accuracy"],
        "macro_f1": model_info["macro_f1"],
        "dataset_split": dataset_split,
        "warnings": warnings,
    }
