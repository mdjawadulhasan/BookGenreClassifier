"""FastAPI backend and the small, same-origin frontend."""

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
FEATURES = {
    "title": "Title only",
    "description": "Description only",
    "title_description": "Title + description",
}


@asynccontextmanager
async def lifespan(app):
    artifact_folder = APP_FOLDER / "artifacts"
    app.state.models = {}
    for feature in FEATURES:
        model_file = artifact_folder / f"{feature}.joblib"
        if not model_file.exists():
            raise RuntimeError(f"Missing {model_file.name}. Run the export cell in its notebook.")
        app.state.models[feature] = joblib.load(model_file)
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
    options = []
    for feature, label in FEATURES.items():
        bundle = app.state.models[feature]
        options.append(
            {
                "feature": feature,
                "label": label,
                "model": bundle["model"].__class__.__name__,
                "accuracy": bundle["accuracy"],
                "macro_f1": bundle["macro_f1"],
            }
        )
    return {"models": options}


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
    return {
        "book": book,
        "prediction": prediction,
        "feature": request.feature,
        "model": bundle["model"].__class__.__name__,
    }
