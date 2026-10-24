"""Reproduce the notebook comparison and export one winner per feature."""

import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from text_features import prepare_text

APP_FOLDER = Path(__file__).resolve().parent
DATA_FILE = APP_FOLDER.parent / "data/cleaned/books_cleaned.csv"
ARTIFACT_FOLDER = APP_FOLDER / "artifacts"

FEATURES = {
    "title": "Title only",
    "description": "Description only",
    "title_description": "Title + description",
}


def train_models():
    df = pd.read_csv(DATA_FILE)
    required = ["title", "description", "category", "book_url"]
    if df[required].isna().any().any():
        raise ValueError("Cleaned data contains missing required values.")
    if set(df["category"]) != {"Fiction", "Nonfiction"}:
        raise ValueError("Expected the Fiction and Nonfiction categories only.")
    if df["book_url"].duplicated().any():
        raise ValueError("Cleaned data contains duplicate book URLs.")

    train_index, test_index = train_test_split(
        df.index, test_size=0.2, random_state=42, stratify=df["category"]
    )
    ARTIFACT_FOLDER.mkdir(exist_ok=True)
    metadata = {
        "selection_metric": "Test accuracy, then macro F1; first model wins exact ties.",
        "training_policy": "Notebook split retained; no refit on test books.",
        "sklearn_version": sklearn.__version__,
        "dataset_sha256": hashlib.sha256(DATA_FILE.read_bytes()).hexdigest(),
        "dataset_rows": len(df),
        "training_rows": len(train_index),
        "test_rows": len(test_index),
        "train_urls": df.loc[train_index, "book_url"].tolist(),
        "test_urls": df.loc[test_index, "book_url"].tolist(),
        "models": [],
    }

    for feature, label in FEATURES.items():
        x = df.apply(lambda row: prepare_text(row["title"], row["description"], feature), axis=1)
        x_train = x.loc[train_index]
        x_test = x.loc[test_index]
        y_train = df.loc[train_index, "category"]
        y_test = df.loc[test_index, "category"]

        # These parameters match all three current modeling notebooks.
        tfidf_vectorizer = TfidfVectorizer(
            max_features=500,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2),
            stop_words="english",
        )
        x_train_tf = tfidf_vectorizer.fit_transform(x_train)
        x_test_tf = tfidf_vectorizer.transform(x_test)

        models = {
            "Naive Bayes": MultinomialNB(),
            "Logistic Regression": LogisticRegression(max_iter=50, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1),
            "SVM": SVC(kernel="linear", random_state=42),
        }
        results = []
        for name, model in models.items():
            model.fit(x_train_tf, y_train)
            y_pred = model.predict(x_test_tf)
            results.append(
                {
                    "model": name,
                    "accuracy": float(accuracy_score(y_test, y_pred)),
                    "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
                }
            )

        best = max(results, key=lambda result: (result["accuracy"], result["macro_f1"]))
        tied = [
            result["model"]
            for result in results
            if (result["accuracy"], result["macro_f1"]) == (best["accuracy"], best["macro_f1"])
        ]
        note = "Highest test accuracy among the four notebook models."
        if len(tied) > 1:
            note = "Tied with " + ", ".join(tied[1:]) + "; Naive Bayes selected for simplicity."

        joblib.dump(
            {"vectorizer": tfidf_vectorizer, "model": models[best["model"]]},
            ARTIFACT_FOLDER / f"{feature}.joblib",
            compress=3,
        )
        metadata["models"].append(
            {
                "feature": feature,
                "label": label,
                **best,
                "selection_note": note,
                "vocabulary_size": len(tfidf_vectorizer.vocabulary_),
                "comparison": results,
            }
        )
        print(
            f"{label}: {best['model']}, accuracy={best['accuracy']:.4f}, "
            f"macro F1={best['macro_f1']:.4f}"
        )

    (ARTIFACT_FOLDER / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved three models trained on {len(train_index)} books to {ARTIFACT_FOLDER}")


if __name__ == "__main__":
    train_models()
