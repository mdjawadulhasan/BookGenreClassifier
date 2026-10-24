from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
import requests
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

import main
import scraper
from scraper import BookError, parse_book, validate_book_url
from text_features import clean_description, clean_title, prepare_text

APP_FOLDER = Path(__file__).resolve().parents[1]
BOOK_URL = "https://books.toscrape.com/catalogue/private-paris-private-10_958/index.html"
HTML = b"""
<ul class="breadcrumb"><li><a>Home</a></li><li><a>Books</a></li>
<li><a>Fiction</a></li><li>Private Paris</li></ul>
<div class="product_main"><h1>Private Paris (Private #10)</h1></div>
<div id="product_description"><h2>Product Description</h2></div>
<p> A story of family, love and adventure. ...more</p>
<div id="product_gallery"><img src="../../media/cache/cover.jpg"></div>
"""


@pytest.fixture
def client():
    with TestClient(main.app) as test_client:
        yield test_client


def mock_response(monkeypatch, status=200, content=HTML):
    response = MagicMock()
    response.status_code = status
    response.headers = {"content-type": "text/html; charset=utf-8"}
    response.content = content
    if status >= 400:
        response.raise_for_status.side_effect = requests.HTTPError()
    response.__enter__.return_value = response
    get = MagicMock(return_value=response)
    monkeypatch.setattr(scraper.requests, "get", get)
    return get


def test_static_frontend_and_model_options(client):
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200
    assert client.get("/health").json() == {"status": "ok", "models_loaded": 3}
    data = client.get("/api/models").json()
    assert [model["model"] for model in data["models"]] == [
        "MultinomialNB",
        "RandomForestClassifier",
        "RandomForestClassifier",
    ]
    assert "train_urls" not in data


@pytest.mark.parametrize("feature", ["title", "description", "title_description"])
def test_scrape_to_prediction(client, monkeypatch, feature):
    get = mock_response(monkeypatch)
    response = client.post("/api/predict", json={"book_url": BOOK_URL, "feature": feature})
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in {"Fiction", "Nonfiction"}
    assert data["feature"] == feature
    assert data["book"]["title"] == "Private Paris (Private #10)"
    assert data["book"]["description"] == "A story of family, love and adventure."
    assert data["book"]["site_category"] == "Fiction"
    assert "dataset_split" not in data
    assert "warnings" not in data
    assert get.call_args.kwargs["allow_redirects"] is False
    assert get.call_args.kwargs["timeout"] == (5, 15)


def test_unknown_words_still_use_the_saved_model(client, monkeypatch):
    book = parse_book(HTML, "https://books.toscrape.com/catalogue/new-book_9999/index.html")
    book.update(title="zzzzqqqqxxxx", site_category="Poetry")
    monkeypatch.setattr(main, "scrape_book", lambda url: book)
    data = client.post("/api/predict", json={"book_url": BOOK_URL, "feature": "title"}).json()
    bundle = main.app.state.models["title"]
    features = bundle["vectorizer"].transform([book["title"]])
    assert features.nnz == 0
    assert data["prediction"] == bundle["model"].predict(features)[0]


def test_startup_and_predictions_do_not_train(monkeypatch):
    def no_training(*args, **kwargs):
        raise AssertionError("The web app must not train models or fit TF-IDF.")

    for estimator in [MultinomialNB, RandomForestClassifier, TfidfVectorizer]:
        monkeypatch.setattr(estimator, "fit", no_training)
    monkeypatch.setattr(TfidfVectorizer, "fit_transform", no_training)
    mock_response(monkeypatch)
    with TestClient(main.app) as client:
        for feature in main.FEATURES:
            response = client.post("/api/predict", json={"book_url": BOOK_URL, "feature": feature})
            assert response.status_code == 200


def test_category_is_not_used_for_prediction(client, monkeypatch):
    book = parse_book(HTML, BOOK_URL)
    monkeypatch.setattr(main, "scrape_book", lambda url: book.copy())
    payload = {"book_url": BOOK_URL, "feature": "title_description"}
    first = client.post("/api/predict", json=payload).json()["prediction"]
    book["site_category"] = "Nonfiction"
    second = client.post("/api/predict", json=payload).json()["prediction"]
    assert first == second


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/catalogue/book_1/index.html",
        "http://127.0.0.1/catalogue/book_1/index.html",
        "https://books.toscrape.com.evil.test/catalogue/book_1/index.html",
        "https://books.toscrape.com@evil.test/catalogue/book_1/index.html",
        "https://user:pass@books.toscrape.com/catalogue/book_1/index.html",
        "https://books.toscrape.com:8080/catalogue/book_1/index.html",
        "file:///etc/passwd",
        "https://books.toscrape.com/catalogue/category/books/fiction_10/index.html",
        "https://books.toscrape.com/catalogue/book_1/index.html?redirect=https://example.com",
        "https://books.toscrape.com/catalogue/../book_1/index.html",
        "https://books.toscrape.com/catalogue/%2e%2e_1/index.html",
        "https://books.toscrape.com:invalid/catalogue/book_1/index.html",
    ],
)
def test_invalid_urls_do_not_make_network_requests(client, monkeypatch, url):
    get = mock_response(monkeypatch)
    response = client.post("/api/predict", json={"book_url": url, "feature": "title"})
    assert response.status_code == 422
    get.assert_not_called()


def test_http_link_is_upgraded_to_https():
    assert validate_book_url(BOOK_URL.replace("https:", "http:")) == BOOK_URL


@pytest.mark.parametrize("feature", ["image", "", None])
def test_unknown_feature_rejected_before_scraping(client, monkeypatch, feature):
    get = mock_response(monkeypatch)
    assert (
        client.post("/api/predict", json={"book_url": BOOK_URL, "feature": feature}).status_code
        == 422
    )
    get.assert_not_called()


def test_missing_description_allows_title_only(client, monkeypatch):
    mock_response(monkeypatch, content=b'<div class="product_main"><h1>Book</h1></div>')
    for feature in ["description", "title_description"]:
        assert (
            client.post("/api/predict", json={"book_url": BOOK_URL, "feature": feature}).status_code
            == 422
        )
    assert (
        client.post("/api/predict", json={"book_url": BOOK_URL, "feature": "title"}).status_code
        == 200
    )


@pytest.mark.parametrize("site_status, expected", [(302, 422), (404, 404), (500, 502)])
def test_upstream_errors(client, monkeypatch, site_status, expected):
    get = mock_response(monkeypatch, status=site_status)
    assert (
        client.post("/api/predict", json={"book_url": BOOK_URL, "feature": "title"}).status_code
        == expected
    )
    assert get.call_count == 1


@pytest.mark.parametrize(
    "error, expected", [(requests.Timeout(), 504), (requests.ConnectionError(), 502)]
)
def test_network_errors(client, monkeypatch, error, expected):
    monkeypatch.setattr(scraper.requests, "get", MagicMock(side_effect=error))
    assert (
        client.post("/api/predict", json={"book_url": BOOK_URL, "feature": "title"}).status_code
        == expected
    )


def test_oversized_upstream_page(client, monkeypatch):
    mock_response(monkeypatch, content=b"x" * 2_000_001)
    assert (
        client.post("/api/predict", json={"book_url": BOOK_URL, "feature": "title"}).status_code
        == 502
    )


def test_non_book_page_rejected():
    with pytest.raises(BookError):
        parse_book(b"<h1>Not a book</h1>", BOOK_URL)


def test_image_url_cannot_point_to_another_site():
    html = HTML.replace(b"../../media/cache/cover.jpg", b"https://example.com/media/x.jpg")
    assert parse_book(html, BOOK_URL)["image_url"] is None


def test_cleaning_matches_saved_course_data():
    raw = pd.read_csv(APP_FOLDER.parent / "data/raw/books_raw.csv").set_index("book_url")
    cleaned = pd.read_csv(APP_FOLDER.parent / "data/cleaned/books_cleaned.csv")
    for _, row in cleaned.iterrows():
        original = raw.loc[row["book_url"]]
        assert clean_title(original["title"]) == row["title"]
        assert clean_description(original["description"]) == row["description"]
    assert clean_description("") == ""


def test_exported_models_reproduce_notebook_test_scores(client):
    df = pd.read_csv(APP_FOLDER.parent / "data/cleaned/books_cleaned.csv").set_index("book_url")
    _, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["category"])
    expected_accuracy = {"title": 23 / 35, "description": 29 / 35, "title_description": 30 / 35}
    for info in client.get("/api/models").json()["models"]:
        feature = info["feature"]
        bundle = main.app.state.models[feature]
        texts = [
            prepare_text(row["title"], row["description"], feature) for _, row in test.iterrows()
        ]
        predicted = bundle["model"].predict(bundle["vectorizer"].transform(texts))
        assert accuracy_score(test["category"], predicted) == info["accuracy"]
        assert info["accuracy"] == expected_accuracy[feature]
        assert f1_score(test["category"], predicted, average="macro") == info["macro_f1"]
        vocabulary = bundle["vectorizer"].vocabulary_
        assert len(vocabulary) == (58 if feature == "title" else 500)
