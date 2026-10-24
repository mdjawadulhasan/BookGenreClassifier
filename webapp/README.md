# Book Category Explorer

A small FastAPI app with a plain HTML, CSS, and JavaScript frontend. Paste a book
page from Books to Scrape and choose title, description, or both. The backend
fetches the page live and predicts Fiction or Nonfiction.

## Run locally

The local `.venv` is already prepared. In a VS Code terminal:

```sh
cd "/Users/mdjawadulhasan/MS/Period 1/DS/books_project/webapp"
source .venv/bin/activate
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. Stop the server with Control+C.
Interactive API documentation is at http://127.0.0.1:8000/docs.

If port 8000 is occupied, use `--port 8001` and open http://127.0.0.1:8001.

To set up on another machine, use Python 3.12:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

The local environment reuses the existing course environment for scientific
packages. `requirements.txt` supplies those dependencies for a fresh installation.
The notebook environment and notebooks are unchanged.

## Selected models

Selection follows the report's accuracy comparisons, with macro F1 as a
tiebreaker. An exact tie uses the first model in notebook order.

| Text feature | Exported model | Test accuracy | Macro F1 |
| :--- | :--- | ---: | ---: |
| Title | Naive Bayes | 65.71% | 0.4643 |
| Description | Random Forest | 82.86% | 0.7900 |
| Title + description | Random Forest | 85.71% | 0.8381 |

Naive Bayes, Logistic Regression, and SVM tie for title accuracy and macro F1.
Naive Bayes is used for simplicity. If selection used macro F1 alone, the title
winner would instead be Random Forest. The app states its accuracy selection rule.

These are the original 35-book test results, not confidence scores or independent
validation after model selection. The exported models retain the 138-book training
split. They are not refitted on all 173 books. The API identifies training books,
original test books, and books outside the cleaned dataset. Because the source
website is the same as the training source, many pasted books will be training books.

Only the website's exact Fiction and Nonfiction categories were used for training.
Other sections can be predicted, but the API marks them as outside the training
scope. The model cannot reject an unfamiliar category automatically.

## How the code relates to the notebooks

- `scraper.py`: uses `requests.get`, `BeautifulSoup(..., 'html.parser')`,
  `soup.find('div', id='product_description')`, and `find_next('p')`, matching
  `01_scraping.ipynb`. The title comes from the single book page's `h1`.
  URL validation, timeouts, response size limits, and error handling are added
  because visitors can submit links.
- `text_features.py`: copies the whitespace, trailing `...more`, and repeated
  80-character prefix cleanup from `02_cleaning.ipynb`. The combined feature uses
  the same `Title: ... Description: ...` format as `04_feature_engineering.ipynb`.
- `train_models.py`: loads `../data/cleaned/books_cleaned.csv`, recreates the
  stratified 80/20 split with seed 42, and compares the same four classifiers for
  each feature. TF-IDF parameters and classifier settings match the notebooks.
  Only each winning trained classifier and its fitted TF-IDF vectorizer are saved.
- `main.py`: loads those saved artifacts once, serves the frontend, and predicts.
  It never trains while handling requests. The scraped website category is returned
  for comparison only and is never used as a model feature. Cover images are for
  display only. Missing descriptions still allow title predictions.
- `artifacts/metadata.json`: records the dataset hash, split membership, all
  comparison scores, selection rule, and scikit-learn version. Do not load model
  files from untrusted sources; joblib artifacts contain executable Python objects.

`Soumission` and the title-only Kindle description remain excluded from training,
as in the cleaned dataset. If pasted into the app, the API marks them as outside that
dataset. The app does not add language detection, so non-English inputs may be
unreliable. A prediction with no matching TF-IDF words receives an API warning.
The frontend shows the prediction and website category; dataset membership,
warnings, the cleaned description, and explanatory notes remain hidden from the UI.

## Retrain after changing your data

From this folder, with the environment activated:

```sh
python train_models.py
```

Restart the server after exporting new artifacts. Training needs the parent
project's cleaned CSV. Inference only needs this `webapp` folder and its artifacts.

## Verify

```sh
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
node --check static/app.js
```

Tests cover saved-model scores, cleanup parity with the cleaned data, API
predictions, invalid URLs, redirects, missing content, upstream errors, and
dataset membership. Tests use mocked HTTP responses; live website checks are
performed separately. Tests require the parent project's raw and cleaned CSVs.

## Azure preparation

Nothing has been deployed and no Azure resources have been created.

Use this `webapp` folder as the deployment root. Include `artifacts/`, `static/`,
the Python files, `requirements.txt`, and `startup.sh`. Exclude `.venv` and caches.
Use a Python 3.12 Linux runtime to match the tested scientific packages and
serialized models. Keep scikit-learn at 1.6.1 unless you retrain with a new version.

For Azure App Service, a custom startup command is supplied:

```sh
sh startup.sh
```

The script binds Uvicorn to `0.0.0.0` and uses the `PORT` environment variable,
defaulting to 8000. `/health` is available for a health check. A Dockerfile is also
included for a later container deployment:

```sh
docker build -t book-category-explorer .
docker run --rm -p 8000:8000 book-category-explorer
```

The Docker build is optional and has not been executed as part of the local setup.
No Azure credentials are required for local testing.

Reference: [Microsoft's Python App Service deployment guide](https://learn.microsoft.com/en-us/azure/app-service/quickstart-python)
and [FastAPI static file documentation](https://fastapi.tiangolo.com/tutorial/static-files/).
