# Book Category Explorer

The app does three things: load your saved models, scrape a book's text, and predict
Fiction or Nonfiction. All model training stays in your notebooks.

## Run locally

```sh
cd "/Users/mdjawadulhasan/MS/Period 1/DS/books_project/webapp"
source .venv/bin/activate
python -m uvicorn main:app
```

Open http://127.0.0.1:8000. Stop with Control+C.

For a fresh installation, create a Python 3.12 environment and install
`requirements.txt`. The local environment is already prepared and reuses the
scientific packages from your course environment.

## Where the models come from

The last cell in each notebook saves its existing trained model and its fitted
TF-IDF vectorizer together as a `.joblib` file:

| Notebook | File read by the app | Current selected model |
| :--- | :--- | :--- |
| `05_modeling_(title).ipynb` | `artifacts/title.joblib` | `nb_model` |
| `05_modeling_(desc).ipynb` | `artifacts/description.joblib` | `rf_model` |
| `05_modeling_(title + desc).ipynb` | `artifacts/title_description.joblib` | `rf_model` |

Each file contains `model`, `vectorizer`, `accuracy`, and `macro_f1`. The two scores
are calculated from that notebook's test predictions for display in the frontend.
The model name is read from the saved classifier itself.

Title Naive Bayes ties with Logistic Regression and SVM for the highest test
accuracy. It remains selected for simplicity. Description and combined features
use Random Forest. These selections follow test accuracy, as in the report.

The current test accuracies are 65.71%, 82.86%, and 85.71%, respectively. These
scores come from the 35-book project test split, not from an independent evaluation
after choosing models. Training books can also be entered into this demo.
The model was trained only on the website's exact Fiction and Nonfiction sections.

## Update a model

1. Train and evaluate it in its original notebook.
2. Run the final **Save the selected model for the web application** cell.
   If a different classifier is now best, change `selected_model` in that cell.
3. Restart your local app so it loads the replacement file, then test it.
4. Package the updated app and upload it as a new Elastic Beanstalk version.

There is no training script in this app and no second training step. Always save
the model and its matching fitted vectorizer together. If your text preparation
changes, update `text_features.py` to match. Keep Python at 3.12 and scikit-learn at
1.6.1, or update the runtime and re-export the models together.

## Backend flow

- `main.py` loads the three `.joblib` files once when the server starts.
- `scraper.py` uses your notebook's `requests.get`, Beautiful Soup,
  `soup.find('div', id='product_description')`, and `find_next('p').text` approach.
  The single book page's title is in `h1`, rather than the category listing's `h3`.
  URL checks and request timeouts handle public inputs and website errors.
- `text_features.py` keeps the same whitespace, trailing `...more`, repeated-prefix
  cleanup, and `Title: ... Description: ...` format as your notebooks.
- The API passes the text to the loaded vectorizer's `transform()` and then the
  loaded model's `predict()`. It never calls `fit()` or reads a training dataset.
- The scraped category and cover are for display only. Neither enters the model.

Only load trusted model files. `.joblib` files contain serialized Python objects.

## AWS deployment

Use `webapp` as the Docker build root. The deployment ZIP needs:

- `Dockerfile`, `.dockerignore`, and `requirements.txt`
- `main.py`, `scraper.py`, and `text_features.py`
- `static/` and the three `.joblib` files in `artifacts/`

Do not include `.venv`, notebooks, CSVs, or tests in the deployment ZIP. No notebook
or dataset is needed at runtime. The Dockerfile listens on port 8000. `/health`
returns the number of loaded models.

In Elastic Beanstalk, choose **Upload and deploy**, upload a freshly packaged ZIP,
and use a new version label. The existing CloudFront URL can keep pointing to the
same Elastic Beanstalk environment.

## Checks

```sh
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
node --check static/app.js
```

Tests check predictions with training methods disabled, notebook export scores,
text cleanup, missing content, and invalid or unavailable website URLs. Tests use
mock HTTP responses and the parent project's CSVs; the running app needs neither.
