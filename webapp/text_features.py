"""The same text cleanup and combined feature used in the notebooks."""

import re


def clean_title(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def clean_description(text):
    text = clean_title(text)
    text = re.sub(r"\s*\.\.\.more$", "", text)

    # From remove_duplicate_text in 02_cleaning.ipynb.
    prefix = text[:80]
    if prefix:
        repeated_at = text.find(prefix, len(prefix))
        if repeated_at != -1:
            text = text[repeated_at:]

    return text.strip()


def prepare_text(title, description, feature):
    if feature == "title":
        return title
    if feature == "description":
        return description
    if feature == "title_description":
        return "Title: " + title + " Description: " + description
    raise ValueError("Unknown feature option.")
