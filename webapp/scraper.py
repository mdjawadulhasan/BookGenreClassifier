"""Scrape a single book using requests and Beautiful Soup, as in 01_scraping."""

import re
from urllib.parse import urljoin, urlsplit

import requests
from bs4 import BeautifulSoup

from text_features import clean_description, clean_title


class BookError(Exception):
    def __init__(self, message, status_code=422):
        super().__init__(message)
        self.status_code = status_code


def validate_book_url(url):
    message = "Enter a book page from books.toscrape.com, ending in /index.html."
    try:
        parts = urlsplit(url.strip())
        if (
            parts.scheme not in {"https", "http"}
            or parts.hostname != "books.toscrape.com"
            or parts.username is not None
            or parts.password is not None
            or parts.port not in {None, 80 if parts.scheme == "http" else 443}
            or parts.query
            or parts.fragment
            or not re.fullmatch(r"/catalogue/[a-zA-Z0-9_-]+_\d+/index\.html", parts.path)
        ):
            raise BookError(message)
    except ValueError as error:
        raise BookError(message) from error
    return "https://books.toscrape.com" + parts.path


def parse_book(html, url):
    soup = BeautifulSoup(html, "html.parser")
    product = soup.find("div", class_="product_main")
    heading = product.find("h1") if product else None
    if heading is None or not heading.get_text(strip=True):
        raise BookError("This page does not contain a book title.")

    title = clean_title(heading.get_text())
    description_heading = soup.find("div", id="product_description")
    paragraph = description_heading.find_next("p") if description_heading else None
    description = clean_description(paragraph.get_text() if paragraph else "")

    # Category is displayed for comparison only. It never enters the model.
    breadcrumb = soup.find("ul", class_="breadcrumb")
    links = breadcrumb.find_all("a") if breadcrumb else []
    site_category = clean_title(links[-1].get_text()) if len(links) >= 3 else None

    cover = soup.find("div", id="product_gallery")
    image = cover.find("img") if cover else None
    image_url = urljoin(url, image.get("src", "")) if image else None
    if image_url:
        parts = urlsplit(image_url)
        if parts.hostname != "books.toscrape.com" or not parts.path.startswith("/media/"):
            image_url = None
        else:
            image_url = "https://books.toscrape.com" + parts.path

    return {
        "title": title,
        "description": description,
        "book_url": url,
        "image_url": image_url,
        "site_category": site_category,
    }


def scrape_book(url):
    url = validate_book_url(url)
    try:
        # Redirects are disabled so a supplied URL cannot redirect to another host.
        with requests.get(url, timeout=(5, 15), allow_redirects=False, stream=True) as response:
            if response.status_code == 404:
                raise BookError("Book not found. Check the link and try again.", 404)
            if 300 <= response.status_code < 400:
                raise BookError("Use the book's direct URL. Redirects are not followed.")
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if "text/html" not in content_type.lower():
                raise BookError("The book website returned an unexpected page type.", 502)
            content = bytearray()
            for chunk in response.iter_content(chunk_size=16384):
                content.extend(chunk)
                if len(content) > 2_000_000:
                    raise BookError("The book page is too large to process.", 502)
    except requests.Timeout as error:
        raise BookError("The book website took too long. Please try again.", 504) from error
    except requests.RequestException as error:
        raise BookError("Could not load the book website. Please try again.", 502) from error
    return parse_book(bytes(content), url)
