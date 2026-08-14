
# Module 1 — Data Pipeline

## Objective

This module implements an end-to-end data pipeline that scrapes book catalogue data, cleans and transforms the data, converts GBP prices to INR using the required fixed project rate, and stores the result in a normalized SQLite relational database.

## Data Source

Data was scraped from:

Books to Scrape — https://books.toscrape.com/

The pipeline scrapes the first five catalogue pages, resulting in 100 books across 29 categories.

## Pipeline

The pipeline follows:

Scrape → Clean → Transform → Normalize → Store → Query → Validate

### Scraping

The `requests` library is used to retrieve webpages and `BeautifulSoup` is used to parse the HTML.

The following fields are scraped:

- title
- price
- star_rating
- availability
- category

### Cleaning

The raw fields are transformed as follows:

- `price` is cleaned and converted to `price_gbp` as a float.
- The pound encoding artifact `Â£` is handled during parsing.
- Text ratings (`One` to `Five`) are mapped to integers from 1 to 5.
- Availability text is converted to the Boolean `in_stock` field.
- Numeric parsing failures are handled using median imputation.
- Rows with unparseable availability are dropped because stock status cannot be reliably inferred.

No rows required imputation or dropping in the final scraped dataset.

### Currency Conversion

The project-defined fixed conversion rate is:

**1 GBP = 105.50 INR**

The `price_inr` column is calculated as:

`price_inr = price_gbp × 105.50`

No external currency API is used.

## Database Design

SQLite is used as the relational database.

The database contains two normalized tables.

### categories

- `category_id` — INTEGER PRIMARY KEY
- `category_name` — TEXT UNIQUE NOT NULL

### books

- `book_id` — INTEGER PRIMARY KEY
- `title` — TEXT NOT NULL
- `price_gbp` — REAL NOT NULL
- `price_inr` — REAL NOT NULL
- `rating` — INTEGER NOT NULL
- `in_stock` — INTEGER NOT NULL
- `category_id` — INTEGER NOT NULL, FOREIGN KEY referencing `categories.category_id`

The category name is stored only in the `categories` table to avoid unnecessary duplication in the normalized `books` table.

## SQL Analysis

The notebook executes six SQL queries demonstrating:

1. SELECT and WHERE
2. ORDER BY
3. LIMIT
4. DISTINCT
5. BETWEEN
6. JOIN

The SQL query strings and their outputs are displayed in the notebook.

## SQL and Pandas Validation

Two query results are explicitly loaded into pandas using `pd.read_sql()`.

The JOIN query is independently reproduced using `pd.merge()`.

The SQL JOIN and pandas merge results are aligned and compared. The equality check returns `True`, confirming equivalent results.

## How to Run

### Requirements

Install the required Python packages:

```bash
pip install requests beautifulsoup4 pandas
