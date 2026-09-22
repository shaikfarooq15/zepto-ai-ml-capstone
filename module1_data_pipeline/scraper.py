import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3


# Convert star ratings from words to numbers
rating_map = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}


# Store all scraped books here
books = []


# Categories we want to scrape
categories = [
    ("Travel", "https://books.toscrape.com/catalogue/category/books/travel_2/index.html"),
    ("Mystery", "https://books.toscrape.com/catalogue/category/books/mystery_3/index.html"),
    ("Historical Fiction", "https://books.toscrape.com/catalogue/category/books/historical-fiction_4/index.html"),
    ("Fiction", "https://books.toscrape.com/catalogue/category/books/fiction_10/index.html")
]


# -----------------------------
# 1. SCRAPE THE WEBSITE
# -----------------------------

for category_name, category_url in categories:

    print("Scraping:", category_name)

    response = requests.get(category_url)
    response.encoding = "utf-8"

    soup = BeautifulSoup(response.text, "html.parser")

    book_list = soup.select("article.product_pod")

    for book in book_list:

        title = book.h3.a["title"]

        price = book.select_one(".price_color").text.strip()
        price = float(price.replace("£", ""))

        rating = book.select_one("p.star-rating")["class"][1]
        rating = rating_map[rating]

        availability = book.select_one(".availability").get_text(strip=True)

        if "In stock" in availability:
            in_stock = True
        else:
            in_stock = False

        books.append({
            "title": title,
            "price_gbp": price,
            "rating": rating,
            "in_stock": in_stock,
            "category": category_name
        })


# -----------------------------
# 2. CHECK SCRAPED DATA
# -----------------------------

print("\nTotal books collected:", len(books))

print("\nFirst book:")
print(books[0])


# -----------------------------
# 3. CREATE PANDAS DATAFRAME
# -----------------------------

df = pd.DataFrame(books)

print("\nDataFrame:")
print(df.head())

print("\nData types:")
print(df.dtypes)


# -----------------------------
# 4. CONVERT GBP TO INR
# -----------------------------

GBP_TO_INR = 115

df["price_inr"] = df["price_gbp"] * GBP_TO_INR

print("\nPrice conversion:")
print(df[["title", "price_gbp", "price_inr"]].head())


# -----------------------------
# 5. CREATE SQLITE DATABASE
# -----------------------------

conn = sqlite3.connect("data/books.db")

cursor = conn.cursor()


# Remove old tables so we don't create duplicate books
cursor.execute("DROP TABLE IF EXISTS Books")
cursor.execute("DROP TABLE IF EXISTS Categories")


# -----------------------------
# 6. CREATE CATEGORIES TABLE
# -----------------------------

cursor.execute("""
CREATE TABLE Categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT UNIQUE
)
""")


# Insert categories
for category in df["category"].unique():

    cursor.execute(
        """
        INSERT INTO Categories (category_name)
        VALUES (?)
        """,
        (category,)
    )


# -----------------------------
# 7. CREATE BOOKS TABLE
# -----------------------------

cursor.execute("""
CREATE TABLE Books (
    book_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    price_gbp REAL,
    price_inr REAL,
    rating INTEGER,
    in_stock BOOLEAN,
    category_id INTEGER,
    FOREIGN KEY (category_id)
        REFERENCES Categories(category_id)
)
""")


# -----------------------------
# 8. INSERT BOOKS
# -----------------------------

for _, row in df.iterrows():

    cursor.execute(
        """
        SELECT category_id
        FROM Categories
        WHERE category_name = ?
        """,
        (row["category"],)
    )

    category_id = cursor.fetchone()[0]

    cursor.execute(
        """
        INSERT INTO Books
        (title, price_gbp, price_inr, rating, in_stock, category_id)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            row["title"],
            row["price_gbp"],
            row["price_inr"],
            row["rating"],
            row["in_stock"],
            category_id
        )
    )


# Save everything
conn.commit()

conn.close()


print("\nSQLite database created successfully!")
print("Database: data/books.db")