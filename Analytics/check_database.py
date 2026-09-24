import sqlite3
import pandas as pd

conn = sqlite3.connect("data/books.db")

books_df = pd.read_sql("SELECT * FROM Books", conn)

categories_df = pd.read_sql("SELECT * FROM Categories", conn)

merged_df = pd.merge(
    books_df,
    categories_df,
    on="category_id",
    how="inner"
)

print("Pandas JOIN result:")
print(
    merged_df[
        ["title", "price_gbp", "category_name"]
    ].head(10)
)

conn.close()