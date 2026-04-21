import pandas as pd
import os

order_items = pd.read_csv("data/order_items.csv", dtype={"promo_id_2": "string"})
products = pd.read_csv("data/products.csv")
orders = pd.read_csv("data/orders.csv")
customers = pd.read_csv("data/customers.csv")
geography = pd.read_csv("data/geography.csv")

# Merge step by step
df_1 = order_items.merge(products, on="product_id", how="left")
df_1= df_1.merge(orders, on="order_id", how="left")
df_1 = df_1.merge(customers, on="customer_id", how="left")
df_1 = df_1.merge(geography, left_on="zip_x", right_on="zip", how="left")

df_1 = df_1.replace("'", "", regex=True)
df_1 = df_1.drop(columns=["zip_y", "zip_x"])

df_1 = df_1.drop(columns = ["city_y"])
df_1 = df_1.rename(columns={"city_x": "city"})

numeric_cols = ["price", "cogs"]

df_1[numeric_cols] = df_1[numeric_cols].apply(
    pd.to_numeric, errors="coerce"
)

print(df_1.dtypes)
print(df_1.shape)
print(df_1.head())
output_folder = "output_data"
os.makedirs(output_folder, exist_ok=True)

df_1.to_csv(f"{output_folder}/df_1.csv", index=False, encoding="utf-8-sig")