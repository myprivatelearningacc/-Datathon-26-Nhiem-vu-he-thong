import pandas as pd
import os

order_items = pd.read_csv("data/order_items.csv", dtype={"promo_id_2": "string"})
products = pd.read_csv("data/products.csv")
orders = pd.read_csv("data/orders.csv")
customers = pd.read_csv("data/customers.csv")
geography = pd.read_csv("data/geography.csv")

orders["order_date"] = pd.to_datetime(orders["order_date"], errors="coerce")
customers["signup_date"] = pd.to_datetime(customers["signup_date"], errors="coerce")

# Rename zip rõ nghĩa
orders = orders.rename(columns={"zip": "shipping_zip"})
customers = customers.rename(columns={"zip": "customer_zip", "city": "customer_city"})
geography = geography.rename(columns={"zip": "shipping_zip", "city": "shipping_city"})

df_1 = order_items.merge(products, on="product_id", how="left")
df_1 = df_1.merge(orders, on="order_id", how="left")
df_1 = df_1.merge(customers, on="customer_id", how="left")
df_1 = df_1.merge(geography, on="shipping_zip", how="left")

# Metrics cho trục 1
df_1["line_revenue"] = df_1["quantity"] * df_1["unit_price"]
df_1["line_cogs"] = df_1["quantity"] * df_1["cogs"]
df_1["gross_profit"] = df_1["line_revenue"] - df_1["line_cogs"]
df_1["gross_margin"] = df_1["gross_profit"] / df_1["line_revenue"]

output_folder = "output_data"
os.makedirs(output_folder, exist_ok=True)
df_1.to_csv(f"{output_folder}/df_1_FIXED.csv", index=False, encoding="utf-8-sig")

print(df_1.shape)
print(df_1["order_date"].min(), df_1["order_date"].max())
print(df_1["order_date"].dt.year.value_counts().sort_index())