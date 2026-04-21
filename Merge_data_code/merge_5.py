import pandas as pd
import os

products = pd.read_csv("data/products.csv")
order_items = pd.read_csv("data/order_items.csv", dtype={"promo_id_2": "string"})
returns = pd.read_csv("data/returns.csv")

# Merge step by step
df_5 = order_items.merge(products, on="product_id", how="left")
df_5 = df_5.merge(returns, on="order_id", how="left")

print(df_5.dtypes)
print(df_5.shape)
print(df_5.head())
output_folder = "output_data"
os.makedirs(output_folder, exist_ok=True)

df_5.to_csv(f"{output_folder}/df_5.csv", index=False, encoding="utf-8-sig")