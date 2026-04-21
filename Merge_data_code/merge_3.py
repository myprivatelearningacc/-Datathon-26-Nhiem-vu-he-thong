import pandas as pd
import os

order_items = pd.read_csv("data/order_items.csv", dtype={"promo_id_2": "string"})
promotions = pd.read_csv("data/promotions.csv")
products = pd.read_csv("data/products.csv")
orders = pd.read_csv("data/orders.csv")

# Merge step by step
df_3 = order_items.merge(orders, on="order_id", how="left")
df_3= df_3.merge(promotions, on="promo_id", how="left")
df_3 = df_3.merge(products, on="product_id", how="left")



print(df_3.dtypes)
print(df_3.shape)
print(df_3.head())
output_folder = "output_data"
os.makedirs(output_folder, exist_ok=True)

df_3.to_csv(f"{output_folder}/df_3.csv", index=False, encoding="utf-8-sig")