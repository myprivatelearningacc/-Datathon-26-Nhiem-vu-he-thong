import pandas as pd
import os

reviews = pd.read_csv("data/reviews.csv")
orders = pd.read_csv("data/orders.csv")
products = pd.read_csv("data/products.csv")
shipments = pd.read_csv("data/shipments.csv")



df_7 = reviews.merge(orders, on="order_id", how="left")
df_7 = df_7.merge(shipments, on ="order_id", how="left")
df_7 = df_7.merge(products, on="product_id", how="left")


print(df_7.dtypes)
print(df_7.shape)
print(df_7.head())
output_folder = "output_data"
os.makedirs(output_folder, exist_ok=True)

df_7.to_csv(f"{output_folder}/df_7.csv", index=False, encoding="utf-8-sig")