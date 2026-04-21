import pandas as pd
import os

orders = pd.read_csv("data/orders.csv")
shipments = pd.read_csv("data/shipments.csv")
reviews = pd.read_csv("data/reviews.csv")
returns = pd.read_csv("data/returns.csv")

# Merge step by step
df_4 = orders.merge(shipments, on="order_id", how="left")
df_4 = df_4.merge(reviews, on="order_id", how="left")
df_4 = df_4.merge(returns, on="order_id", how="left")



print(df_4.dtypes)
print(df_4.shape)
print(df_4.head())
output_folder = "output_data"
os.makedirs(output_folder, exist_ok=True)

df_4.to_csv(f"{output_folder}/df_4.csv", index=False, encoding="utf-8-sig")