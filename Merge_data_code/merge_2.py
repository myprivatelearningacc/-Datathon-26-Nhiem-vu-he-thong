import pandas as pd
import os

order_items = pd.read_csv("data/order_items.csv", dtype={"promo_id_2": "string"})
orders = pd.read_csv("data/orders.csv")
customers = pd.read_csv("data/customers.csv")


# Merge step by step
df_2 = order_items.merge(orders, on="order_id", how="left")
df_2= df_2.merge(customers, on="customer_id", how="left")



print(df_2.dtypes)
print(df_2.shape)
print(df_2.head())
output_folder = "output_data"
os.makedirs(output_folder, exist_ok=True)

df_2.to_csv(f"{output_folder}/df_2.csv", index=False, encoding="utf-8-sig")