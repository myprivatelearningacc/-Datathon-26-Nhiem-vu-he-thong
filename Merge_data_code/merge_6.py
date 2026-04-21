import pandas as pd
import os

web_traffic = pd.read_csv("data/web_traffic.csv")
orders = pd.read_csv("data/orders.csv")

web_traffic["date"] = pd.to_datetime(web_traffic["date"])
orders["order_date"] = pd.to_datetime(orders["order_date"])


orders_daily = orders.groupby("order_date").agg(
    num_orders=("order_id", "count"),
    num_customers=("customer_id", "nunique")
).reset_index()


orders_daily = orders_daily.rename(columns={"order_date": "date"})


df_6 = web_traffic.merge(orders_daily, on="date", how="left")


df_6["num_orders"] = df_6["num_orders"].fillna(0)
df_6["num_customers"] = df_6["num_customers"].fillna(0)





print(df_6.dtypes)
print(df_6.shape)
print(df_6.head())
output_folder = "output_data"
os.makedirs(output_folder, exist_ok=True)

df_6.to_csv(f"{output_folder}/df_6.csv", index=False, encoding="utf-8-sig")