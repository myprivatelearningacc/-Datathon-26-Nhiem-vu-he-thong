import pandas as pd
import os

web_traffic = pd.read_csv("data/web_traffic.csv")
orders = pd.read_csv("data/orders.csv")

web_traffic["date"] = pd.to_datetime(web_traffic["date"])
orders["order_date"] = pd.to_datetime(orders["order_date"])

# Check source names first
print("Traffic sources:")
print(web_traffic["traffic_source"].value_counts(dropna=False))

print("\nOrder sources:")
print(orders["order_source"].value_counts(dropna=False))

# Aggregate traffic by date and traffic source
traffic_source_daily = web_traffic.groupby(["date", "traffic_source"]).agg(
    sessions=("sessions", "sum"),
    unique_visitors=("unique_visitors", "sum"),
    page_views=("page_views", "sum"),
    bounce_rate=("bounce_rate", "mean"),
    avg_session_duration_sec=("avg_session_duration_sec", "mean")
).reset_index()

# Aggregate orders by date and order source
orders_source_daily = orders.groupby(["order_date", "order_source"]).agg(
    num_orders=("order_id", "count"),
    num_customers=("customer_id", "nunique")
).reset_index()

orders_source_daily = orders_source_daily.rename(columns={
    "order_date": "date",
    "order_source": "traffic_source"
})

# Merge by both date and source
df_6_channel = traffic_source_daily.merge(
    orders_source_daily,
    on=["date", "traffic_source"],
    how="left"
)

df_6_channel["num_orders"] = df_6_channel["num_orders"].fillna(0)
df_6_channel["num_customers"] = df_6_channel["num_customers"].fillna(0)

df_6_channel["conversion_rate"] = df_6_channel["num_orders"] / df_6_channel["sessions"]
df_6_channel["orders_per_1000_sessions"] = df_6_channel["conversion_rate"] * 1000

output_folder = "output_data"
os.makedirs(output_folder, exist_ok=True)

df_6_channel.to_csv(f"{output_folder}/df_6_channel.csv", index=False, encoding="utf-8-sig")

print(df_6_channel.dtypes)
print(df_6_channel.shape)
print(df_6_channel.head())