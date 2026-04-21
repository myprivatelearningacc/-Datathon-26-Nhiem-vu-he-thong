import pandas as pd
import os

inventory = pd.read_csv("data/inventory.csv")
products = pd.read_csv("data/oproducts.csv")


# Merge step by step
df_4_2 = inventory.merge(products, on="product_id", how="left")




print(df_4_2.dtypes)
print(df_4_2.shape)
print(df_4_2.head())
output_folder = "output_data"
os.makedirs(output_folder, exist_ok=True)

df_4_2.to_csv(f"{output_folder}/df_4_2.csv", index=False, encoding="utf-8-sig")