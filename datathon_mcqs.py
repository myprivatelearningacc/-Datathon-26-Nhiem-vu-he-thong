import pandas as pd

#%% 1

orders = pd.read_csv("orders.csv")
orders.head()
orders.columns

orders['order_date'] = pd.to_datetime(orders['order_date'])

# Filter customers with more than one order
order_counts = orders.groupby('customer_id')['order_id'].count()
multi_order_customers = order_counts[order_counts > 1].index

orders_multi = orders[orders['customer_id'].isin(multi_order_customers)].copy()

# Sort by customer and date
orders_multi = orders_multi.sort_values(['customer_id', 'order_date'])

# Calculate inter-order gaps (days between consecutive orders per customer)
orders_multi['prev_date'] = orders_multi.groupby('customer_id')['order_date'].shift(1)
orders_multi['gap_days'] = (orders_multi['order_date'] - orders_multi['prev_date']).dt.days

# Drop first orders (no previous order)
gaps = orders_multi['gap_days'].dropna()

median_gap = gaps.median()
print(f"Số khách hàng có >1 đơn: {len(multi_order_customers)}")
print(f"Tổng số cặp đơn hàng liên tiếp: {len(gaps)}")
print(f"Trung vị inter-order gap: {median_gap:.1f} ngày")

# => C. 144 ngày

#%% 2

products = pd.read_csv("products.csv")
products.head()
products.columns

products['gross_margin'] = (products['price'] - products['cogs']) / products['price']

# Tính trung bình gross margin theo segment
segment_margin = products.groupby('segment')['gross_margin'].mean().sort_values(ascending=False)

print("Tỷ suất lợi nhuận gộp trung bình theo segment:")
print(segment_margin.apply(lambda x: f"{x:.2%}"))
print(f"\nSegment có gross margin cao nhất: '{segment_margin.idxmax()}' ({segment_margin.max():.2%})")

# => D. Standard

#%% 3

returns = pd.read_csv("returns.csv")
returns.head()
returns.columns

# Join returns với products theo product_id
merged = returns.merge(products[['product_id', 'category']], on='product_id', how='inner')

# Lọc category = Streetwear
streetwear_returns = merged[merged['category'] == 'Streetwear']

# Đếm return_reason
top_reason = streetwear_returns['return_reason'].value_counts()

print("Lý do trả hàng trong danh mục Streetwear:")
print(top_reason)
print(f"\nLý do phổ biến nhất: '{top_reason.idxmax()}' ({top_reason.max()} lần)")

# => B. wrong_size

#%% 4

web_traf = pd.read_csv("web_traffic.csv")
web_traf.head()
web_traf.columns

avg_bounce = web_traf.groupby('traffic_source')['bounce_rate'].mean().sort_values()

print("Tỷ lệ thoát trung bình theo traffic_source:")
print(avg_bounce.apply(lambda x: f"{x:.4f}"))
print(f"\nNguồn có bounce_rate thấp nhất: '{avg_bounce.idxmin()}' ({avg_bounce.min():.4f})")

# => C. email_campaign

#%% 5

order_items = pd.read_csv("order_items.csv")
order_items.head()
order_items.columns

total = len(order_items)
has_promo = order_items['promo_id'].notna().sum()
pct = has_promo / total * 100

print(f"Tổng số dòng       : {total}")
print(f"Dòng có promo_id   : {has_promo}")
print(f"\nTỷ lệ có khuyến mãi: {pct:.2f}%")

# => C. 39%

#%% 6

customers = pd.read_csv("customers.csv")
customers.head()
customers.columns

# Lọc customers có age_group không null
customers_valid = customers[customers['age_group'].notna()]

# Đếm số đơn hàng mỗi customer
order_counts = orders.groupby('customer_id')['order_id'].count().reset_index()
order_counts.columns = ['customer_id', 'order_count']

# Join với customers
merged = customers_valid.merge(order_counts, on='customer_id', how='left')
merged['order_count'] = merged['order_count'].fillna(0)  # khách chưa có đơn = 0

# Tính trung bình số đơn theo age_group
avg_orders = merged.groupby('age_group')['order_count'].mean().sort_values(ascending=False)

print("Số đơn hàng trung bình theo age_group:")
print(avg_orders.apply(lambda x: f"{x:.4f}"))
print(f"\nNhóm tuổi có đơn TB cao nhất: '{avg_orders.idxmax()}' ({avg_orders.max():.4f} đơn/khách)")

# => A. 55+

#%% 7

geography = pd.read_csv("geography.csv")
geography.head()
geography.columns

sales = pd.read_csv("sales.csv")
sales.head()
sales.columns

# Tính doanh thu mỗi order từ order_items
order_items['revenue'] = order_items['unit_price'] * order_items['quantity'] - order_items['discount_amount']
order_revenue = order_items.groupby('order_id')['revenue'].sum().reset_index()

# Join orders -> order_revenue -> geography
merged = orders[['order_id', 'zip']].merge(order_revenue, on='order_id', how='left') \
                                     .merge(geography[['zip', 'region']], on='zip', how='left')

# Tổng doanh thu theo region
revenue_by_region = merged.groupby('region')['revenue'].sum().sort_values(ascending=False)

print("Tổng doanh thu theo region:")
print(revenue_by_region.apply(lambda x: f"{x:,.0f}"))
print(f"\nRegion có doanh thu cao nhất: '{revenue_by_region.idxmax()}' ({revenue_by_region.max():,.0f})")

# => C. East

#%% 8

# Lọc đơn hàng bị huỷ
cancelled = orders[orders['order_status'] == 'cancelled']

# Đếm payment_method
top_payment = cancelled['payment_method'].value_counts()

print("Phương thức thanh toán trong đơn huỷ:")
print(top_payment)
print(f"\nPhương thức nhiều nhất: '{top_payment.idxmax()}' ({top_payment.max()} đơn)")

# => A. credit_card

#%% 9

sizes = ['S', 'M', 'L', 'XL']

# Join order_items với products để lấy size
oi_size = order_items.merge(products[['product_id', 'size']], on='product_id', how='left')
oi_size = oi_size[oi_size['size'].isin(sizes)]

# Đếm số dòng order_items theo size
total_by_size = oi_size.groupby('size')['order_id'].count()

# Join returns với products để lấy size
ret_size = returns.merge(products[['product_id', 'size']], on='product_id', how='left')
ret_size = ret_size[ret_size['size'].isin(sizes)]

# Đếm số bản ghi returns theo size
returns_by_size = ret_size.groupby('size')['return_id'].count()

# Tỷ lệ trả hàng = returns / order_items
return_rate = (returns_by_size / total_by_size).sort_values(ascending=False)

print("Tỷ lệ trả hàng theo kích thước:")
for size, rate in return_rate.items():
    print(f"  {size}: {rate:.4f} ({rate:.2%})")
print(f"\nKích thước có tỷ lệ trả hàng cao nhất: '{return_rate.idxmax()}' ({return_rate.max():.2%})")

# => A. S

#%% 10

payments = pd.read_csv("payments.csv")
payments.head()
payments.columns

# Tính tổng payment_value mỗi order
order_total = payments.groupby(['order_id', 'installments'])['payment_value'].sum().reset_index()

# Tính trung bình payment_value theo installments
avg_by_installment = order_total.groupby('installments')['payment_value'].mean().sort_values(ascending=False)

print("Giá trị thanh toán TB theo kế hoạch trả góp:")
print(avg_by_installment.apply(lambda x: f"{x:,.2f}"))
print(f"\nKế hoạch trả góp có giá trị TB cao nhất: '{avg_by_installment.idxmax()}' kỳ ({avg_by_installment.max():,.2f})")

# => C. 6 kỳ